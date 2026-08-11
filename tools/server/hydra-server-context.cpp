// Hydra A/B extension seam implementation (epic #610).
//
// This file is NOT an independent translation unit. It is #include'd at the
// bottom of server-context.cpp, so it compiles as part of that TU and can reach
// server_context_impl's private members through the friend declaration on
// hydra_engine_extension (and, for hydra_process_task, because it is a member
// of server_context_impl itself). Do NOT add this file to CMakeLists.txt.
//
// WS1: no-op extension hooks (handle_task/pre_loop/on_empty_batch return false),
// so seam mode is behavior-identical to legacy.
// WS2: the HYDRA task dispatch is extracted into server_context_impl::
//       hydra_process_task(), called from BOTH the legacy switch path (via a
//       thin fall-through in process_single_task) and the seam handle_task().
//       Both modes run the exact same method, so A/B parity is by construction;
//       the A/B toggle then proves the seam plumbing (routing + claiming) is
//       behavior-identical to the legacy switch.
// WS3: update_slots() clusters move into pre_loop()/on_empty_batch().

#include "server-hydra-extension.h"

#include "server-task.h"

// ---------------------------------------------------------------------------
// WS2: HYDRA task dispatch — member of server_context_impl, defined in this TU.
// ---------------------------------------------------------------------------
    // epic #610 WS2: Hydra task dispatch, moved out of process_single_task().
    // Same TU (see #include at bottom of server-context.cpp) so member access
    // to server_context_impl privates is available. A switch(task.type) wrapper
    // keeps all internal break/continue semantics identical to the inline code.
    void server_context_impl::hydra_process_task(server_task & task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_HYDRA_STATE_GET:
                {
                    // M1: background serialization thread — inference loop continues during state transfer.
                    // llama_state_seq_get_data reads KV cells for an IDLE sequence; llama_decode
                    // writes cells for ACTIVE sequences only — no memory overlap for different seq IDs.
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_state>();
                    res->id      = task.id;
                    res->id_slot = id_slot;
                    res->op      = HYDRA_OP_STATE_GET;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error      = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }
                    if (slot->is_processing() || slot->hydra_transferring->load()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Snapshot on inference thread (cheap — dry-run serialization, no GPU copies).
                    const size_t state_size = llama_state_seq_get_size(ctx_tgt, slot->id);
                    int actual_n_past = slot->n_prompt_tokens_cache + slot->n_decoded;
                    // Cold prefill: n_prompt_tokens_cache is still 0 so n_decoded (1) dominates.
                    // Use prompt token count instead — matches STATE_META fallback.
                    if (slot->n_prompt_tokens_cache == 0 && slot->prompt.tokens.size() > 0) {
                        actual_n_past = (int)slot->prompt.tokens.size();
                    }
                    res->n_past     = actual_n_past;
                    res->rpc_status = HYDRA_STATUS_OK;
                    // M-Perf.9 #289: surface model identity alongside the state
                    // bytes so the Coordinator can record the model that built
                    // the KV (for cross-model safety on restore). The background
                    // thread that streams the bytes to the socket can mutate
                    // res->state_data freely; the model fields are immutable for
                    // the duration of the response.
                    res->model_alias = model_name;
                    res->model_path  = params_base.model.path;
                    if (model_tgt) {
                        res->tokenizer   = llama_model_get_tokenizer_model(model_tgt);
                        res->model_name  = llama_model_get_display_name(model_tgt);
                        res->model_quant = llama_model_get_quant_label(model_tgt);
                        res->model_capabilities = llama_model_get_capabilities_bitfield(model_tgt);
                    }
                    SRV_INF("hydra: STATE_GET slot=%d n_past=%d state=%.1f MiB — async\n",
                            id_slot, res->n_past, state_size / (1024.0 * 1024.0));

                    slot->hydra_transferring->store(true);

                    // M2: stream directly to socket (zero-copy).
                    // Runs SYNCHRONOUSLY on the inference thread to avoid
                    // concurrent ggml-RPC socket access with llama_decode
                    // on another slot (fixes crash at ggml-rpc.cpp:532).
                    // The coordinator already does Store Put as fire-and-forget
                    // so blocking here only delays slot release, not decode.
                    const int      snap_seq_id = slot->id;
                    llama_context * snap_ctx   = ctx_tgt;
                    // shared_ptr keeps the atomic alive even if the slot is reallocated
                    std::shared_ptr<std::atomic<bool>> flag_ptr = slot->hydra_transferring;
                    const int      hydra_fd    = task.hydra_action.hydra_fd;

                    // Capture prompt tokens for M1 path header (slot is valid on inference thread)
                    const llama_tokens prompt_tokens_get = slot->prompt.tokens.get_text_tokens();
                    const int32_t     n_past_val         = res->n_past;

                    // Snapshot the most recent native checkpoint so STATE_PUT can
                    // register it instead of fabricating one at the final position.
                    // Fabricating at pos_max=n-1 corrupts hybrid/recurrent model
                    // decode because the recurrent state is one token ahead of the
                    // decode resume point — it has already processed the final token.
                    std::vector<uint8_t> snapshot_ckpt;
                    uint8_t hdr_flags = 0x00;
                    int32_t ckpt_pos_min = 0, ckpt_pos_max = 0;
                    int64_t ckpt_n_tokens = 0;
                    if (!slot->prompt.checkpoints.empty()) {
                        hdr_flags |= 0x01;
                        const auto & ckpt = slot->prompt.checkpoints.back();
                        ckpt_pos_min = ckpt.pos_min;
                        ckpt_pos_max = ckpt.pos_max;
                        ckpt_n_tokens = ckpt.n_tokens;

                        const uint64_t tgt_sz = ckpt.data_tgt.size();
                        const uint64_t dft_sz = ckpt.data_dft.size();
                        const size_t ckpt_hdr_sz = 4 + 4 + 8 + 8 + (size_t)tgt_sz + 8 + (size_t)dft_sz;
                        snapshot_ckpt.resize(ckpt_hdr_sz);
                        size_t off = 0;
                        memcpy(snapshot_ckpt.data() + off, &ckpt_pos_min, 4); off += 4;
                        memcpy(snapshot_ckpt.data() + off, &ckpt_pos_max, 4); off += 4;
                        memcpy(snapshot_ckpt.data() + off, &ckpt_n_tokens, 8); off += 8;
                        memcpy(snapshot_ckpt.data() + off, &tgt_sz, 8); off += 8;
                        if (tgt_sz > 0) { memcpy(snapshot_ckpt.data() + off, ckpt.data_tgt.data(), (size_t)tgt_sz); off += (size_t)tgt_sz; }
                        memcpy(snapshot_ckpt.data() + off, &dft_sz, 8); off += 8;
                        if (dft_sz > 0) memcpy(snapshot_ckpt.data() + off, ckpt.data_dft.data(), (size_t)dft_sz);
                    }

                    {
                        SRV_INF("hydra: STATE_GET streaming (fd=%d state=%.1f MiB)\n",
                                hydra_fd, state_size / (1024.0 * 1024.0));
                        if (hydra_fd >= 0) {
                            // M2 path: stream v2 blob (header + checkpoint + GPU state) to fd.
                            // Response header + meta JSON sent first, then v2 header bytes,
                            // then llama_state_seq_get_data_to_fd writes GPU state directly.
                            const size_t n_tok = prompt_tokens_get.size();
                            const uint32_t hdr_n_tok = (uint32_t)n_tok;
                            const uint32_t hdr_n_past = (uint32_t)n_past_val;
                            const uint8_t version_byte = 0x02;
                            const size_t base_hdr_size = 1 + 4 + 4 + n_tok * sizeof(llama_token) + 1;
                            const size_t hdr_size = base_hdr_size + snapshot_ckpt.size();
                            const size_t total_payload = hdr_size + state_size;

                            // Build v2 header buffer
                            std::vector<uint8_t> v2_hdr(hdr_size);
                            {
                                size_t off = 0;
                                memcpy(v2_hdr.data() + off, &version_byte, 1); off += 1;
                                memcpy(v2_hdr.data() + off, &hdr_n_past, 4);    off += 4;
                                memcpy(v2_hdr.data() + off, &hdr_n_tok, 4);     off += 4;
                                memcpy(v2_hdr.data() + off, prompt_tokens_get.data(), n_tok * sizeof(llama_token)); off += n_tok * sizeof(llama_token);
                                memcpy(v2_hdr.data() + off, &hdr_flags, 1);     off += 1;
                                if (!snapshot_ckpt.empty()) {
                                    memcpy(v2_hdr.data() + off, snapshot_ckpt.data(), snapshot_ckpt.size());
                                    off += snapshot_ckpt.size();
                                }
                            }

                            {
                                json meta_j;
                                meta_j["n_past"]     = res->n_past;
                                meta_j["state_size"] = (uint64_t)state_size;
                                if (!res->model_alias.empty()) meta_j["model_alias"] = res->model_alias;
                                if (!res->model_path.empty())  meta_j["model_path"]  = res->model_path;
                                if (!res->tokenizer.empty())   meta_j["tokenizer"]   = res->tokenizer;
                                if (!res->model_name.empty())  meta_j["model_name"]  = res->model_name;
                                if (!res->model_quant.empty()) meta_j["model_quant"] = res->model_quant;
                                if (res->model_capabilities)   meta_j["model_capabilities"] = res->model_capabilities;
                                const std::string meta_str = meta_j.dump();

                                const uint32_t meta_len  = (uint32_t)meta_str.size();
                                const uint64_t payload_l = (uint64_t)total_payload;
                                uint8_t hdr[HYDRA_RES_HEADER_SIZE] = {};
                                hdr[0] = HYDRA_STATUS_OK;
                                hdr[1] = (meta_len)       & 0xFF;
                                hdr[2] = (meta_len >>  8) & 0xFF;
                                hdr[3] = (meta_len >> 16) & 0xFF;
                                memcpy(hdr + 4, &payload_l, 8);
                                hydra_send_all(hydra_fd, hdr,           HYDRA_RES_HEADER_SIZE);
                                hydra_send_all(hydra_fd, meta_str.data(), meta_str.size());

                                // Write v2 blob header before GPU state — STATE_PUT needs tokens + checkpoint
                                hydra_send_all(hydra_fd, v2_hdr.data(), v2_hdr.size());

                                res->header_sent = true; // META + header + v2-hdr before payload
                            }
                             // Stream GPU state to fd (zero-copy from GPU memory)
                             const size_t streamed = llama_state_seq_get_data_to_fd(snap_ctx, snap_seq_id, hydra_fd);
                             if (streamed != state_size) {
                                 // TOCTOU: state size changed between get_size (header already
                                 // promised state_size bytes) and the stream, or the stream
                                 // failed mid-way. The wire framing is now broken — the only
                                 // safe recovery is to kill the connection. Use shutdown(),
                                 // not close(): the RPC connection loop owns the fd and will
                                 // close it when its next read fails; closing here would race
                                 // (double-close / fd-reuse against unrelated threads).
                                 res->rpc_status = HYDRA_STATUS_ERROR;
                                 res->error      = "llama_state_seq_get_data_to_fd streamed " +
                                                   std::to_string(streamed) + " B, expected " +
                                                   std::to_string(state_size) + " B";
                                 ::shutdown(hydra_fd, SHUT_RDWR);
                             } else {
                                 res->streamed_bytes = total_payload;
                             }
                         } else {
                             // M1 path: buffer in memory, RPC thread sends afterwards.
                             // v2 blob format (0x02): [1B version][4B n_past][4B n_tok][n_tok*4B tokens]
                             //   [1B flags (bit 0 = has_checkpoint)]
                             //   [if flags & 0x01: 4B pos_min | 4B pos_max | 8B n_tokens | 8B tgt_sz | tgt_data | 8B dft_sz | dft_data]
                             //   [raw KV state from llama_state_seq_get_data]
                             const size_t n_tok = prompt_tokens_get.size();
                             const uint32_t hdr_n_tok = (uint32_t)n_tok;
                             const uint32_t hdr_n_past = (uint32_t)n_past_val;
                             const uint8_t version_byte = 0x02;
                             const size_t base_hdr_size = 1 + 4 + 4 + n_tok * sizeof(llama_token) + 1; // version + n_past + n_tok + tokens + flags
                             const size_t hdr_size = base_hdr_size + snapshot_ckpt.size();

                              // TOCTOU retry: if another slot grew the state between
                              // get_size (inference thread) and get_data (background thread),
                              // the copy returns 0. Retry up to 3 times with fresh sizing.
                              size_t buf_size = hdr_size + state_size;
                              res->state_data.resize(buf_size);
                              {
                                  size_t off = 0;
                                  memcpy(res->state_data.data() + off, &version_byte, 1); off += 1;
                                  memcpy(res->state_data.data() + off, &hdr_n_past, 4);    off += 4;
                                  memcpy(res->state_data.data() + off, &hdr_n_tok, 4);     off += 4;
                                  memcpy(res->state_data.data() + off, prompt_tokens_get.data(), n_tok * sizeof(llama_token)); off += n_tok * sizeof(llama_token);
                                  memcpy(res->state_data.data() + off, &hdr_flags, 1);     off += 1;
                                  if (!snapshot_ckpt.empty()) {
                                      memcpy(res->state_data.data() + off, snapshot_ckpt.data(), snapshot_ckpt.size());
                                      off += snapshot_ckpt.size();
                                  }
                              }

                              size_t cur_state_size = state_size;
                              size_t copied = 0;
                              int retries = 3;
                              while (retries-- > 0) {
                                  copied = llama_state_seq_get_data(
                                          snap_ctx, res->state_data.data() + hdr_size, cur_state_size, snap_seq_id);
                                  if (copied > 0) break;
                                  // State grew — re-measure and retry
                                  cur_state_size = llama_state_seq_get_size(snap_ctx, snap_seq_id);
                                  res->state_data.resize(hdr_size + cur_state_size);
                              }
                              if (copied == 0) {
                                  res->rpc_status = HYDRA_STATUS_ERROR;
                                  res->error      = "llama_state_get_data failed after 3 retries";
                                  res->state_data.clear();
                              }
                         }
                        flag_ptr->store(false);
                        // M2 streams to fd (streamed_bytes); M1 buffers into state_data.
                        const uint64_t out_bytes = (hydra_fd >= 0)
                                ? res->streamed_bytes
                                : (uint64_t) res->state_data.size();
                        SRV_INF("hydra: STATE_GET done slot=%d rpc_status=%d path=%s bytes=%" PRIu64 "\n",
                                snap_seq_id, res->rpc_status,
                                hydra_fd >= 0 ? "M2-stream" : "M1-buffer", out_bytes);
                        queue_results.send(std::move(res));
                    }

                    // STATE_GET is synchronous — blocks until KV state is fully
                    // streamed to the socket. The coordinator's Store Put is
                    // fire-and-forget, so only slot release is delayed.
                } break;

            case SERVER_TASK_TYPE_HYDRA_STATE_PUT:
                {
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_state>();
                    res->id      = task.id;
                    res->id_slot = id_slot;
                    res->op      = HYDRA_OP_STATE_PUT;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error      = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }
                    if (slot->is_processing() || slot->hydra_transferring->load()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        queue_results.send(std::move(res));
                        break;
                    }

                    // M-Perf.9 #289: populate model identity from resident model.
                    // model_match = true always (infrastructure only; actual KV
                    // validation comes when model identity is embedded in the KV header).
                    res->model_alias = model_name;
                    res->model_path   = params_base.model.path;
                    res->model_match  = true;
                    if (model_tgt) {
                        res->tokenizer   = llama_model_get_tokenizer_model(model_tgt);
                        res->model_name  = llama_model_get_display_name(model_tgt);
                        res->model_quant = llama_model_get_quant_label(model_tgt);
                        res->model_capabilities = llama_model_get_capabilities_bitfield(model_tgt);
                    }

                    // Erase existing checkpoints to avoid collision with restored session state
                    if (task.hydra_action.erase_existing && !slot->prompt.checkpoints.empty()) {
                        SLT_INF(*slot, "erasing %zu existing checkpoints before STATE_PUT restore\n",
                                slot->prompt.checkpoints.size());
                        slot->prompt.checkpoints.clear();
                    }

                    const auto & buf = task.hydra_action.state_data;

                    // Detect v2 blob (0x02 at offset 0) vs legacy format (no version byte).
                    // v2: [1B version=0x02][4B n_past][4B n_tok][n_tok*4B tokens][1B flags][?ckpt?][KV state]
                    // legacy: [4B n_past][4B n_tok][n_tok*4B tokens][KV state]
                    const bool is_v2 = buf.size() >= 1 && buf[0] == 0x02;

                    size_t hdr_offset = 0;
                    int32_t hdr_n_tok = 0;
                    int32_t hdr_n_past = 0;
                    bool has_chkpt = false;
                    int32_t ckpt_pos_min_in = 0, ckpt_pos_max_in = 0;
                    int64_t ckpt_n_tokens_in = 0;
                    std::vector<uint8_t> ckpt_tgt_data, ckpt_dft_data;

                    if (is_v2) {
                        // v2: version at [0], n_past at [1..4], n_tok at [5..8]
                        if (buf.size() >= 9) {
                            memcpy(&hdr_n_past, buf.data() + 1, 4);
                            memcpy(&hdr_n_tok,  buf.data() + 5, 4);
                        }
                        const size_t token_start = 9;
                        const size_t token_end = token_start + (size_t)hdr_n_tok * sizeof(llama_token);
                        hdr_offset = token_end;
                        if (hdr_offset < buf.size()) {
                            const uint8_t flags = buf[hdr_offset];
                            hdr_offset += 1; // past flags byte
                            if (flags & 0x01) {
                                // Parse checkpoint: 4B pos_min | 4B pos_max | 8B n_tokens | 8B tgt_sz | tgt_data | 8B dft_sz | dft_data
                                if (hdr_offset + 4 + 4 + 8 + 8 <= buf.size()) {
                                    memcpy(&ckpt_pos_min_in, buf.data() + hdr_offset, 4); hdr_offset += 4;
                                    memcpy(&ckpt_pos_max_in, buf.data() + hdr_offset, 4); hdr_offset += 4;
                                    memcpy(&ckpt_n_tokens_in, buf.data() + hdr_offset, 8); hdr_offset += 8;
                                    uint64_t tgt_sz_in;
                                    memcpy(&tgt_sz_in, buf.data() + hdr_offset, 8); hdr_offset += 8;
                                    if (tgt_sz_in > 0 && hdr_offset + tgt_sz_in <= buf.size()) {
                                        ckpt_tgt_data.assign(buf.data() + hdr_offset, buf.data() + hdr_offset + (size_t)tgt_sz_in);
                                        hdr_offset += (size_t)tgt_sz_in;
                                    }
                                    if (hdr_offset + 8 <= buf.size()) {
                                        uint64_t dft_sz_in;
                                        memcpy(&dft_sz_in, buf.data() + hdr_offset, 8); hdr_offset += 8;
                                        if (dft_sz_in > 0 && hdr_offset + dft_sz_in <= buf.size()) {
                                            ckpt_dft_data.assign(buf.data() + hdr_offset, buf.data() + hdr_offset + (size_t)dft_sz_in);
                                            hdr_offset += (size_t)dft_sz_in;
                                        }
                                    }
                                    has_chkpt = true;
                                }
                            }
                        }
                        // Restore tokens from token_start
                        if (hdr_n_tok > 0 && token_start + (size_t)hdr_n_tok * sizeof(llama_token) <= buf.size()) {
                            slot->prompt.tokens.clear();
                            const llama_token * tok_ptr = (const llama_token *)(buf.data() + token_start);
                            llama_tokens restored_tokens(tok_ptr, tok_ptr + (size_t)hdr_n_tok);
                            slot->prompt.tokens.insert(restored_tokens);
                        }
                    } else {
                        // Legacy v1 format
                        if (buf.size() >= 8) {
                            memcpy(&hdr_n_past, buf.data(), 4);
                            memcpy(&hdr_n_tok, buf.data() + 4, 4);
                            hdr_offset = 8 + (size_t)hdr_n_tok * sizeof(llama_token);
                        }
                        if (hdr_offset > 0 && hdr_offset <= buf.size()) {
                            const size_t n_tokens = (size_t)hdr_n_tok;
                            slot->prompt.tokens.clear();
                            if (n_tokens > 0) {
                                const llama_token * tok_ptr = (const llama_token *)(buf.data() + 8);
                                llama_tokens restored_tokens(tok_ptr, tok_ptr + n_tokens);
                                slot->prompt.tokens.insert(restored_tokens);
                            }
                        }
                    }
                    const bool has_hdr = hdr_offset > 0 && hdr_offset <= buf.size();
                    const uint8_t * state_ptr = has_hdr ? buf.data() + hdr_offset : buf.data();
                    const size_t    state_len = has_hdr ? buf.size() - hdr_offset : buf.size();
                    const size_t n_read = llama_state_seq_set_data(ctx_tgt, state_ptr, state_len, slot->id);
                    if (n_read == 0) {
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->error      = "llama_state_set_data returned 0";
                        // Tokens were registered before set_data — clear them so the slot
                        // is not left poisoned (n_past > 0 with no KV cells → pos_min == -1
                        // abort on the next decode that touches this slot).
                        slot->prompt.tokens.clear();
                        slot->prompt.checkpoints.clear();
                        slot->n_prompt_tokens_cache = 0;
                        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot->id, -1, -1);
                    } else {
                        // D4: Inject trailing logits into per-slot buffer instead of the
                        // shared context-wide llama_get_logits(). This avoids the race where
                        // another slot's decode clobbers restored logits between STATE_PUT
                        // and the first sample.
                        const size_t remaining = state_len - n_read;
                        const size_t expected_logits = (size_t)llama_vocab_n_tokens(vocab) * sizeof(float);
                        if (remaining == expected_logits) {
                            const float * src = (const float *)(state_ptr + n_read);
                            const size_t n_floats = llama_vocab_n_tokens(vocab);
                            slot->restored_logits.assign(src, src + n_floats);
                            slot->logits_valid = true;
                            SRV_INF("hydra: STATE_PUT slot=%d restored %zu logits to per-slot buffer\n",
                                    id_slot, n_floats);
                        }

                        res->rpc_status = HYDRA_STATUS_OK;
                        res->restored   = true;
                        res->bytes      = (uint64_t)n_read;
                        // #469 trace: log restored state for cross-flow comparison
                        SRV_DBG("hydra: STATE_PUT slot=%d RESTORED n_past=%d n_prompt_tok=%d state_bytes=%zu just_restored=true\n",
                                id_slot, hdr_n_tok, hdr_n_tok, n_read);
                        {
                            std::string tok_ids;
                            for (size_t i = 0; i < std::min<size_t>(16, slot->prompt.tokens.size()); ++i) {
                                if (i > 0) tok_ids += ",";
                                tok_ids += std::to_string(slot->prompt.tokens[i]);
                            }
                            SRV_DBG("hydra: STATE_PUT slot=%d first16_tokens=[%s] total=%zu\n",
                                    id_slot, tok_ids.c_str(), slot->prompt.tokens.size());
                        }
                        if (hdr_n_tok > 0) {
                            slot->n_prompt_tokens_cache = hdr_n_tok;
                            slot->n_decoded = 0;
                            res->n_past = hdr_n_tok;

                            // Register native checkpoint from the blob (v2) or fabricate one (legacy).
                            // The native checkpoint has pos_max at n-4 (created before the last
                            // few prompt tokens were decoded), so loading it rewinds the recurrent
                            // state to a clean position. The old fabricated checkpoint at (0, n-1)
                            // puts the recurrent state at the final position — one token ahead of
                            // where decode must resume — corrupting hybrid/recurrent model output.
                            slot->prompt.checkpoints.clear();
                            if (has_chkpt) {
                                auto & ckpt = slot->prompt.checkpoints.emplace_back();
                                ckpt.n_tokens = ckpt_n_tokens_in;
                                ckpt.pos_min  = ckpt_pos_min_in;
                                ckpt.pos_max  = ckpt_pos_max_in;
                                ckpt.data_tgt = std::move(ckpt_tgt_data);
                                ckpt.data_dft = std::move(ckpt_dft_data);
                                SLT_INF(*slot, "STATE_PUT registered native checkpoint (pos_min=%d pos_max=%d n_tokens=%" PRId64 " tgt_sz=%zu)\n",
                                        ckpt.pos_min, ckpt.pos_max, ckpt.n_tokens, ckpt.data_tgt.size());
                            } else {
                                create_checkpoint(*slot, 0, 0, (llama_pos)(hdr_n_tok - 1));
                            }
                            slot->just_restored = true;
                        }
                        SRV_INF("hydra: STATE_PUT slot=%d restored=%zu B n_past=%d n_prompt_tok=%d\n",
                                id_slot, n_read, res->n_past, hdr_n_tok);
                    }
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_STATE_META:
                {
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_state>();
                    res->id      = task.id;
                    res->id_slot = id_slot;
                    res->op      = HYDRA_OP_STATE_META;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error      = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }
                    // META is safe to serve even while processing or transferring (read-only metadata)
                    int actual_n_past = slot->n_prompt_tokens_cache + slot->n_decoded;
                    // For cold prefills n_prompt_tokens_cache is 0 — use prompt token count
                    if (slot->n_prompt_tokens_cache == 0 && slot->prompt.tokens.size() > 0) {
                        actual_n_past = (int)slot->prompt.tokens.size();
                    }
                    res->n_past = actual_n_past;
                    res->is_processing   = slot->is_processing();
                    res->is_transferring = slot->hydra_transferring->load();
                    res->state_size    = (uint64_t)llama_state_seq_get_size(ctx_tgt, slot->id);
                    // M-Perf.9 #289: surface model identity. The Coordinator uses
                    // these to detect cross-model restores — a slot holding a Mini
                    // KV cache must never have it decoded by a Balanced-loaded model.
                    res->model_alias = model_name;
                    res->model_path  = params_base.model.path;
                    if (model_tgt) {
                        res->tokenizer   = llama_model_get_tokenizer_model(model_tgt);
                        res->model_name  = llama_model_get_display_name(model_tgt);
                        res->model_quant = llama_model_get_quant_label(model_tgt);
                        res->model_capabilities = llama_model_get_capabilities_bitfield(model_tgt);
                    }
                    // #451: populate progress fields based on slot state
                    switch (slot->state) {
                        case SLOT_STATE_PROCESSING_PROMPT:
                            res->operation = "prefill";
                            res->tokens_processed = slot->n_prompt_tokens_processed;
                            // task->n_tokens() is the total tokens to process (fixed);
                            // prompt.tokens.size() grows during prefill and is WRONG for total.
                            res->tokens_total = slot->task ? slot->task->n_tokens() : 0;
                            if (res->tokens_total > 0) {
                                res->progress = (float)res->tokens_processed / (float)res->tokens_total;
                            }
                            res->elapsed_ms = (slot->t_start_process_prompt > 0) 
                                ? (ggml_time_ms() - slot->t_start_process_prompt) : 0;
                            break;
                        case SLOT_STATE_GENERATING:
                            res->operation = "decode";
                            res->tokens_processed = slot->n_decoded;
                            // n_remaining == -1 is the "unlimited generation" sentinel
                            // (no finite n_predict). Don't compute progress in that case.
                            if (slot->n_remaining > 0) {
                                res->tokens_total = slot->n_decoded + slot->n_remaining;
                                res->progress = (float)res->tokens_processed / (float)res->tokens_total;
                            }
                            res->elapsed_ms = (slot->t_start_generation > 0) 
                                ? (ggml_time_ms() - slot->t_start_generation) : 0;
                            break;
                        case SLOT_STATE_IDLE:
                            res->operation = "idle";
                            res->progress = 1.0f;
                            break;
                        default:
                            res->operation = "unknown";
                            break;
                    }
                    // Handle save/restore operations via hydra_transferring flag.
                    // Clear any stale progress from the prior state since we're
                    // now in a transferring context, not the previous operation.
                    if (slot->hydra_transferring->load()) {
                        res->operation = "save";
                        res->progress = 0.0f;
                        res->tokens_processed = 0;
                        res->tokens_total = 0;
                        res->elapsed_ms = 0;
                    }
                    res->rpc_status    = HYDRA_STATUS_OK;
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_CONFIGURE:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_CONFIGURE;
                    res->rpc_status = HYDRA_STATUS_OK;
                    res->success = true;

                    // hydra#406: tiered CONFIGURE (T1/T2/T3). Backward compat:
                    // a legacy {"state_chunk_size":N} payload is treated as a
                    // degenerate T1 (the original hydra#334 startup call from
                    // WorkerSchedulerService.cs:2842).
                    if (task.hydra_action.config_json.empty()) {
                        res->tier = "T1";
                        SRV_INF("hydra: CONFIGURE (empty payload, slot %d) — T1 no-op\n",
                                task.hydra_action.id_slot);
                        queue_results.send(std::move(res));
                        break;
                    }

                    json cfg;
                    try {
                        cfg = json::parse(task.hydra_action.config_json);
                    } catch (const std::exception & e) {
                        res->success = false;
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->error = std::string("CONFIGURE: invalid config_json: ") + e.what();
                        SRV_WRN("hydra: CONFIGURE failed to parse config_json (slot %d): %s\n",
                                task.hydra_action.id_slot, e.what());
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Route through the shared classify → apply helper.
                    // sync=false: T2/T3 are staged for the slot-free moment.
                    hydra_config_result cfg_result = hydra_apply_config(cfg, /*sync=*/false);

                    if (cfg_result.highest_tier == 0) {
                        // No recognized keys — still emit a T1 success
                        // (the legacy {"state_chunk_size":N} case).
                        res->tier = "T1";
                        SRV_INF("hydra: CONFIGURE (no recognized keys, slot %d) — T1 no-op\n",
                                task.hydra_action.id_slot);
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (!cfg_result.ok) {
                        res->success = false;
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->error = "CONFIGURE: " + cfg_result.error;
                        SRV_WRN("hydra: CONFIGURE apply failed (slot %d): %s\n",
                                task.hydra_action.id_slot, cfg_result.error.c_str());
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Build the response from the shared helper's result.
                    res->tier = hydra_tier_label(cfg_result.highest_tier);
                    res->params_applied = std::move(cfg_result.params_applied);
                    res->deferred_keys = std::move(cfg_result.deferred_keys);
                    res->state_chunk_size_applied = cfg_result.state_chunk_size_applied;

                    SRV_INF("hydra: CONFIGURE tier=%s applied=%zu deferred=%zu (slot %d)\n",
                            res->tier.c_str(),
                            res->params_applied.size(),
                            res->deferred_keys.size(),
                            task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_INFO:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_INFO;
                    res->rpc_status = HYDRA_STATUS_OK;
                    // M-Perf.9 #289: advertise the model identity features so
                    // the Coordinator knows it can send `model` in PREFILL and
                    // expect model_alias/model_path/tokenizer/model_name/model_quant/model_capabilities
                    // in META responses.
                    // `preset_aliases` lists every alias loaded from
                    // --models-preset (empty when no preset is configured).
                    json preset_aliases_j = json::array();
                    for (const auto & [alias, _path] : preset_alias_to_path) {
                        preset_aliases_j.push_back(alias);
                    }
                    // Hydra #287/#260/#348: two-engine "work together" status
                    // — see specs/rpc-protocol.md's ENGINE_INFO (0x41)
                    // contract. pipeline_capable stays false until #287's
                    // PIPELINE half lands; mode only ever reports
                    // solo/combined until then. solo_active/rpc_backend_active/
                    // peer_reachable/combined_head_attached are independent
                    // booleans (#348) — replaces the old single "role" string
                    // and the peer_connected/combined_capable field-aliasing.
                    const int32_t expert_mode = ctx_tgt ? llama_hydra_get_expert_mode(ctx_tgt) : 0;
                    // Hydra #383 T1 / #375: advertise "combined" capability when this
                    // engine is ready to serve in COMBINED mode — either via expert-split
                    // (hydra_combined_head_attached) or via layer-split (hydra_combined_static).
                    json capabilities_j = {"prefill", "decode", "state_transfer",
                                           "expert_mode", "quant_swap",
                                           "preset", "tokenizer", "model_name",
                                           "model_quant", "model_capabilities",
                                           "merged_decode"};
                    if (hydra_combined_head_attached || hydra_combined_static) {
                        capabilities_j.push_back("combined");
                    }
                    // In layer-split static mode the engine is always in combined mode;
                    // in expert-split mode it follows the per-request SET_EXPERT_MODE state.
                    const std::string mode_str = hydra_combined_static ? "combined"
                                               : (expert_mode == 1 ? "combined" : "solo");
                    json info_j = {
                        {"engine", "llama-server-hydra"},
                        {"version", "E1"},
                        {"capabilities", capabilities_j},
                        {"preset_aliases", preset_aliases_j},
                        {"solo_active",            hydra_solo_active},
                        {"rpc_backend_active",     hydra_rpc_backend_active},
                        {"mode",                   mode_str},
                        {"split_mode",             hydra_split_mode},
                        {"peer_addr",              hydra_peer},
                        {"peer_reachable",         hydra_peer_reachable},
                        {"layer_split",            hydra_combined_pattern},
                        {"combined_head_attached", hydra_combined_head_attached || hydra_combined_static},
                        {"pipeline_capable",       false}
                    };
                    res->info_json = info_j.dump();
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_PREFILL:
                {
                    const int id_slot = task.hydra_action.id_slot;
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_PREFILL;

                    // #451: track timing for PREFILL metrics
                    const int64_t prefill_start_ms = ggml_time_ms();

                    // Set by the model-resolution block below when a real
                    // `load_model` swap happens. Used at the response site to
                    // decide whether the post-prefill model identity is the
                    // freshly loaded model (swap) or the original (no-swap /
                    // fallback).
                    bool model_was_swapped = false;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (slot->is_processing()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        res->error = "slot is busy";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // M-Perf.9 #289: parse the optional `model` key from the
                    // request body and swap the resident model when the preset
                    // registry knows the alias. The parse is reused for the
                    // tokenization step below. Falls back to the resident model
                    // (with `model_fallback:true` in the response) when the
                    // alias is unknown or no preset is configured.
                    json parsed_body;
                    std::string requested_model;
                    json hydra_cfg; // optional hydra_config object
                    bool has_hydra_config = false;
                    if (!task.hydra_action.request_json.empty()) {
                        try {
                            parsed_body = json::parse(task.hydra_action.request_json);
                            if (parsed_body.is_object() && parsed_body.contains("model")
                                && parsed_body["model"].is_string()) {
                                requested_model = parsed_body["model"].get<std::string>();
                            }
                            // hydra_config: optional config object from Hydra.Core
                            // containing topology/sampling overrides. When present
                            // with model_path, it drives the model swap directly
                            // (bypassing the preset alias lookup).
                            if (parsed_body.is_object() && parsed_body.contains("hydra_config")
                                && parsed_body["hydra_config"].is_object()) {
                                hydra_cfg = parsed_body["hydra_config"];
                                has_hydra_config = true;
                            }
                        } catch (const std::exception & e) {
                            res->rpc_status = HYDRA_STATUS_BAD_REQUEST;
                            res->error = std::string("invalid JSON: ") + e.what();
                            queue_results.send(std::move(res));
                            break;
                        }
                    }

                    // Apply hydra_config synchronously when present.
                    // T1 keys (sampling, n_predict, etc.) are applied in-place.
                    // T2/T3 keys (n_ctx, cache_type, model_path, split_mode, etc.)
                    // trigger immediate rebuilds on this task-queue thread.
                    if (has_hydra_config) {
                        SRV_INF("hydra: PREFILL slot=%d applying hydra_config (%zu keys)\n",
                                id_slot, hydra_cfg.size());
                        hydra_config_result cfg_result = hydra_apply_config(hydra_cfg, /*sync=*/true);
                        if (!cfg_result.ok) {
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->error = "hydra_config apply failed: " + cfg_result.error;
                            SRV_WRN("hydra: PREFILL hydra_config apply failed (slot %d): %s\n",
                                    id_slot, cfg_result.error.c_str());
                            queue_results.send(std::move(res));
                            break;
                        }
                        // If T3 rebuild happened (model swap via model_path),
                        // track it and re-look-up the slot.
                        if (cfg_result.highest_tier == 3) {
                            model_was_swapped = true;
                            res->model_load_ms = (double)(ggml_time_ms() - prefill_start_ms);
                            SRV_INF("hydra: PREFILL hydra_config T3 applied model_alias='%s' tokenizer='%s' model_name='%s' quant='%s' caps=0x%x\n",
                                    model_name.empty() ? "?" : model_name.c_str(),
                                    model_tgt ? llama_model_get_tokenizer_model(model_tgt) : "",
                                    model_tgt ? llama_model_get_display_name(model_tgt) : "",
                                    model_tgt ? llama_model_get_quant_label(model_tgt) : "",
                                    model_tgt ? llama_model_get_capabilities_bitfield(model_tgt) : 0);
                            slot = get_slot_by_id(id_slot);
                            if (slot == nullptr) {
                                res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                                res->error = "slot disappeared after hydra_config T3 rebuild";
                                queue_results.send(std::move(res));
                                break;
                            }
                        }
                        // When hydra_config carries model_path, the model swap is
                        // handled by apply_t3_rebuild() above — skip the bare
                        // model alias lookup below.
                        if (hydra_cfg.contains("model_path")) {
                            requested_model.clear();
                        }
                    }

                    // Fallback: bare model alias lookup when hydra_config didn't
                    // handle the model swap (no hydra_config, or no model_path).
                    if (!requested_model.empty()) {
                        auto it = preset_alias_to_path.find(requested_model);
                        if (it == preset_alias_to_path.end()) {
                            SRV_WRN("hydra: PREFILL model='%s' unknown (preset has %zu alias(es)) — falling back to resident '%s'\n",
                                    requested_model.c_str(), preset_alias_to_path.size(),
                                    model_name.c_str());
                            res->model_fallback = true;
                        } else if (it->second != params_base.model.path) {
                            SRV_INF("hydra: PREFILL model='%s' swapping %s -> %s\n",
                                    requested_model.c_str(), params_base.model.path.c_str(),
                                    it->second.c_str());
                            common_params swapped_params = params_base;
                            // Apply the target alias's full preset so that
                            // tensor_buft_overrides, n_gpu_layers, split_mode,
                            // tensor_split, etc. are replaced — not inherited
                            // from the source model. Intentionally the FULL
                            // preset (sampling, chat template, n_ctx, etc.
                            // included), not just tensor-placement keys: a
                            // real model swap targets a different model,
                            // which plausibly needs its own sampling
                            // defaults/chat template too, not just a new
                            // memory layout.
                            auto pit = preset_alias_to_preset.find(requested_model);
                            if (pit != preset_alias_to_preset.end()) {
                                // Clear inherited tensor_buft_overrides (padded
                                // to 4096 by common_params_parse_ex) BEFORE
                                // apply_to_params, which push_back()'s the new
                                // preset's entries via CLI handlers.  Without
                                // this, the new entries land after the
                                // nullptr-terminator and exceed the 4096 limit,
                                // triggering GGML_ASSERT in
                                // common_model_params_to_llama (#499 regression).
                                swapped_params.tensor_buft_overrides.clear();
                                try {
                                    // apply_to_params() replays CLI handlers
                                    // (parse_tensor_buffer_overrides, the
                                    // n-cpu-moe std::stoi, two-value option
                                    // parsers) which throw on a malformed
                                    // target preset. Uncaught, that exception
                                    // would escape the task-queue loop and
                                    // kill the task thread — fail the swap
                                    // instead.
                                    pit->second.apply_to_params(swapped_params);
                                    hydra_repad_tensor_buft_overrides(swapped_params, "PREFILL swap");
                                } catch (const std::exception & e) {
                                    SRV_WRN("hydra: PREFILL swap preset apply for '%s' failed: %s\n",
                                            requested_model.c_str(), e.what());
                                    res->rpc_status = HYDRA_STATUS_ERROR;
                                    res->error = std::string("model swap preset apply failed: ") + e.what();
                                    queue_results.send(std::move(res));
                                    break;
                                }
                                SRV_INF("hydra: PREFILL swap applied preset for '%s' "
                                        "(tensor_buft_overrides=%zu entries)\n",
                                        requested_model.c_str(),
                                        swapped_params.tensor_buft_overrides.size());
                            }
                            swapped_params.model.path   = it->second;
                            // Update the alias so model_name is re-derived
                            // correctly in load_model() (model_name is set from
                            // model_alias.first when non-empty).
                            swapped_params.model_alias  = { requested_model };
                            // #514: tear down COMBINED state before the
                            // reload — otherwise the engine loads the
                            // correct model file but keeps routing tokens
                            // through the stale peer/expert-binding config,
                            // collapsing decode throughput.
                            const bool was_combined = hydra_combined_head_attached || hydra_combined_static;
                            if (was_combined) {
                                hydra_teardown_combined_before_reload();
                            }
                            const int64_t model_load_start_ms = ggml_time_ms();
                            if (!load_model(swapped_params)) {
                                res->rpc_status = HYDRA_STATUS_ERROR;
                                res->error = "model swap to '" + requested_model + "' failed";
                                queue_results.send(std::move(res));
                                break;
                            }
                            if (was_combined) {
                                hydra_reattach_combined_after_reload();
                            }
                            res->model_load_ms = (double)(ggml_time_ms() - model_load_start_ms);
                            model_was_swapped = true;
                            SRV_INF("hydra: PREFILL swap confirmed model_alias='%s' tokenizer='%s' model_name='%s' quant='%s' caps=0x%x model_load_ms=%.1f\n",
                                    swapped_params.model_alias.empty() ? "?" : swapped_params.model_alias.begin()->c_str(),
                                    model_tgt ? llama_model_get_tokenizer_model(model_tgt) : "",
                                    model_tgt ? llama_model_get_display_name(model_tgt) : "",
                                    model_tgt ? llama_model_get_quant_label(model_tgt) : "",
                                    model_tgt ? llama_model_get_capabilities_bitfield(model_tgt) : 0,
                                    res->model_load_ms);
                            // After load_model, `this` state is reset (new
                            // slots, new context). Re-look up the slot by id.
                            slot = get_slot_by_id(id_slot);
                            if (slot == nullptr) {
                                res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                                res->error = "slot disappeared after model swap";
                                queue_results.send(std::move(res));
                                break;
                            }
                        } else {
                            SRV_DBG("hydra: PREFILL model='%s' already resident, no swap\n",
                                    requested_model.c_str());
                        }
                    }

                    // Tokenize from JSON messages if request_json is provided;
                    // otherwise fall back to pre-tokenized prompt_tokens for back-compat.
                    std::vector<llama_token> prompt_tokens = std::move(task.hydra_action.prompt_tokens);
                    if (!parsed_body.is_null()) {
                        try {
                            std::vector<raw_buffer> dummy_files;
                            json parsed = oaicompat_chat_params_parse(parsed_body, chat_params, dummy_files);
                            if (!parsed.contains("prompt")) {
                                res->rpc_status = HYDRA_STATUS_ERROR;
                                res->error = "chat template produced no prompt";
                                queue_results.send(std::move(res));
                                break;
                            }
                            auto tokenized = tokenize_input_prompts(vocab, mctx, parsed["prompt"], true, true);
                            if (tokenized.empty()) {
                                res->rpc_status = HYDRA_STATUS_ERROR;
                                res->error = "tokenization produced no tokens";
                                queue_results.send(std::move(res));
                                break;
                            }
                            prompt_tokens = tokenized[0].get_tokens();
                        } catch (const std::exception & e) {
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->error = std::string("JSON/tokenization error: ") + e.what();
                            queue_results.send(std::move(res));
                            break;
                        }
                    }

                    SRV_INF("hydra: PREFILL slot=%d tokens=%zu\n", id_slot, prompt_tokens.size());
                    // #469 trace: log first 16 token IDs for cross-flow comparison
                    {
                        std::string tok_ids;
                        for (size_t i = 0; i < std::min<size_t>(16, prompt_tokens.size()); ++i) {
                            if (i > 0) tok_ids += ",";
                            tok_ids += std::to_string(prompt_tokens[i]);
                        }
                        SRV_DBG("hydra: PREFILL slot=%d first16_tokens=[%s] total=%zu\n",
                                id_slot, tok_ids.c_str(), prompt_tokens.size());
                    }

                    // Clear existing slot state
                    slot->prompt_clear(false);
                    slot->n_prompt_tokens_cache = 0;
                    slot->n_prompt_tokens_processed = 0;
                    slot->n_decoded = 0;

                    // Insert prompt tokens
                    if (prompt_tokens.empty()) {
                        res->rpc_status = HYDRA_STATUS_OK;
                        res->n_past = 0;
                        res->state_size = 0;
                        queue_results.send(std::move(res));
                        break;
                    }

                    slot->prompt.tokens.insert(prompt_tokens);
                    const auto & tokens = slot->prompt.tokens.get_tokens();
                    const int n_tokens = (int)tokens.size();

                    // Add BOS if needed (check if slot uses BOS)
                    int token_offset = 0;
                    llama_token bos = llama_vocab_bos(vocab);
                    if (add_bos_token && bos != LLAMA_TOKEN_NULL && (tokens.empty() || tokens[0] != bos)) {
                        token_offset = 1;
                    }

                    // Decode prompt in batches. Hydra #469 fix: upstream's own
                    // invariant (see create_checkpoint call in update_slots,
                    // "we create the checkpoint before calling llama_decode(),
                    // so the current batch is not yet processed and therefore
                    // it is not part of the checkpoint") requires the
                    // checkpoint to be created BEFORE the final token is
                    // decoded. The previous version of this handler decoded
                    // the whole prompt first and only afterward claimed (via
                    // create_checkpoint's pos_max arg, below) that the last
                    // token was still unprocessed. For hybrid/recurrent (SSM)
                    // models, whose memory can't be partially rolled back via
                    // seq_rm, that lie meant a cross-node restore would
                    // re-decode a token that was already baked into the
                    // recurrent state — double-applying it and corrupting the
                    // hidden state. Splitting the loop so the checkpoint is
                    // captured after n_tokens-1 tokens (matching what
                    // create_checkpoint's pos_max already claimed) makes the
                    // claim honest, same as the standard update_slots() path.
                    const int total_tokens = n_tokens + token_offset;
                    const int n_ubatch = llama_n_ubatch(ctx_tgt);
                    const int n_before_last = total_tokens > 1 ? total_tokens - 1 : total_tokens;
                    bool decode_ok = true;
                    for (int i = 0; i < n_before_last && decode_ok; i += n_ubatch) {
                        const int n_tokens_batch = std::min(n_ubatch, n_before_last - i);
                        common_batch_clear(batch);
                        for (int j = 0; j < n_tokens_batch; j++) {
                            const int tok_idx = i + j;
                            llama_token id;
                            if (token_offset > 0 && tok_idx == 0) {
                                id = bos;
                            } else {
                                id = tokens[tok_idx - token_offset];
                            }
                            // No token in this phase is the final prompt
                            // token, so logits are never needed here.
                            common_batch_add(batch, id, tok_idx, {slot->id}, false);
                        }
                        if (llama_decode(ctx_tgt, batch) != 0) {
                            SRV_ERR("hydra: PREFILL slot=%d llama_decode failed at batch %d\n", id_slot, i);
                            decode_ok = false;
                        }
                    }

                    if (!decode_ok) {
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->error = "llama_decode failed during prefill";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Register checkpoint BEFORE decoding the final token, so
                    // its pos_max claim (n_tokens - 1) is honest. Moved up
                    // from after the full-prompt decode (see #469 above).
                    if (n_tokens > 0) {
                        create_checkpoint(*slot, 0, 0, (llama_pos)(n_tokens - 1));
                    }

                    // Decode the held-back final token (if any) now that the
                    // checkpoint has captured the state before it.
                    if (total_tokens > n_before_last) {
                        common_batch_clear(batch);
                        const int tok_idx = total_tokens - 1;
                        llama_token id = (token_offset > 0 && tok_idx == 0)
                            ? bos
                            : tokens[tok_idx - token_offset];
                        common_batch_add(batch, id, tok_idx, {slot->id}, true);
                        if (llama_decode(ctx_tgt, batch) != 0) {
                            SRV_ERR("hydra: PREFILL slot=%d llama_decode failed on final token\n", id_slot);
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->error = "llama_decode failed during prefill (final token)";
                            queue_results.send(std::move(res));
                            break;
                        }
                    }

                    // Update slot tracking
                    slot->n_prompt_tokens_processed = n_tokens;
                    slot->n_prompt_tokens_cache = n_tokens;

                    // Checkpoint already registered above, before the final
                    // token was decoded (#469 fix).

                    // Build v2 blob: [1B version=0x02][4B n_past][4B n_tok][n_tok*4B tokens][1B flags][?ckpt?][raw KV state]
                    const uint32_t hdr_n_past = (uint32_t)n_tokens;
                    const uint32_t hdr_n_tok  = (uint32_t)(tokens.size());
                    uint8_t hdr_flags = 0x00;
                    std::vector<uint8_t> ckpt_buf;
                    int32_t ckpt_pos_min = 0, ckpt_pos_max = 0;
                    int64_t ckpt_n_tokens = 0;
                    if (!slot->prompt.checkpoints.empty()) {
                        hdr_flags |= 0x01;
                        const auto & ckpt = slot->prompt.checkpoints.back();
                        ckpt_pos_min = ckpt.pos_min;
                        ckpt_pos_max = ckpt.pos_max;
                        ckpt_n_tokens = ckpt.n_tokens;
                        const uint64_t tgt_sz = ckpt.data_tgt.size();
                        const uint64_t dft_sz = ckpt.data_dft.size();
                        ckpt_buf.resize(4 + 4 + 8 + 8 + (size_t)tgt_sz + 8 + (size_t)dft_sz);
                        size_t off = 0;
                        memcpy(ckpt_buf.data() + off, &ckpt_pos_min, 4); off += 4;
                        memcpy(ckpt_buf.data() + off, &ckpt_pos_max, 4); off += 4;
                        memcpy(ckpt_buf.data() + off, &ckpt_n_tokens, 8); off += 8;
                        memcpy(ckpt_buf.data() + off, &tgt_sz, 8); off += 8;
                        if (tgt_sz > 0) { memcpy(ckpt_buf.data() + off, ckpt.data_tgt.data(), (size_t)tgt_sz); off += (size_t)tgt_sz; }
                        memcpy(ckpt_buf.data() + off, &dft_sz, 8); off += 8;
                        if (dft_sz > 0) memcpy(ckpt_buf.data() + off, ckpt.data_dft.data(), (size_t)dft_sz);
                    }
                    const size_t base_hdr_size = 1 + 4 + 4 + hdr_n_tok * sizeof(llama_token) + 1;
                    const size_t v2_size = base_hdr_size + ckpt_buf.size();

                    // Get raw KV state
                    const size_t state_size = llama_state_seq_get_size(ctx_tgt, slot->id);
                    std::vector<uint8_t> v2_blob(v2_size + state_size);
                    {
                        size_t off = 0;
                        const uint8_t version_byte = 0x02;
                        memcpy(v2_blob.data() + off, &version_byte, 1); off += 1;
                        memcpy(v2_blob.data() + off, &hdr_n_past, 4);   off += 4;
                        memcpy(v2_blob.data() + off, &hdr_n_tok, 4);    off += 4;
                        if (hdr_n_tok > 0) {
                            const auto & toks = slot->prompt.tokens.get_text_tokens();
                            memcpy(v2_blob.data() + off, toks.data(), toks.size() * sizeof(llama_token));
                            off += toks.size() * sizeof(llama_token);
                        }
                        memcpy(v2_blob.data() + off, &hdr_flags, 1);    off += 1;
                        if (!ckpt_buf.empty()) {
                            memcpy(v2_blob.data() + off, ckpt_buf.data(), ckpt_buf.size());
                            off += ckpt_buf.size();
                        }
                        if (state_size > 0) {
                            llama_state_seq_get_data(ctx_tgt, v2_blob.data() + off, state_size, slot->id);
                        }
                    }

                    // Append logits for activation handoff — eliminates the 1-token trick on the
                    // decode GPU. llama_state_seq_get_data saves KV (k/v tensors) but not the
                    // logits buffer; without these, common_sampler_sample reads garbage after
                    // StatePut. Appending n_vocab floats here lets STATE_PUT inject them directly
                    // into ctx->logits so DECODE can sample immediately.
                    uint64_t logits_size = 0;
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    const float * logits_ptr = llama_get_logits(ctx_tgt);
                    if (logits_ptr && n_vocab > 0) {
                        logits_size = (uint64_t)n_vocab * sizeof(float);
                        const size_t old_sz = v2_blob.size();
                        v2_blob.resize(old_sz + (size_t)logits_size);
                        memcpy(v2_blob.data() + old_sz, logits_ptr, (size_t)logits_size);
                    }

                    SRV_INF("hydra: PREFILL slot=%d done n_past=%d kv=%zu logits=%" PRIu64 "B total=%zu\n",
                            id_slot, n_tokens, state_size, logits_size, v2_blob.size());

                    // M-Perf.9 #289: model identity for the slot the prefill
                    // was just built on. Coordinator uses this to populate
                    // item.KvModelAlias/Hash and to gate RestoreKvAsync. When
                    // a `model` swap happened earlier in this handler, the
                    // post-swap `model_name` / `params_base.model.path` /
                    // `model` are used. `res->model_fallback` was set by the
                    // model-resolution block above; we preserve it here.
                    res->model_alias    = model_name;
                    res->model_path     = params_base.model.path;
                    // res->model_fallback may already be true (alias unknown
                    // or no preset); only set false when no swap was needed.
                    if (!model_was_swapped && !res->model_fallback) {
                        // nothing to do — leave as-is
                    }
                    if (model_tgt) {
                        res->tokenizer   = llama_model_get_tokenizer_model(model_tgt);
                        res->model_name  = llama_model_get_display_name(model_tgt);
                        res->model_quant = llama_model_get_quant_label(model_tgt);
                        res->model_capabilities = llama_model_get_capabilities_bitfield(model_tgt);
                    }

                    res->rpc_status  = HYDRA_STATUS_OK;
                    res->n_past      = n_tokens;
                    res->state_data  = std::move(v2_blob);
                    res->state_size  = state_size;
                    res->logits_size = logits_size;
                    // #451: populate PREFILL metrics
                    res->prefill_ms = (double)(ggml_time_ms() - prefill_start_ms);
                    res->prompt_tokens = n_tokens;
                    res->kv_size = state_size;
                    if (res->prefill_ms > 0 && n_tokens > 0) {
                        res->tokens_per_second = (double)n_tokens / (res->prefill_ms / 1000.0);
                    }
                    res->cache_tokens = slot->n_prompt_tokens_cache;
                    // #469 trace: log PREFILL completion with token IDs for cross-flow comparison
                    SRV_DBG("hydra: PREFILL_DONE slot=%d n_past=%d state_size=%zu logits_size=%zu blob_size=%zu prefill_ms=%.1f\n",
                            id_slot, n_tokens, state_size, logits_size, v2_blob.size(), res->prefill_ms);
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_DECODE:
                {
                    // ── Sync phase: Gate A (header-only, no GGUF reads, ~1 ms) ──
                    // Identity validation, slot reservation, post DECODE_APPLY.
                    // No model I/O, no KV touched.
                    const int id_slot = task.hydra_action.id_slot;
                    const int32_t decode_request_id = task.hydra_action.decode_request_id;
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_DECODE;
                    res->decode_request_id = decode_request_id;
                    res->id_slot = id_slot;

                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        res->rpc_status = HYDRA_STATUS_NOT_FOUND;
                        res->error = "invalid slot ID";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (slot->is_processing()) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        res->error = "slot is busy";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Reject if slot is reserved for another decode
                    if (slot->reserved_for_decode_id != -1 && slot->reserved_for_decode_id != decode_request_id) {
                        res->rpc_status = HYDRA_STATUS_BUSY;
                        res->error = "slot reserved for another decode";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Parse the merged DECODE JSON header
                    json decode_req;
                    try {
                        decode_req = json::parse(task.hydra_action.decode_json);
                    } catch (const std::exception & e) {
                        res->rpc_status = HYDRA_STATUS_BAD_REQUEST;
                        res->error = std::string("invalid JSON: ") + e.what();
                        queue_results.send(std::move(res));
                        break;
                    }

                    // ── Gate A: header-only metadata comparison ─────────────
                    // Compare kv_metadata vs model_metadata from the control
                    // header. No GGUF reads, no KV touched.
                    const json & kv_meta    = decode_req["kv_metadata"];
                    const json & model_meta = decode_req.value("model_metadata", json::object());

                    // Read request identities from header
                    const std::string req_tokenizer   = kv_meta.value("tokenizer", "");
                    const std::string req_model_name  = kv_meta.value("model_name", "");
                    const uint32_t    req_capabilities = kv_meta.value("model_capabilities", 0u);

                    // Read target identities from header
                    const std::string tgt_tokenizer   = model_meta.value("tokenizer", "");
                    const std::string tgt_model_name  = model_meta.value("model_name", "");

                    const bool tokenizer_match  = (req_tokenizer == tgt_tokenizer);
                    bool model_name_match = (req_model_name == tgt_model_name);
                    // #589: cross-node same-model name tolerance. The KV's
                    // model_name is the display name (GGUF metadata) of the
                    // file that BUILT the KV — a different build/quant of the
                    // same model than the decode node's resident file, so
                    // string equality legitimately fails for the same logical
                    // model (e.g. kv_metadata carries the source node's
                    // display name, the decode node reports its resident
                    // filename). When the header carries the KV's source
                    // alias (kv_metadata.model_alias) or the resolved request
                    // alias ("model") and that alias maps through the preset
                    // table to the resident model path, the KV was built by
                    // the same logical model — accept. The alias→path check
                    // is exact (per-node preset INI), so a different model
                    // (Mini vs Balanced, 27B vs 35B) still maps to a
                    // different path and is rejected.
                    if (!model_name_match) {
                        const std::string kv_alias  = kv_meta.value("model_alias", "");
                        const std::string hdr_alias = decode_req.value("model", std::string());
                        for (const auto & cand : { kv_alias, hdr_alias }) {
                            if (cand.empty()) {
                                continue;
                            }
                            auto pit = preset_alias_to_path.find(cand);
                            if (pit != preset_alias_to_path.end() && pit->second == params_base.model.path) {
                                SRV_INF("hydra: DECODE slot=%d Gate A name fallback — alias '%s' maps to resident path, same logical model\n",
                                        id_slot, cand.c_str());
                                model_name_match = true;
                                break;
                            }
                        }
                    }
                    const uint32_t capabilities_xor = req_capabilities ^ model_meta.value("model_capabilities", 0u);

                    static const char * kCapBitNames[] = {"MTP", "VISION", "REASONING", "TOOL_USE", "CODE"};
                    std::vector<std::string> capabilities_diff_bits;
                    for (int b = 0; b < 5; b++) {
                        if (capabilities_xor & (1u << b)) {
                            capabilities_diff_bits.push_back(kCapBitNames[b]);
                        }
                    }

                    // MTP(bit0) + VISION(bit1) mismatch → hard reject
                    const bool valid = tokenizer_match && model_name_match
                                       && !(capabilities_xor & 0x3);

                    json match_j = {
                        {"tokenizer_match",        tokenizer_match},
                        {"model_name_match",       model_name_match},
                        {"capabilities_xor",       capabilities_xor},
                        {"capabilities_diff_bits", capabilities_diff_bits},
                        {"model_quant_match",      kv_meta.value("model_quant", "") == model_meta.value("model_quant", "")},
                        {"model_alias_match",      true},
                    };
                    res->match_json = match_j;
                    res->match_valid = valid;

                    if (!valid) {
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->error = "model_capabilities_mismatch";
                        SRV_WRN("hydra: DECODE slot=%d Gate A reject — tokenizer=%d name=%d caps_xor=0x%x\n",
                                id_slot, tokenizer_match, model_name_match, capabilities_xor);
                        queue_results.send(std::move(res));
                        break;
                    }

                    // ── Reserve slot ────────────────────────────────────────
                    slot->reserved_for_decode_id = decode_request_id;

                    SRV_INF("hydra: DECODE slot=%d Gate A pass, reserved for request_id=%d\n",
                            id_slot, decode_request_id);

                    // ── Create decode_result_entry (LOADING state) ─────────
                    // So GET /v1/decode/{id} returns 202 instead of 404
                    // while async DECODE_APPLY is pending.
                    if (routes_ptr) {
                        server_routes::decode_result_entry entry;
                        entry.id_slot    = id_slot;
                        entry.state      = server_routes::DECODE_STATE_LOADING;
                        entry.match_json = match_j;
                        entry.created_at = std::time(nullptr);
                        entry.ttl_s      = routes_ptr->decode_result_ttl_s;
                        entry.model_metadata = decode_req.value("model_metadata", json::object());
                        entry.model_identity = json::object();
                        std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                        routes_ptr->decode_results[decode_request_id] = std::move(entry);
                        routes_ptr->evict_decode_results_locked();
                    }

                    // ── Send sync validation response ───────────────────────
                    res->rpc_status = HYDRA_STATUS_OK;
                    queue_results.send(std::move(res));

                    // ── Post DECODE_APPLY async task ────────────────────────
                    {
                        server_task apply_task(SERVER_TASK_TYPE_HYDRA_DECODE_APPLY);
                        apply_task.id = queue_tasks.get_new_id();
                        apply_task.hydra_action.id_slot = id_slot;
                        apply_task.hydra_action.decode_json = std::move(task.hydra_action.decode_json);
                        apply_task.hydra_action.kv_data = std::move(task.hydra_action.kv_data);
                        apply_task.hydra_action.decode_request_id = decode_request_id;
                        queue_tasks.post(std::move(apply_task));
                        SRV_INF("hydra: DECODE slot=%d posted DECODE_APPLY (request_id=%d)\n",
                                id_slot, decode_request_id);
                    }
                } break;

            case SERVER_TASK_TYPE_HYDRA_DECODE_APPLY:
                {
                    // ── Async phase: model swap + Gate B + KV restore + completion ──
                    const int id_slot = task.hydra_action.id_slot;
                    const int32_t decode_request_id = task.hydra_action.decode_request_id;

                    // Parse the DECODE JSON header (re-parsed for async context)
                    json decode_req;
                    try {
                        decode_req = json::parse(task.hydra_action.decode_json);
                    } catch (const std::exception & e) {
                        SRV_WRN("hydra: DECODE_APPLY slot=%d invalid JSON: %s\n", id_slot, e.what());
                        // Release reservation on error
                        server_slot * s = get_slot_by_id(id_slot);
                        if (s) s->reserved_for_decode_id = -1;
                        if (routes_ptr) {
                            server_routes::decode_result_entry entry;
                            entry.id_slot = id_slot;
                            entry.error = std::string("DECODE_APPLY JSON parse error: ") + e.what();
                            entry.created_at = std::time(nullptr);
                            entry.ttl_s = routes_ptr->decode_result_ttl_s;
                            std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                            routes_ptr->decode_results[decode_request_id] = std::move(entry);
                            routes_ptr->evict_decode_results_locked();
                        }
                        break;
                    }

                    const json & kv_meta    = decode_req["kv_metadata"];
                    const json & model_meta = decode_req.value("model_metadata", json::object());

                    // ── Model swap (if requested model != resident) ─────────
                    const std::string requested_model = decode_req.value("model", std::string());
                    double model_load_ms = 0.0;
                    bool model_fallback = false;

                    if (!requested_model.empty()) {
                        auto it = preset_alias_to_path.find(requested_model);
                        if (it == preset_alias_to_path.end()) {
                            SRV_WRN("hydra: DECODE_APPLY slot=%d model='%s' unknown — falling back to resident '%s'\n",
                                    id_slot, requested_model.c_str(), model_name.c_str());
                            model_fallback = true;
                        } else if (it->second != params_base.model.path) {
                            SRV_INF("hydra: DECODE_APPLY slot=%d model='%s' swapping %s -> %s\n",
                                    id_slot, requested_model.c_str(), params_base.model.path.c_str(),
                                    it->second.c_str());
                            common_params swapped_params = params_base;
                            // Apply the target alias's full preset (same
                            // treatment, and same intentional full-preset
                            // scope, as the PREFILL path above).
                            auto pit = preset_alias_to_preset.find(requested_model);
                            bool preset_apply_failed = false;
                            if (pit != preset_alias_to_preset.end()) {
                                // Same clear+re-pad+try/catch as the PREFILL path.
                                swapped_params.tensor_buft_overrides.clear();
                                try {
                                    pit->second.apply_to_params(swapped_params);
                                    hydra_repad_tensor_buft_overrides(swapped_params, "DECODE_APPLY swap");
                                } catch (const std::exception & e) {
                                    SRV_WRN("hydra: DECODE_APPLY slot=%d swap preset apply for '%s' failed: %s\n",
                                            id_slot, requested_model.c_str(), e.what());
                                    preset_apply_failed = true;
                                    if (routes_ptr) {
                                        server_routes::decode_result_entry entry;
                                        entry.id_slot = id_slot;
                                        entry.error = std::string("model swap preset apply failed: ") + e.what();
                                        entry.created_at = std::time(nullptr);
                                        entry.ttl_s = routes_ptr->decode_result_ttl_s;
                                        std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                                        routes_ptr->decode_results[decode_request_id] = std::move(entry);
                                        routes_ptr->evict_decode_results_locked();
                                    }
                                }
                            }
                            if (preset_apply_failed) {
                                server_slot * s = get_slot_by_id(id_slot);
                                if (s) s->reserved_for_decode_id = -1;
                                break;
                            }
                            swapped_params.model.path  = it->second;
                            swapped_params.model_alias = { requested_model };
                            // #514: tear down COMBINED state before the
                            // reload — see hydra_teardown_combined_before_reload().
                            const bool was_combined = hydra_combined_head_attached || hydra_combined_static;
                            if (was_combined) {
                                hydra_teardown_combined_before_reload();
                            }
                            const int64_t model_load_start_ms = ggml_time_ms();
                            if (!load_model(swapped_params)) {
                                SRV_WRN("hydra: DECODE_APPLY slot=%d model swap to '%s' failed\n",
                                        id_slot, requested_model.c_str());
                                server_slot * s = get_slot_by_id(id_slot);
                                if (s) s->reserved_for_decode_id = -1;
                                if (routes_ptr) {
                                    server_routes::decode_result_entry entry;
                                    entry.id_slot = id_slot;
                                    entry.error = "model swap to '" + requested_model + "' failed";
                                    entry.created_at = std::time(nullptr);
                                    entry.ttl_s = routes_ptr->decode_result_ttl_s;
                                    std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                                    routes_ptr->decode_results[decode_request_id] = std::move(entry);
                                    routes_ptr->evict_decode_results_locked();
                                }
                                break;
                            }
                            if (was_combined) {
                                hydra_reattach_combined_after_reload();
                            }
                            model_load_ms = (double)(ggml_time_ms() - model_load_start_ms);
                            SRV_INF("hydra: DECODE_APPLY slot=%d swap confirmed model_load_ms=%.1f\n",
                                    id_slot, model_load_ms);
                        }
                    }

                    // ── Gate B: post-load identity check ────────────────────
                    // Compare model_metadata from header vs ACTUAL resident GGUF identity.
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        SRV_WRN("hydra: DECODE_APPLY slot=%d disappeared after model swap\n", id_slot);
                        if (routes_ptr) {
                            server_routes::decode_result_entry entry;
                            entry.id_slot = id_slot;
                            entry.error = "slot disappeared after model swap";
                            entry.created_at = std::time(nullptr);
                            entry.ttl_s = routes_ptr->decode_result_ttl_s;
                            std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                            routes_ptr->decode_results[decode_request_id] = std::move(entry);
                            routes_ptr->evict_decode_results_locked();
                        }
                        break;
                    }

                    const std::string resident_tokenizer   = llama_model_get_tokenizer_model(model_tgt);
                    const std::string resident_model_name  = llama_model_get_display_name(model_tgt);
                    const std::string resident_model_quant = llama_model_get_quant_label(model_tgt);
                    const uint32_t    resident_capabilities = llama_model_get_capabilities_bitfield(model_tgt);

                    const std::string hdr_model_name  = model_meta.value("model_name", "");
                    const std::string hdr_model_quant = model_meta.value("model_quant", "");
                    const uint32_t    hdr_capabilities = model_meta.value("model_capabilities", 0u);

                    const bool gate_b_tokenizer   = (resident_tokenizer  == model_meta.value("tokenizer", ""));
                    const bool gate_b_model_name  = (resident_model_name == hdr_model_name);
                    const uint32_t gate_b_caps_xor = resident_capabilities ^ hdr_capabilities;

                    if (!gate_b_tokenizer || !gate_b_model_name || (gate_b_caps_xor & 0x3)) {
                        SRV_WRN("hydra: DECODE_APPLY slot=%d Gate B reject — tokenizer=%d name=%d caps_xor=0x%x\n",
                                id_slot, gate_b_tokenizer, gate_b_model_name, gate_b_caps_xor);
                        slot->reserved_for_decode_id = -1;
                        if (routes_ptr) {
                            server_routes::decode_result_entry entry;
                            entry.id_slot = id_slot;
                            entry.error = "Gate B identity mismatch after model load";
                            entry.match_json = {{"gate_b_tokenizer", gate_b_tokenizer}, {"gate_b_name", gate_b_model_name}, {"gate_b_caps_xor", gate_b_caps_xor}};
                            entry.created_at = std::time(nullptr);
                            entry.ttl_s = routes_ptr->decode_result_ttl_s;
                            std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                            routes_ptr->decode_results[decode_request_id] = std::move(entry);
                            routes_ptr->evict_decode_results_locked();
                        }
                        break;
                    }

                    if (resident_model_quant != hdr_model_quant) {
                        SRV_INF("hydra: DECODE_APPLY slot=%d Gate B quant differs (%s → %s) — mix-quant allowed\n",
                                id_slot, hdr_model_quant.c_str(), resident_model_quant.c_str());
                    }

                    // ── KV restore ─────────────────────────────────────────
                    const int64_t restore_start_ms = ggml_time_ms();

                    if (!task.hydra_action.kv_data.empty()) {
                        slot->prompt_clear(false);
                        slot->n_prompt_tokens_cache = 0;
                        slot->n_prompt_tokens_processed = 0;
                        slot->n_decoded = 0;

                        // The coordinator may send the v2 blob (header + raw KV)
                        // or just the raw KV data.  Parse the v2 header to extract
                        // the token list so update_slots()'s n_common decision can
                        // match incoming tokens against the restored KV — without
                        // this, prompt.tokens is empty after prompt_clear(), n_past
                        // computes to 0, and seq_rm(slot, 0, -1) wipes the KV that
                        // llama_state_seq_set_data just loaded (issue #506).
                        const uint8_t * kv_ptr = task.hydra_action.kv_data.data();
                        size_t          kv_len = task.hydra_action.kv_data.size();
                        int32_t         blob_n_past = 0;
                        int32_t         blob_n_tok  = 0;
                        bool            has_chkpt = false;
                        int32_t         ckpt_pos_min_in = 0, ckpt_pos_max_in = 0;
                        int64_t         ckpt_n_tokens_in = 0;
                        std::vector<uint8_t> ckpt_tgt_data, ckpt_dft_data;

                        const bool is_v2 = kv_len >= 1 && kv_ptr[0] == 0x02;
                        if (is_v2 && kv_len >= 9) {
                            memcpy(&blob_n_past, kv_ptr + 1, 4);
                            memcpy(&blob_n_tok,  kv_ptr + 5, 4);

                            const size_t token_start = 9;
                            const size_t token_end   = token_start + (size_t)blob_n_tok * sizeof(llama_token);
                            if (blob_n_tok > 0 && token_end <= kv_len) {
                                // Restore token list from v2 blob header
                                slot->prompt.tokens.clear();
                                const llama_token * tok_ptr = (const llama_token *)(kv_ptr + token_start);
                                llama_tokens restored_tokens(tok_ptr, tok_ptr + (size_t)blob_n_tok);
                                slot->prompt.tokens.insert(restored_tokens);
                                SRV_INF("hydra: DECODE_APPLY slot=%d v2 blob: restored %d tokens from header\n",
                                        id_slot, blob_n_tok);
                            }

                            // Skip past v2 header (version + n_past + n_tok + tokens + flags + optional checkpoint)
                            size_t hdr_offset = token_end;
                            if (hdr_offset < kv_len) {
                                const uint8_t flags = kv_ptr[hdr_offset];
                                hdr_offset += 1; // past flags byte
                                if (flags & 0x01) {
                                    // Capture checkpoint: 4B pos_min | 4B pos_max | 8B n_tokens | 8B tgt_sz | tgt_data | 8B dft_sz | dft_data
                                    // Mirrors the STATE_PUT sibling (~line 3343) — the native
                                    // checkpoint is registered after restore so hybrid/recurrent
                                    // models get their recurrent memory back (KV restored without
                                    // its checkpoint is corrupt).
                                    if (hdr_offset + 4 + 4 + 8 + 8 <= kv_len) {
                                        memcpy(&ckpt_pos_min_in, kv_ptr + hdr_offset, 4); hdr_offset += 4;
                                        memcpy(&ckpt_pos_max_in, kv_ptr + hdr_offset, 4); hdr_offset += 4;
                                        memcpy(&ckpt_n_tokens_in, kv_ptr + hdr_offset, 8); hdr_offset += 8;
                                        uint64_t tgt_sz_in;
                                        memcpy(&tgt_sz_in, kv_ptr + hdr_offset, 8); hdr_offset += 8;
                                        if (tgt_sz_in > 0 && hdr_offset + tgt_sz_in <= kv_len) {
                                            ckpt_tgt_data.assign(kv_ptr + hdr_offset, kv_ptr + hdr_offset + (size_t)tgt_sz_in);
                                            hdr_offset += (size_t)tgt_sz_in;
                                        }
                                        if (hdr_offset + 8 <= kv_len) {
                                            uint64_t dft_sz_in;
                                            memcpy(&dft_sz_in, kv_ptr + hdr_offset, 8); hdr_offset += 8;
                                            if (dft_sz_in > 0 && hdr_offset + dft_sz_in <= kv_len) {
                                                ckpt_dft_data.assign(kv_ptr + hdr_offset, kv_ptr + hdr_offset + (size_t)dft_sz_in);
                                                hdr_offset += (size_t)dft_sz_in;
                                            }
                                        }
                                        has_chkpt = true;
                                    }
                                }
                            }
                            // Advance kv_ptr/kv_len past the v2 header to the raw KV state
                            if (hdr_offset <= kv_len) {
                                kv_ptr = kv_ptr + hdr_offset;
                                kv_len = kv_len - hdr_offset;
                            }
                        }

                        auto status = llama_state_seq_set_data(
                            ctx_tgt,
                            kv_ptr,
                            kv_len,
                            slot->id);

                        // llama_state_seq_set_data returns the number of bytes
                        // read on success (0 means failed to load) — see its
                        // doc comment in include/llama.h. `status` only counts
                        // the KV-cache bytes the reader consumed; it does NOT
                        // include the trailing logits PREFILL_DONE appends
                        // (see ~line 4098), so status < kv_len is the normal
                        // case whenever logits are present — compare against
                        // kv_len here and this false-fails on every restore
                        // with logits. Matches the STATE_PUT sibling check
                        // (server-context.cpp ~line 3395: `if (n_read == 0)`).
                        if (status == 0) {
                            SRV_WRN("hydra: DECODE_APPLY slot=%d KV restore failed (%d)\n", id_slot, status);
                            slot->reserved_for_decode_id = -1;
                            // Tokens were registered from the v2 header before set_data —
                            // clear them so the slot is not left poisoned (n_past > 0
                            // with no KV cells → pos_min == -1 abort on the next decode
                            // that touches this slot). Matches the STATE_PUT failure path.
                            slot->prompt.tokens.clear();
                            slot->prompt.checkpoints.clear();
                            slot->n_prompt_tokens_cache = 0;
                            llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot->id, -1, -1);
                            if (routes_ptr) {
                                server_routes::decode_result_entry entry;
                                entry.id_slot = id_slot;
                                entry.error = "KV restore failed (llama_state_seq_set_data returned " + std::to_string(status) + ")";
                                entry.created_at = std::time(nullptr);
                                entry.ttl_s = routes_ptr->decode_result_ttl_s;
                                std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                                routes_ptr->decode_results[decode_request_id] = std::move(entry);
                                routes_ptr->evict_decode_results_locked();
                            }
                            break;
                        }

                        // Trailing logits: PREFILL_DONE appends n_vocab floats
                        // after the KV state (~line 4098) so the decode side
                        // can sample immediately instead of reading garbage
                        // after restore. Mirrors STATE_PUT's per-slot
                        // injection (~line 3405) — DECODE_APPLY was missing
                        // this step entirely.
                        {
                            const size_t remaining = kv_len - status;
                            const size_t expected_logits = (size_t)llama_vocab_n_tokens(vocab) * sizeof(float);
                            if (remaining == expected_logits) {
                                const float * src = (const float *)(kv_ptr + status);
                                const size_t n_floats = llama_vocab_n_tokens(vocab);
                                slot->restored_logits.assign(src, src + n_floats);
                                slot->logits_valid = true;
                                SRV_INF("hydra: DECODE_APPLY slot=%d restored %zu logits to per-slot buffer\n",
                                        id_slot, n_floats);
                            }
                        }

                        const int n_past = is_v2 ? blob_n_past : kv_meta.value("n_past", 0);
                        if (n_past > 0) {
                            // Cache/processed counters come from the same header field
                            // STATE_PUT reads (hdr_n_tok == blob_n_tok here); PREFILL writes
                            // both fields as n_tokens so the values are identical today,
                            // but the two restore paths must read the SAME source.
                            slot->n_prompt_tokens_cache = is_v2 ? blob_n_tok : n_past;
                            slot->n_prompt_tokens_processed = is_v2 ? blob_n_tok : n_past;

                            // Register the native checkpoint from the blob (v2) or
                            // fabricate one (legacy) — mirrors STATE_PUT (~line 3447).
                            // KV restored without its recurrent-memory checkpoint
                            // corrupts hybrid/recurrent model output.
                            slot->prompt.checkpoints.clear();
                            if (has_chkpt) {
                                auto & ckpt = slot->prompt.checkpoints.emplace_back();
                                ckpt.n_tokens = ckpt_n_tokens_in;
                                ckpt.pos_min  = ckpt_pos_min_in;
                                ckpt.pos_max  = ckpt_pos_max_in;
                                ckpt.data_tgt = std::move(ckpt_tgt_data);
                                ckpt.data_dft = std::move(ckpt_dft_data);
                                SLT_INF(*slot, "DECODE_APPLY registered native checkpoint (pos_min=%d pos_max=%d n_tokens=%" PRId64 " tgt_sz=%zu)\n",
                                        ckpt.pos_min, ckpt.pos_max, ckpt.n_tokens, ckpt.data_tgt.size());
                            } else {
                                create_checkpoint(*slot, 0, 0, (llama_pos)(n_past - 1));
                            }
                        }
                        slot->just_restored = true;
                    }

                    const double restore_slot_ms = (double)(ggml_time_ms() - restore_start_ms);
                    const int n_past = slot->n_prompt_tokens_cache + slot->n_decoded;

                    SRV_INF("hydra: DECODE_APPLY slot=%d restore=%.1fms n_past=%d model_load_ms=%.1f\n",
                            id_slot, restore_slot_ms, n_past, model_load_ms);

                    // Release reservation — slot is now processing via completion
                    slot->reserved_for_decode_id = -1;

                    // ── Build and post COMPLETION task ──────────────────────
                    {
                        json prompt = decode_req["prompt"];
                        json cmpl_data;
                        cmpl_data["stream"] = prompt.value("stream", false);
                        cmpl_data["n_predict"] = prompt.value("n_predict", 256);
                        cmpl_data["id_slot"] = id_slot;
                        if (prompt.contains("sampling")) {
                            const json & samp = prompt["sampling"];
                            if (samp.contains("temperature")) cmpl_data["temperature"] = samp["temperature"];
                            if (samp.contains("top_p"))       cmpl_data["top_p"] = samp["top_p"];
                            if (samp.contains("top_k"))       cmpl_data["top_k"] = samp["top_k"];
                            if (samp.contains("seed"))        cmpl_data["seed"] = samp["seed"];
                        }
                        if (prompt.contains("stop")) cmpl_data["stop"] = prompt["stop"];

                        std::string prompt_str;
                        if (prompt.contains("messages") && !prompt["messages"].is_null()) {
                            json chat_body;
                            chat_body["messages"] = prompt["messages"];
                            if (prompt.contains("tools"))        chat_body["tools"]        = prompt["tools"];
                            if (prompt.contains("tool_choice"))  chat_body["tool_choice"]  = prompt["tool_choice"];
                            if (prompt.contains("response_format")) chat_body["response_format"] = prompt["response_format"];
                            if (prompt.contains("add_generation_prompt")) chat_body["add_generation_prompt"] = prompt["add_generation_prompt"];
                            if (prompt.contains("continue_final_message")) chat_body["continue_final_message"] = prompt["continue_final_message"];
                            if (prompt.contains("reasoning_format")) chat_body["reasoning_format"] = prompt["reasoning_format"];
                            if (prompt.contains("enable_thinking")) chat_body["enable_thinking"] = prompt["enable_thinking"];
                            if (prompt.contains("chat_template_kwargs")) chat_body["chat_template_kwargs"] = prompt["chat_template_kwargs"];

                            try {
                                std::vector<raw_buffer> dummy_files;
                                json chat_result = oaicompat_chat_params_parse(chat_body, chat_params, dummy_files);
                                prompt_str = chat_result.value("prompt", std::string());
                                if (chat_result.contains("grammar") && !chat_result["grammar"].is_null()) cmpl_data["grammar"] = chat_result["grammar"];
                                if (chat_result.contains("grammar_type")) cmpl_data["grammar_type"] = chat_result["grammar_type"];
                                if (chat_result.contains("grammar_lazy")) cmpl_data["grammar_lazy"] = chat_result["grammar_lazy"];
                                if (chat_result.contains("grammar_triggers")) cmpl_data["grammar_triggers"] = chat_result["grammar_triggers"];
                                if (chat_result.contains("chat_format")) cmpl_data["chat_format"] = chat_result["chat_format"];
                                if (chat_result.contains("chat_parser")) cmpl_data["chat_parser"] = chat_result["chat_parser"];
                                if (chat_result.contains("parse_tool_calls")) cmpl_data["parse_tool_calls"] = chat_result["parse_tool_calls"];
                                if (chat_result.contains("preserved_tokens")) cmpl_data["preserved_tokens"] = chat_result["preserved_tokens"];
                                if (chat_result.contains("reasoning_budget_tokens")) cmpl_data["reasoning_budget_tokens"] = chat_result["reasoning_budget_tokens"];
                                if (chat_result.contains("reasoning_budget_start_tag")) cmpl_data["reasoning_budget_start_tag"] = chat_result["reasoning_budget_start_tag"];
                                if (chat_result.contains("reasoning_budget_end_tag")) cmpl_data["reasoning_budget_end_tag"] = chat_result["reasoning_budget_end_tag"];
                                if (chat_result.contains("reasoning_budget_message")) cmpl_data["reasoning_budget_message"] = chat_result["reasoning_budget_message"];
                                if (chat_result.contains("reasoning_control")) cmpl_data["reasoning_control"] = chat_result["reasoning_control"];
                                if (chat_result.contains("stop") && chat_result["stop"].is_array()) {
                                    json existing_stops = cmpl_data.value("stop", json::array());
                                    for (const auto & s : chat_result["stop"]) existing_stops.push_back(s);
                                    cmpl_data["stop"] = existing_stops;
                                }
                            } catch (const std::exception & e) {
                                SRV_WRN("hydra: DECODE_APPLY slot=%d chat template failed: %s\n", id_slot, e.what());
                                if (routes_ptr) {
                                    server_routes::decode_result_entry entry;
                                    entry.id_slot = id_slot;
                                    entry.error = std::string("chat template error: ") + e.what();
                                    entry.created_at = std::time(nullptr);
                                    entry.ttl_s = routes_ptr->decode_result_ttl_s;
                                    std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                                    routes_ptr->decode_results[decode_request_id] = std::move(entry);
                                    routes_ptr->evict_decode_results_locked();
                                }
                                break;
                            }
                        } else {
                            prompt_str = prompt.value("prompt", std::string());
                        }
                        cmpl_data["prompt"] = prompt_str;

                        auto inputs = tokenize_input_prompts(vocab, mctx, prompt_str, true, true);
                        if (!inputs.empty()) {
                            const int32_t completion_id = queue_tasks.get_new_id();

                            server_task cmpl_task(SERVER_TASK_TYPE_COMPLETION);
                            cmpl_task.id = completion_id;
                            cmpl_task.id_slot = id_slot;
                            cmpl_task.tokens = std::move(inputs[0]);
                            cmpl_task.params = server_task::params_from_json_cmpl(
                                vocab, params_base, get_slot_n_ctx(), params_base.sampling.logit_bias_eog, cmpl_data);
                            cmpl_task.params.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
                            cmpl_task.params.oaicompat_cmpl_id = gen_chatcmplid();
                            cmpl_task.params.oaicompat_model = model_name;

                            queue_results.add_waiting_task_id(completion_id);
                            queue_tasks.post(std::move(cmpl_task));
                            SRV_INF("hydra: DECODE_APPLY slot=%d posted COMPLETION (completion_id=%d, request_id=%d)\n",
                                    id_slot, completion_id, decode_request_id);

                            // Update state to GENERATING
                            if (routes_ptr) {
                                std::lock_guard<std::mutex> lk(routes_ptr->decode_results_mutex);
                                auto dit = routes_ptr->decode_results.find(decode_request_id);
                                if (dit != routes_ptr->decode_results.end()) {
                                    dit->second.state = server_routes::DECODE_STATE_GENERATING;
                                    dit->second.completion_id = std::to_string(completion_id);
                                    dit->second.stream->completion_task_id = completion_id;
                                    // Capture n_common observability from the slot
                                    dit->second.n_common           = slot->n_common;
                                    dit->second.n_prompt_processed = slot->n_prompt_processed;
                                    dit->second.logits_reused      = slot->logits_reused;
                                }
                            }

                            // ── Background consumer ─────────────────────────
                            // Sole listener on the completion task.  Relays
                            // partial results into the decode_result_entry's
                            // streaming_queue so GET /v1/decode can stream
                            // them to the client.  Stores the final result
                            // when generation completes.
                            if (routes_ptr) {
                                // Read match_json from the decode_result_entry (set by sync DECODE)
                                json match_j_bg;
                                {
                                    std::lock_guard<std::mutex> lk(routes_ptr->decode_results_mutex);
                                    auto dit = routes_ptr->decode_results.find(decode_request_id);
                                    if (dit != routes_ptr->decode_results.end()) {
                                        match_j_bg = dit->second.match_json;
                                    }
                                }
                                std::thread([this, completion_id, decode_request_id, id_slot,
                                             match_j = std::move(match_j_bg), resident_tokenizer, resident_model_name,
                                             resident_model_quant, resident_capabilities,
                                             oaicompat_model_name = model_name,
                                             model_load_ms, restore_slot_ms, n_past,
                                             &results = queue_results]() mutable {
                                    std::unordered_set<int> ids = {(int)completion_id};
                                    bool got_final = false;

                                    // Loop: receive partials and relay, wait for final
                                    while (!got_final) {
                                        auto res_ptr = results.recv_with_timeout(ids, 120);
                                        if (!res_ptr) {
                                            SRV_WRN("hydra: DECODE_APPLY slot=%d generation timeout (request_id=%d, completion_id=%d)\n",
                                                    id_slot, decode_request_id, completion_id);
                                            // Mark stream as finished so GET handler unblocks
                                            {
                                                std::lock_guard<std::mutex> lk(routes_ptr->decode_results_mutex);
                                                auto dit = routes_ptr->decode_results.find(decode_request_id);
                                                if (dit != routes_ptr->decode_results.end() && dit->second.stream) {
                                                    std::lock_guard<std::mutex> slk(dit->second.stream->streaming_mutex);
                                                    dit->second.stream->stream_finished = true;
                                                    dit->second.stream->streaming_cv.notify_all();
                                                }
                                            }
                                            return;
                                        }

                                        // Check if this is a partial or final result
                                        auto * partial = dynamic_cast<server_task_result_cmpl_partial*>(res_ptr.get());
                                        auto * final_r = dynamic_cast<server_task_result_cmpl_final*>(res_ptr.get());

                                        if (partial && !partial->is_begin) {
                                            // Relay partial to streaming queue
                                            std::lock_guard<std::mutex> lk(routes_ptr->decode_results_mutex);
                                            auto dit = routes_ptr->decode_results.find(decode_request_id);
                                            if (dit != routes_ptr->decode_results.end() && dit->second.stream) {
                                                std::lock_guard<std::mutex> slk(dit->second.stream->streaming_mutex);
                                                dit->second.stream->streaming_queue.push_back(std::move(res_ptr));
                                                dit->second.stream->streaming_cv.notify_all();
                                            }
                                        } else if (final_r) {
                                            // Store final result and mark DONE
                                            got_final = true;

                                            server_routes::decode_result_entry entry;
                                            entry.id_slot               = id_slot;
                                            entry.completion_id         = final_r->oaicompat_cmpl_id;
                                            entry.oaicompat_model       = oaicompat_model_name;
                                            entry.content               = final_r->content;
                                            entry.n_decoded             = final_r->n_decoded;
                                            entry.n_prompt_tokens       = final_r->n_prompt_tokens;
                                            entry.n_prompt_tokens_cache = final_r->n_prompt_tokens_cache;
                                            entry.timings               = final_r->timings;
                                            entry.stop                  = final_r->stop;
                                            entry.include_usage         = final_r->include_usage;
                                            entry.match_json            = match_j;
                                            entry.created_at            = std::time(nullptr);
                                            entry.ttl_s                 = routes_ptr->decode_result_ttl_s;

                                            json metrics = json::object();
                                            metrics["decode_request_id"] = decode_request_id;
                                            metrics["id_slot"]           = id_slot;
                                            metrics["n_past"]            = final_r->n_prompt_tokens_cache + final_r->n_decoded;
                                            metrics["decode_ms"]         = final_r->timings.predicted_ms;
                                            metrics["prompt_ms"]         = final_r->timings.prompt_ms;
                                            metrics["model_load_ms"]     = model_load_ms;
                                            metrics["restore_slot_ms"]   = restore_slot_ms;
                                            metrics["model_identity"]    = {
                                                {"tokenizer", resident_tokenizer},
                                                {"model_name", resident_model_name},
                                                {"model_quant", resident_model_quant},
                                                {"model_capabilities", resident_capabilities}
                                            };
                                            metrics["match"]        = match_j;
                                            metrics["model_fallback"] = false;
                                            // Hydra n_common observability
                                            metrics["n_common"]           = entry.n_common;
                                            metrics["n_prompt_processed"] = entry.n_prompt_processed;
                                            metrics["logits_reused"]      = entry.logits_reused;
                                            entry.hydra_metrics = metrics;
                                            entry.state = server_routes::DECODE_STATE_DONE;

                                            // Signal stream finished before storing entry
                                            {
                                                std::lock_guard<std::mutex> lk(routes_ptr->decode_results_mutex);
                                                auto dit = routes_ptr->decode_results.find(decode_request_id);
                                                if (dit != routes_ptr->decode_results.end() && dit->second.stream) {
                                                    // Transfer streaming state to the new entry
                                                    entry.stream = std::move(dit->second.stream);
                                                    {
                                                        std::lock_guard<std::mutex> slk(entry.stream->streaming_mutex);
                                                        entry.stream->stream_finished = true;
                                                    }
                                                    entry.stream->streaming_cv.notify_all();
                                                }
                                            }

                                            std::lock_guard<std::mutex> lock(routes_ptr->decode_results_mutex);
                                            routes_ptr->decode_results[decode_request_id] = std::move(entry);
                                            routes_ptr->evict_decode_results_locked();

                                            SRV_INF("hydra: DECODE_APPLY slot=%d generation complete (request_id=%d, n_decoded=%d)\n",
                                                    id_slot, decode_request_id, final_r->n_decoded);
                                        } else {
                                            // is_begin partial — just consume it
                                        }
                                    }

                                    results.remove_waiting_task_id(completion_id);
                                }).detach();
                            }
                        } else {
                            SRV_WRN("hydra: DECODE_APPLY slot=%d tokenization failed\n", id_slot);
                        }
                    }
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_SET_EXPERT_MODE:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_SET_EXPERT_MODE;

                    // Parse the payload. For backward compatibility, a raw string
                    // ("solo" or "combined") is accepted. Phase D (C# side) sends
                    // a JSON payload: {"mode":"combined","peer":"host:port",...}.
                    std::string requested;
                    std::string peer_override;
                    const std::string & raw = task.hydra_action.expert_mode;
                    if (!raw.empty() && raw[0] == '{') {
                        try {
                            json j = json::parse(raw);
                            requested    = j.value("mode", "solo");
                            peer_override = j.value("peer", "");
                        } catch (...) {
                            requested = "solo";
                        }
                    } else {
                        requested = raw;
                    }

                    if (requested != "solo" && requested != "combined") {
                        res->rpc_status = HYDRA_STATUS_ERROR;
                        res->success = false;
                        res->error = "expert_mode must be 'solo' or 'combined'";
                        queue_results.send(std::move(res));
                        break;
                    }

                    // #29 Phase B: per-request peer switching. If the peer changes,
                    // clean up the old binding and register the new one. The peer
                    // info comes from the SET_EXPERT_MODE control-plane payload
                    // (JSON {"mode":"combined","peer":"host:port"}), NOT from the
                    // HTTP inference body — keeping control and data separate.
                    if (!peer_override.empty() && peer_override != hydra_current_peer) {
                        // Guard: peer switch is unsafe while any slot is decoding.
                        // sched_reserve() destroys and rebuilds the scheduler, which
                        // invalidates in-flight decode state across all slots.
                        bool any_active = false;
                        for (const auto & s : slots) {
                            if (s.is_processing()) { any_active = true; break; }
                        }
                        if (any_active) {
                            SRV_WRN("hydra: cannot switch peers — %zu slot(s) are processing, rejecting SET_EXPERT_MODE\n", slots.size());
                            res->rpc_status = HYDRA_STATUS_BUSY;
                            res->success = false;
                            res->error = "cannot switch peers while slots are processing";
                            queue_results.send(std::move(res));
                            break;
                        }
                        if (!hydra_current_peer.empty()) {
                            SRV_INF("hydra: switching from peer %s to %s — cleaning up old binding\n",
                                    hydra_current_peer.c_str(), peer_override.c_str());
                            ctx_tgt->hydra_remove_combined_rpc_backend(hydra_current_peer.c_str());
                        }
                        hydra_current_peer = peer_override;
                    }

                    // Hydra #383 T1: layer-split (static combined) engines cannot
                    // switch modes at runtime — the split is baked in at model load.
                    // "combined" is a no-op (already combined); "solo" is rejected.
                    if (hydra_combined_static) {
                        if (requested == "solo") {
                            res->rpc_status = HYDRA_STATUS_ERROR;
                            res->success = false;
                            res->error = "combined_static: this engine loaded in layer-split COMBINED mode; cannot switch to solo at runtime";
                            LOG_WRN("srv  %12.*s: hydra: SET_EXPERT_MODE solo rejected — engine is combined_static (layer-split)\n", 12, __func__);
                            queue_results.send(std::move(res));
                            break;
                        }
                        // requested == "combined": success no-op
                        res->expert_mode_applied = "combined";
                        res->rpc_status = HYDRA_STATUS_OK;
                        res->success = true;
                        LOG_INF("srv  %12.*s: hydra: SET_EXPERT_MODE combined no-op — engine is combined_static (layer-split)\n", 12, __func__);
                        queue_results.send(std::move(res));
                        break;
                    }

                    // #368 fix: gate on "configured as combined head" (non-empty
                    // peer addr + OT pattern), NOT on whether the startup
                    // dual-load succeeded. The rebind path below is fail-open —
                    // if the peer is still unreachable it stays solo — so
                    // hydra_combined_head_attached (set only when startup
                    // succeeded) must NOT block the attempt. Hydra #287/#260/#348
                    // intent is preserved: an unconfigured engine (no peer/
                    // pattern) still falls back to solo immediately.
                    const bool want_combined = requested == "combined" &&
                        !hydra_peer.empty() && !hydra_combined_pattern.empty();

                    // #368 (#357 fix): bind-on-activation. Re-bind the peer's
                    // expert tensors on each SET_EXPERT_MODE("combined") request
                    // so a peer that was down at boot is picked up on the first
                    // COMBINED request after it comes up. Fail-open: if the
                    // rebind fails we stay solo and the Coordinator's
                    // ReportsSolo path handles it.
                    bool actually_combined = want_combined;
                    if (want_combined) {
                        if (hydra_peer.empty() || hydra_combined_pattern.empty()) {
                            SRV_WRN("%s\n", "hydra: SET_EXPERT_MODE(combined) but no peer/pattern configured; staying solo");
                            actually_combined = false;
                        } else {
                            // ggml_backend_rpc_add_server is idempotent — returns
                            // the existing reg if the peer was registered before.
                            ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
                            if (!rpc_reg) {
                                SRV_WRN("%s\n", "hydra: SET_EXPERT_MODE(combined) but RPC backend not available; staying solo");
                                actually_combined = false;
                            } else {
                                using add_server_fn_t = ggml_backend_reg_t (*)(const char *);
                                auto add_server_fn = (add_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
                                ggml_backend_reg_t peer_reg = add_server_fn ? add_server_fn(hydra_peer.c_str()) : nullptr;
                                ggml_backend_dev_t  peer_dev = (peer_reg && ggml_backend_reg_dev_count(peer_reg) > 0) ? ggml_backend_reg_dev_get(peer_reg, 0) : nullptr;
                                if (!peer_dev) {
                                    SRV_WRN("hydra: SET_EXPERT_MODE(combined) but peer %s has no registered device; staying solo\n",
                                            hydra_peer.c_str());
                                    actually_combined = false;
                                } else {
                                    int32_t n_bound = llama_hydra_rebind_combined_experts(
                                            ctx_tgt, hydra_peer.c_str(), peer_dev, hydra_combined_pattern.c_str());
                                    if (n_bound <= 0) {
                                        SRV_WRN("hydra: SET_EXPERT_MODE(combined) rebind on peer %s returned %d; staying solo\n",
                                                hydra_peer.c_str(), n_bound);
                                        actually_combined = false;
                                    } else {
                                        // Peer is up — latch so INFO RPC advertises combined.
                                        hydra_combined_head_attached = true;
                                    }
                                }
                            }
                        }
                    }

                    llama_hydra_set_expert_mode(ctx_tgt, actually_combined ? 1 : 0);
                    res->expert_mode_applied = actually_combined ? "combined" : "solo";

                    res->rpc_status = HYDRA_STATUS_OK;
                    res->success = true;
                    SRV_INF("hydra: SET_EXPERT_MODE requested='%s' applied='%s' (slot %d)\n",
                            requested.c_str(), res->expert_mode_applied.c_str(), task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;

            case SERVER_TASK_TYPE_HYDRA_ENGINE_SWAP_QUANT:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_SWAP_QUANT;
                    res->rpc_status = HYDRA_STATUS_OK;
                    res->success = true;
                    SRV_INF("hydra: SWAP_QUANT quant='%s' pattern='%s' (slot %d)\n",
                            task.hydra_action.quant_key.c_str(),
                            task.hydra_action.tensor_pattern.c_str(),
                            task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;

            // M-Perf.9 (#289) / issue #287: PIPELINE_ATTACH is part of the
            // two-engine "work together" routing tracked in #287. The
            // coordinator wires the request; the engine-side scaffolding
            // (--override-tensor local-load, activation passing, COMBINED
            // expert mode) is the next deliverable. For now this opcode
            // returns NOT_IMPLEMENTED so the wire stays in sync — the
            // coordinator will treat that as a fallback to solo mode.
            case SERVER_TASK_TYPE_HYDRA_ENGINE_PIPELINE_ATTACH:
                {
                    auto res = std::make_unique<server_task_result_hydra_engine>();
                    res->id = task.id;
                    res->op = HYDRA_OP_PIPELINE_ATTACH;
                    res->rpc_status = HYDRA_STATUS_NOT_IMPLEMENTED;
                    res->success = false;
                    res->error = "HYDRA_OP_PIPELINE_ATTACH not yet implemented in this build (see issue #287)";
                    SRV_WRN("hydra: PIPELINE_ATTACH received (slot %d) — stubbed, issue #287\n",
                            task.hydra_action.id_slot);
                    queue_results.send(std::move(res));
                } break;
            default:
                break;
        }
    }


// ---------------------------------------------------------------------------
// WS1/WS2: the extension object. handle_task() routes HYDRA tasks to the same
// hydra_process_task() method the legacy switch calls — seam == legacy behavior.
// ---------------------------------------------------------------------------
struct hydra_engine_extension : server_hydra_extension {
    const char * name() const override {
        return "hydra-task-ws2";
    }

    bool handle_task(server_context_impl & impl, server_task & task) override {
        switch (task.type) {
            case SERVER_TASK_TYPE_HYDRA_STATE_GET:
            case SERVER_TASK_TYPE_HYDRA_STATE_PUT:
            case SERVER_TASK_TYPE_HYDRA_STATE_META:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_CONFIGURE:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_INFO:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_PREFILL:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_DECODE:
            case SERVER_TASK_TYPE_HYDRA_DECODE_APPLY:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_SET_EXPERT_MODE:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_SWAP_QUANT:
            case SERVER_TASK_TYPE_HYDRA_ENGINE_PIPELINE_ATTACH:
                impl.hydra_process_task(task);
                return true; // claimed
            default:
                return false; // not a Hydra task — fall through to inline dispatch
        }
    }

    bool pre_loop(server_context_impl & impl) override {
        // WS3: handle the slot-free CONFIGURE/T3 moment at the top of
        // update_slots(). Mirrors the inline all-idle block; returns true only
        // when a staged CONFIGURE was actually applied (so the rest of
        // update_slots() is skipped, matching the inline `return;`).
        bool all_idle = true;
        for (auto & slot : impl.slots) {
            if (slot.is_processing() || slot.hydra_transferring->load()) {
                all_idle = false;
                break;
            }
        }
        if (!all_idle) {
            return false; // not the slot-free moment — inline decode body runs
        }

        if (impl.ctx_tgt && impl.ctx_tgt->hydra_has_pending_config()) {
            SRV_INF("hydra ext: slot-free moment — applying pending CONFIGURE (tier=%s)\n",
                    impl.ctx_tgt->hydra_get_pending_config_tier().c_str());
            impl.apply_pending_hydra_config();
            return true;
        }
        if (!impl.ctx_tgt && impl.first_load_pending) {
            SRV_INF("%s", "hydra ext: slot-free moment — first load (no context yet)\n");
            impl.apply_pending_hydra_config();
            return true;
        }
        return false; // nothing staged — inline all-idle block logs + returns
    }

    bool on_empty_batch(server_context_impl & impl) override {
        // WS3: replicate the inline empty-batch transfer-suppression, including
        // the 2s grace window after a transfer ends. Returns true when the
        // empty batch was caused by a STATE_GET transfer (suppress the abort);
        // false otherwise (inline logic runs, which eventually aborts).
        static int64_t hydra_last_transfer_ms = 0;
        static int64_t hydra_suppress_count   = 0;

        bool any_transferring = false;
        for (const auto & s : impl.slots) {
            if (s.hydra_transferring && s.hydra_transferring->load()) {
                any_transferring = true;
                break;
            }
        }
        const int64_t now_ms = ggml_time_us() / 1000;
        if (any_transferring) {
            hydra_last_transfer_ms = now_ms;
        }
        if (any_transferring || now_ms - hydra_last_transfer_ms < 2000) {
            if (hydra_suppress_count++ % 256 == 0) {
                SRV_WRN("hydra ext: empty batch suppressed — transfer %s\n",
                        any_transferring ? "in flight" : "just ended");
            }
            impl.n_empty_consecutive = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return true;
        }
        return false;
    }
};

std::unique_ptr<server_hydra_extension> hydra_create_extension() {
    return std::make_unique<hydra_engine_extension>();
}

// ---------------------------------------------------------------------------
// WS3.5: Hydra RPC server (moved from server-context.cpp, epic #610).
// Same TU via the bottom #include, so hydra_rpc_ctx and the static handlers
// can reach server_context_impl privates (friend) and the file-scope
// queue/response objects. Wire format: specs/rpc-protocol.md | server-rpc.h.
// ---------------------------------------------------------------------------
// ═══════════════════════════════════════════════════════════════════════════════
// Hydra RPC server — KV state transfer (M1: task-queue based)
// Wire format: specs/rpc-protocol.md  |  constants: server-rpc.h
// Ops implemented: STATE_GET (0x30), STATE_PUT (0x31), STATE_META (0x32)
// M1: All llama API calls routed through task queue (inference thread safe)
// ═══════════════════════════════════════════════════════════════════════════════

#if !defined(_WIN32)

// ── Context for RPC thread — pass to handlers ─────────────────────────────────

struct hydra_rpc_ctx {
    server_queue * queue_tasks = nullptr;
    server_response * queue_results = nullptr;
};

// ── Low-level I/O helpers ─────────────────────────────────────────────────────

// Hydra #43: failures here were previously silent — every caller treats a
// `false` return as "give up" but none logged *why*, so a wedged RPC
// response looked identical to a client that vanished. Log once, centrally,
// instead of touching the ~30 call sites.
static bool hydra_recv_all(int fd, void * buf, size_t n) {
    char * p = reinterpret_cast<char *>(buf);
    const size_t total = n;
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, 0);
        if (r < 0) {
            SRV_WRN("hydra rpc: recv failed on fd=%d (%zu/%zu bytes): %s\n",
                    fd, total - n, total, std::strerror(errno));
            return false;
        }
        if (r == 0) {
            SRV_DBG("hydra rpc: recv EOF on fd=%d (%zu/%zu bytes)\n", fd, total - n, total);
            return false;
        }
        p += r; n -= r;
    }
    return true;
}

static bool hydra_send_all(int fd, const void * buf, size_t n) {
    const char * p = reinterpret_cast<const char *>(buf);
    const size_t total = n;
    while (n > 0) {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) {
            SRV_WRN("hydra rpc: send failed on fd=%d (%zu/%zu bytes) w=%zd: %s\n",
                    fd, total - n, total, w, std::strerror(errno));
            return false;
        }
        p += w; n -= w;
    }
    return true;
}

// Response header: status(1) | meta_len(3 LE uint24) | payload_len(8 LE) — 12 bytes
static void hydra_write_res(int fd, uint8_t status, uint32_t meta_len, uint64_t payload_len) {
    uint8_t buf[HYDRA_RES_HEADER_SIZE] = {};
    buf[0] = status;
    buf[1] = (meta_len)       & 0xFF;
    buf[2] = (meta_len >>  8) & 0xFF;
    buf[3] = (meta_len >> 16) & 0xFF;
    memcpy(buf + 4, &payload_len, 8); // little-endian (x86/arm64)
    hydra_send_all(fd, buf, HYDRA_RES_HEADER_SIZE);
}

// ── Op handlers (M1: dispatch via task queue) ─────────────────────────────────

// STATE_GET (0x30): Post task, wait for result.
//
// M1 path (hydra_fd < 0): inference thread serializes 800 MB into result buffer;
//   RPC thread sends response header + meta JSON + buffer here.
//
// M2 path (hydra_fd = fd): background thread streams GPU→socket directly using
//   llama_state_seq_get_data_to_fd; result carries only n_past + streamed_bytes.
//   Response header + meta are sent BEFORE the task (we know size from STATE_META),
//   so the payload is already on the wire before we even get the result back.
//   Actually: we must send header AFTER knowing state_size. So:
//   - If M2: we get state_size first from a quick STATE_META query (n_past already known),
//     OR we embed state_size in the result from get_size() on the inference thread.
//   The inference thread always calls llama_state_seq_get_size (cheap) and stores it
//   in res->state_size for M2 so we can send the header before the stream completes.
//
// Timeout: 30s — streaming 800 MB over localhost may take a few seconds.
static void hydra_handle_state_get(int fd, int slot_id, const hydra_rpc_ctx & ctx) {
    // Build task — pass fd for M2 zero-copy streaming
    server_task task(SERVER_TASK_TYPE_HYDRA_STATE_GET);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot  = slot_id;
    task.hydra_action.hydra_fd = fd;    // M2: background thread streams here
    const int task_id = task.id;
    // Register BEFORE posting — server_response::send() silently drops results
    // for ids not in waiting_task_ids.
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    // Wait for result (n_past + state_size always set; state_data only on M1)
    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 30); // seconds
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        SRV_WRN("hydra rpc: STATE_GET timeout for slot %d\n", slot_id);
        // M2 caveat: the background thread may own the fd (header possibly sent);
        // writing an error header here could interleave with the stream. Shut the
        // socket down instead so the client unblocks with a clean EOF.
        ::shutdown(fd, SHUT_RDWR);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_state*>(res_ptr.get());
    if (!res) {
        SRV_WRN("hydra rpc: STATE_GET result type mismatch for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    if (res->rpc_status != HYDRA_STATUS_OK) {
        if (res->header_sent) {
            // M2 failure: header already sent but stream failed; background thread
            // shut the socket down — connection loop will close the fd on next read.
            // Log and return without sending a second response header.
            SRV_WRN("hydra rpc: STATE_GET slot=%d M2 stream failed: %s\n",
                    slot_id, res->error.c_str());
            return;
        }
        hydra_write_res(fd, res->rpc_status, 0, 0);
        if (!res->error.empty()) {
            hydra_send_all(fd, res->error.data(), res->error.size());
        }
        return;
    }

    if (res->streamed_bytes > 0) {
        // M2 path: data already on the wire — response header + meta were sent by background thread.
        // Nothing left for RPC thread to do. The protocol framing (header + meta + payload)
        // was completed inside llama_io_write_socket / the background thread.
        // Note: header was sent AFTER state_size was known (inference thread called get_size).
        SRV_INF("hydra rpc: STATE_GET slot=%d M2 streamed %.1f MiB directly\n",
                slot_id, res->streamed_bytes / (1024.0 * 1024.0));
    } else {
        // M1 path: inference thread buffered 800 MB; send it now.
        const uint64_t payload = (uint64_t)res->state_data.size();
        json meta_j;
        meta_j["n_past"]     = res->n_past;
        meta_j["state_size"] = payload;
        if (!res->model_alias.empty()) meta_j["model_alias"] = res->model_alias;
        if (!res->model_path.empty())  meta_j["model_path"]  = res->model_path;
        if (!res->tokenizer.empty())   meta_j["tokenizer"]   = res->tokenizer;
        if (!res->model_name.empty())  meta_j["model_name"]  = res->model_name;
        if (!res->model_quant.empty()) meta_j["model_quant"] = res->model_quant;
        if (res->model_capabilities)   meta_j["model_capabilities"] = res->model_capabilities;
        const std::string meta_str = meta_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), payload);
        hydra_send_all(fd, meta_str.data(), meta_str.size());
        hydra_send_all(fd, res->state_data.data(), (size_t)payload);
        SRV_INF("hydra rpc: STATE_GET slot=%d M1 sent %.1f MiB from buffer\n",
                slot_id, payload / (1024.0 * 1024.0));
    }
}

// STATE_PUT (0x31): Receive payload, post task, wait for result, send ack.
static void hydra_handle_state_put(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    if (payload_len > HYDRA_MAX_STATE_BYTES) {
        SRV_WRN("hydra rpc: STATE_PUT payload %" PRIu64 " B exceeds cap %" PRIu64 " B\n",
                payload_len, HYDRA_MAX_STATE_BYTES);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        // Drain to keep persistent connection alive
        std::vector<uint8_t> drain(65536);
        for (uint64_t rem = payload_len; rem > 0; ) {
            size_t chunk = (size_t)std::min(rem, (uint64_t)drain.size());
            if (!hydra_recv_all(fd, drain.data(), chunk)) break;
            rem -= chunk;
        }
        return;
    }

    // Read payload from socket
    std::vector<uint8_t> buf((size_t)payload_len);
    if (!hydra_recv_all(fd, buf.data(), (size_t)payload_len)) {
        SRV_WRN("%s", "hydra rpc: STATE_PUT failed to read payload\n");
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Post task to inference thread
    server_task task(SERVER_TASK_TYPE_HYDRA_STATE_PUT);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.erase_existing = true; // RPC restore always replaces slot state
    task.hydra_action.state_data = std::move(buf);
    const int task_id = task.id;
    // Register BEFORE posting — results for unregistered ids are dropped.
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    // Wait for result from inference thread (30s timeout for large restore)
    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 30); // seconds
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        SRV_WRN("hydra rpc: STATE_PUT timeout for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_state*>(res_ptr.get());
    if (!res) {
        SRV_WRN("hydra rpc: STATE_PUT result type mismatch for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Send result back to client
    uint8_t rpc_status = res->rpc_status;
    if (rpc_status == HYDRA_STATUS_OK) {
        json meta_j;
        meta_j["restored"]    = true;
        meta_j["bytes"]       = res->bytes;
        meta_j["model_match"] = res->model_match;
        if (!res->model_alias.empty()) meta_j["model_alias"] = res->model_alias;
        if (!res->model_path.empty())  meta_j["model_path"]  = res->model_path;
        if (!res->tokenizer.empty())   meta_j["tokenizer"]   = res->tokenizer;
        if (!res->model_name.empty())  meta_j["model_name"]  = res->model_name;
        if (!res->model_quant.empty()) meta_j["model_quant"] = res->model_quant;
        if (res->model_capabilities)   meta_j["model_capabilities"] = res->model_capabilities;
        const std::string meta_str = meta_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
        hydra_send_all(fd, meta_str.data(), meta_str.size());
    } else {
        json err_j;
        err_j["error"] = res->error;
        const std::string err_str = err_j.dump();
        hydra_write_res(fd, rpc_status, (uint32_t)err_str.size(), 0);
        hydra_send_all(fd, err_str.data(), err_str.size());
    }
}

// STATE_META (0x32): Post task, wait for result, send JSON metadata.
static void hydra_handle_state_meta(int fd, int slot_id, const hydra_rpc_ctx & ctx) {
    server_task task(SERVER_TASK_TYPE_HYDRA_STATE_META);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    const int task_id = task.id;
    // Register BEFORE posting — results for unregistered ids are dropped.
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    // Wait for result from inference thread (5s timeout — allows for queue congestion)
    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5); // seconds
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        SRV_WRN("hydra rpc: STATE_META timeout for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_state*>(res_ptr.get());
    if (!res) {
        SRV_WRN("hydra rpc: STATE_META result type mismatch for slot %d\n", slot_id);
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Send result back to client
    uint8_t rpc_status = res->rpc_status;
    if (rpc_status == HYDRA_STATUS_OK) {
        json meta_j;
        meta_j["slot_id"]         = res->id_slot;
        meta_j["n_past"]          = res->n_past;
        meta_j["state_size"]      = res->state_size;
        meta_j["is_processing"]   = res->is_processing;
        meta_j["is_transferring"] = res->is_transferring;
        if (!res->model_alias.empty()) meta_j["model_alias"] = res->model_alias;
        if (!res->model_path.empty())  meta_j["model_path"]  = res->model_path;
        if (!res->tokenizer.empty())   meta_j["tokenizer"]   = res->tokenizer;
        if (!res->model_name.empty())  meta_j["model_name"]  = res->model_name;
        if (!res->model_quant.empty()) meta_j["model_quant"] = res->model_quant;
        if (res->model_capabilities)   meta_j["model_capabilities"] = res->model_capabilities;
        const std::string meta_str = meta_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
        hydra_send_all(fd, meta_str.data(), meta_str.size());
    } else {
        hydra_write_res(fd, rpc_status, 0, 0);
    }
}

// ── E1 Engine control handlers ────────────────────────────────────────────────

// CONFIGURE (0x33): Read JSON config payload, post task, return success.
static void hydra_handle_configure(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    std::string config_json(payload_len, '\0');
    if (payload_len > 0 && !hydra_recv_all(fd, config_json.data(), payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_CONFIGURE);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.config_json = std::move(config_json);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || !res->success) {
        // hydra#406: on failure, include the error message in the meta so
        // the Coordinator can distinguish "drain timeout" from a parse
        // error. We still write HYDRA_STATUS_ERROR (0x02) per the wire
        // contract — the meta body is for diagnostics only.
        if (res && !res->error.empty()) {
            json err_j = {{"success", false}, {"error", res->error}};
            if (!res->tier.empty()) err_j["tier"] = res->tier;
            const std::string err_str = err_j.dump();
            hydra_write_res(fd, HYDRA_STATUS_ERROR, (uint32_t)err_str.size(), 0);
            hydra_send_all(fd, err_str.data(), err_str.size());
        } else {
            hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        }
        return;
    }

    // hydra#406: tiered CONFIGURE response shape. Always present: success,
    // tier, params_applied (T1 keys), deferred_keys (T2/T3 keys).
    json meta_j = {
        {"success",        true},
        {"tier",           res->tier.empty() ? std::string("T1") : res->tier},
        {"params_applied", json::object()},
        {"deferred_keys",  json::array()},
    };
    for (const auto & kv : res->params_applied) {
        meta_j["params_applied"][kv.first] = kv.second;
    }
    for (const auto & k : res->deferred_keys) {
        meta_j["deferred_keys"].push_back(k);
    }
    // hydra#334: echo the post-clamp value for the state_chunk_size legacy
    // path so the Coordinator's existing detection logic still works
    // (the same value is also in params_applied, with the dotted key).
    if (res->state_chunk_size_applied > 0) {
        meta_j["state_chunk_size_applied"] = res->state_chunk_size_applied;
    }
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// INFO (0x34): Return engine capabilities as JSON.
static void hydra_handle_info(int fd, int slot_id, const hydra_rpc_ctx & ctx) {
    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_INFO);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    const std::string & info_str = res->info_json;
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)info_str.size(), 0);
    hydra_send_all(fd, info_str.data(), info_str.size());
}

// PREFILL (0x35): Read JSON payload with {"messages": [...]},
// tokenize internally, run prefill, return n_past + KV state blob.
static void hydra_handle_prefill(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    if (payload_len == 0) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    std::string json_str((size_t)payload_len, '\0');
    if (!hydra_recv_all(fd, json_str.data(), (size_t)payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_PREFILL);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.request_json = std::move(json_str);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    // Bumped from 60s to 180s. Prefill for 32k+ token prompts exceeds 120s
    // (we measured 32s for 22k tokens; 48k ≈ 70s, 100k ≈ 150s+). Long autoregressive
    // decode on P100 (28 tok/s) for 4k+ token outputs also exceeds 120s. The C++
    // side was timing out and returning HYDRA_STATUS_ERROR before the C# client
    // gave up, surfacing as a 503 from the coordinator even though the model was
    // still working.
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 180);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || res->rpc_status != HYDRA_STATUS_OK) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Return n_past + sizes + model identity in meta; full blob (v2 header + KV + logits) as payload.
    // logits_size > 0 signals the decode GPU to inject them into ctx->logits via STATE_PUT.
    // M-Perf.9 #289: model identity fields (already populated on res by the PREFILL handler)
    // are included so the Coordinator can record which model built the KV.
    json meta_j = {
        {"n_past",      res->n_past},
        {"state_size",  res->state_size},
        {"logits_size", res->logits_size}
    };
    if (!res->model_alias.empty()) meta_j["model_alias"] = res->model_alias;
    if (!res->model_path.empty())  meta_j["model_path"]  = res->model_path;
    if (!res->tokenizer.empty())   meta_j["tokenizer"]   = res->tokenizer;
    if (!res->model_name.empty())  meta_j["model_name"]  = res->model_name;
    if (!res->model_quant.empty()) meta_j["model_quant"] = res->model_quant;
    if (res->model_capabilities)   meta_j["model_capabilities"] = res->model_capabilities;
    meta_j["model_fallback"] = res->model_fallback;
    if (res->prefill_ms > 0)     meta_j["prefill_ms"]     = res->prefill_ms;
    if (res->model_load_ms > 0)  meta_j["model_load_ms"]  = res->model_load_ms;
    const std::string meta_str = meta_j.dump();
    const uint64_t total_payload = (uint64_t)res->state_data.size();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), total_payload);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
    if (total_payload > 0) {
        hydra_send_all(fd, res->state_data.data(), (size_t)total_payload);
    }
    SRV_INF("hydra: PREFILL slot=%d sent n_past=%d kv=%" PRIu64 "B logits=%" PRIu64 "B total=%" PRIu64 "B\n",
            slot_id, res->n_past, res->state_size, res->logits_size, total_payload);
}

// DECODE (0x43) — Merged P/D: framed request with async HTTP retrieval.
// Wire format v3 (segmented):
//   [4B hdr_len LE]       <= 32768
//   [8B  hdr_hash LE]     xxh3-64 of the hdr JSON bytes that follow
//   [hdr_len bytes]       control header JSON
//   [prompt_len bytes]    prompt JSON segment (may be zero-length)
//   [kv_len bytes]        raw KV blob (may be zero-length)
//
// Control header:
//   { "v": 3, "model": "...", "kv_metadata": {...}, "model_metadata": {...},
//     "generation": {...}, "segments": [...] }
//
// Two-phase flow:
//   Phase 1 (sync): identity validation + KV restore — waits for inference thread
//   Phase 2 (async): background thread posts SERVER_TASK_TYPE_COMPLETION,
//           update_slots() drives generation, result stored in decode_results buffer.
// Actual result retrieved via GET /v1/decode/{decode_request_id}.
static void hydra_handle_decode(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    // ── Read frame header: [4B hdr_len][8B hdr_hash] ──────────────────────
    if (payload_len < sizeof(uint32_t) + sizeof(uint64_t)) {
        SRV_WRN("%s", "hydra rpc: DECODE payload too small for frame header\n");
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    uint32_t hdr_len = 0;
    if (!hydra_recv_all(fd, &hdr_len, sizeof(hdr_len))) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    if (hdr_len > HYDRA_MAX_JSON_HEADER) {
        SRV_WRN("hydra rpc: DECODE hdr_len %u B exceeds cap %u B\n",
                hdr_len, HYDRA_MAX_JSON_HEADER);
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    uint64_t hdr_hash = 0;
    if (!hydra_recv_all(fd, &hdr_hash, sizeof(hdr_hash))) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // ── Read control header JSON ──────────────────────────────────────────
    std::string hdr_json_str(hdr_len, '\0');
    if (hdr_len > 0 && !hydra_recv_all(fd, hdr_json_str.data(), hdr_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Verify hdr_hash (xxh3-64 of the JSON bytes)
    {
        const uint64_t computed = XXH3_64bits(hdr_json_str.data(), hdr_json_str.size());
        if (computed != hdr_hash) {
            SRV_WRN("hydra rpc: DECODE HDR_HASH_MISMATCH expected=%016" PRIx64 " got=%016" PRIx64 "\n",
                    hdr_hash, computed);
            hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
            return;
        }
    }

    // Parse control header
    json req;
    try {
        req = json::parse(hdr_json_str);
    } catch (const std::exception & e) {
        SRV_WRN("hydra rpc: DECODE invalid JSON in control header: %s\n", e.what());
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    // Validate version
    const int hdr_version = req.value("v", 0);
    if (hdr_version < 3) {
        SRV_WRN("hydra rpc: DECODE unsupported version %d (need >= 3)\n", hdr_version);
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    // Validate required fields
    if (!req.contains("kv_metadata")) {
        SRV_WRN("%s", "hydra rpc: DECODE missing kv_metadata in control header\n");
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }
    if (!req.contains("segments") || !req["segments"].is_array()) {
        SRV_WRN("%s", "hydra rpc: DECODE missing or invalid segments array\n");
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    // ── Parse and validate segment table ──────────────────────────────────
    const json & segments = req["segments"];
    const size_t n_segments = segments.size();
    if (n_segments > 3) {
        SRV_WRN("hydra rpc: DECODE SEGMENT_TABLE_INVALID: too many segments (%zu)\n", n_segments);
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    // Each segment: {"id":"prompt"|"kv", "offset":N, "len":N, "hash":"xxh3:HEX"}
    uint64_t prompt_len = 0;
    uint64_t kv_len = 0;
    std::string prompt_hash_str;
    std::string kv_hash_str;
    uint64_t expected_offset = 0;
    for (size_t i = 0; i < n_segments; i++) {
        const json & seg = segments[i];
        if (!seg.contains("id") || !seg.contains("offset") || !seg.contains("len") || !seg.contains("hash")) {
            SRV_WRN("hydra rpc: DECODE SEGMENT_TABLE_INVALID: segment %zu missing required fields\n", i);
            hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
            return;
        }
        const std::string id = seg["id"].get<std::string>();
        const uint64_t offset = seg["offset"].get<uint64_t>();
        const uint64_t len = seg["len"].get<uint64_t>();
        const std::string hash = seg["hash"].get<std::string>();

        if (offset != expected_offset) {
            SRV_WRN("hydra rpc: DECODE SEGMENT_TABLE_INVALID: segment %zu offset=%" PRIu64 " expected=%" PRIu64 "\n",
                    i, offset, expected_offset);
            hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
            return;
        }
        expected_offset = offset + len;

        if (id == "prompt") {
            prompt_len = len;
            prompt_hash_str = hash;
        } else if (id == "kv") {
            kv_len = len;
            kv_hash_str = hash;
        } else {
            SRV_WRN("hydra rpc: DECODE SEGMENT_TABLE_INVALID: unknown segment id '%s'\n", id.c_str());
            hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
            return;
        }
    }

    // Verify total segment size matches remaining payload
    const uint64_t segments_total = prompt_len + kv_len;
    const uint64_t remaining_after_hdr = payload_len - sizeof(uint32_t) - sizeof(uint64_t) - hdr_len;
    if (segments_total != remaining_after_hdr) {
        SRV_WRN("hydra rpc: DECODE SEGMENT_TABLE_INVALID: segments total %" PRIu64 " != remaining %" PRIu64 "\n",
                segments_total, remaining_after_hdr);
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    // Caps
    if (prompt_len > HYDRA_MAX_PROMPT_BYTES) {
        SRV_WRN("hydra rpc: DECODE PROMPT_TOO_LARGE %" PRIu64 " > %" PRIu64 "\n",
                prompt_len, HYDRA_MAX_PROMPT_BYTES);
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }
    if (kv_len > HYDRA_MAX_STATE_BYTES) {
        SRV_WRN("hydra rpc: DECODE KV_TOO_LARGE %" PRIu64 " > %" PRIu64 "\n",
                kv_len, HYDRA_MAX_STATE_BYTES);
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
        return;
    }

    // ── Read prompt segment ───────────────────────────────────────────────
    std::vector<uint8_t> prompt_data((size_t)prompt_len);
    if (prompt_len > 0 && !hydra_recv_all(fd, prompt_data.data(), (size_t)prompt_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // ── Read KV segment (may be zero-length) ──────────────────────────────
    std::vector<uint8_t> kv_data((size_t)kv_len);
    if (kv_len > 0 && !hydra_recv_all(fd, kv_data.data(), (size_t)kv_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    // Verify KV segment hash BEFORE passing to llama_state_seq_set_data
    if (kv_len > 0 && !kv_hash_str.empty()) {
        // Parse "xxh3:HEX" format
        if (kv_hash_str.rfind("xxh3:", 0) == 0) {
            const std::string hex_str = kv_hash_str.substr(5);
            uint64_t expected_kv_hash = 0;
            try {
                expected_kv_hash = std::stoull(hex_str, nullptr, 16);
            } catch (const std::exception &) {
                SRV_WRN("hydra rpc: DECODE invalid KV hash format: %s\n", kv_hash_str.c_str());
                hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
                return;
            }
            const uint64_t computed_kv = XXH3_64bits(kv_data.data(), kv_data.size());
            if (computed_kv != expected_kv_hash) {
                SRV_WRN("hydra rpc: DECODE SEGMENT_HASH_MISMATCH kv expected=%016" PRIx64 " got=%016" PRIx64 "\n",
                        expected_kv_hash, computed_kv);
                hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
                return;
            }
            SRV_INF("hydra rpc: DECODE KV hash verified (%" PRIu64 " B)\n", kv_len);
        } else {
            SRV_WRN("hydra rpc: DECODE unsupported KV hash prefix: %s\n", kv_hash_str.c_str());
            hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
            return;
        }
    }

    // ── Build decode_json from control header + prompt segment ─────────────
    // The prompt JSON segment may contain { "prompt": "..." } or { "messages": [...] }
    // Merge it into the control header as decode_req["prompt"].
    // Also merge generation params from control header's "generation" key.
    json decode_req = req; // control header already has kv_metadata, model, etc.
    json prompt_obj;
    if (prompt_len > 0) {
        try {
            prompt_obj = json::parse(std::string(prompt_data.begin(), prompt_data.end()));
        } catch (const std::exception & e) {
            SRV_WRN("hydra rpc: DECODE invalid prompt segment JSON: %s\n", e.what());
            hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, 0, 0);
            return;
        }
    }
    // The coordinator sends the prompt segment as the BARE messages array
    // (item.Request["messages"].ToString()). The generation-merge below and
    // DECODE_APPLY's chat-template path both expect an OBJECT with a
    // "messages" key — merging generation keys into an array throws
    // nlohmann::type_error, which was silently swallowed by the RPC worker
    // pool (the connection leaked, no response written, coordinator timed out
    // after 180s). Wrap a bare array so the prompt object matches the
    // downstream contract.
    if (prompt_obj.is_array()) {
        json wrapped;
        wrapped["messages"] = std::move(prompt_obj);
        prompt_obj = std::move(wrapped);
    }
    // Merge generation params from control header into prompt object
    std::string decode_json_str;
    try {
        if (req.contains("generation") && req["generation"].is_object()) {
            const json & gen = req["generation"];
            for (auto it = gen.begin(); it != gen.end(); ++it) {
                if (!prompt_obj.contains(it.key())) {
                    prompt_obj[it.key()] = it.value();
                }
            }
        }
        decode_req["prompt"] = std::move(prompt_obj);

        decode_json_str = decode_req.dump();
    } catch (const std::exception & e) {
        // Never let a malformed prompt object leak the connection: the worker
        // pool swallows exceptions and the fd stays open with no response,
        // hanging the coordinator until its own timeout. Always write an
        // error frame so the caller sees a terminal (retryable-free) result.
        SRV_WRN("hydra rpc: DECODE prompt build failed (slot %d): %s\n", slot_id, e.what());
        json err_j = {
            {"error", std::string("prompt build failed: ") + e.what()},
            {"decode_request_id", -1},
        };
        const std::string err_str = err_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_BAD_REQUEST, (uint32_t) err_str.size(), 0);
        hydra_send_all(fd, err_str.data(), err_str.size());
        return;
    }

    // ── Phase 1: sync validate + restore ──────────────────────────────────
    const int32_t decode_request_id = ctx.queue_tasks->get_new_id();

    server_task val_task(SERVER_TASK_TYPE_HYDRA_ENGINE_DECODE);
    val_task.id = decode_request_id;
    val_task.hydra_action.id_slot = slot_id;
    val_task.hydra_action.decode_json = std::move(decode_json_str);
    val_task.hydra_action.kv_data = std::move(kv_data);
    val_task.hydra_action.decode_request_id = decode_request_id;
    ctx.queue_results->add_waiting_task_id(decode_request_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(val_task));

    // Wait for validation+restore to complete (30s timeout for large KV blobs)
    std::unordered_set<int> val_ids = {decode_request_id};
    auto val_res_ptr = ctx.queue_results->recv_with_timeout(val_ids, 30);
    ctx.queue_results->remove_waiting_task_id(decode_request_id);

    if (!val_res_ptr) {
        SRV_WRN("hydra rpc: DECODE validation timeout for slot %d (request_id=%d)\n",
                slot_id, decode_request_id);
        json err_j = {
            {"error", "validation timeout"},
            {"decode_request_id", decode_request_id},
        };
        const std::string err_str = err_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_ERROR, (uint32_t)err_str.size(), 0);
        hydra_send_all(fd, err_str.data(), err_str.size());
        return;
    }

    auto * val_res = dynamic_cast<server_task_result_hydra_engine*>(val_res_ptr.get());
    if (!val_res || val_res->rpc_status != HYDRA_STATUS_OK) {
        json err_j = {
            {"valid", false},
            {"decode_request_id", decode_request_id},
        };
        if (val_res) {
            if (!val_res->match_json.is_null()) err_j["match"] = val_res->match_json;
            if (!val_res->error.empty()) err_j["reason"] = val_res->error;
            err_j["error_code"] = "CAP_MISMATCH";
        }
        const std::string err_str = err_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_ERROR, (uint32_t)err_str.size(), 0);
        hydra_send_all(fd, err_str.data(), err_str.size());
        return;
    }

    // Validation passed — build real success response
    json meta_j = {
        {"valid", true},
        {"match", val_res->match_json},
        {"decode_request_id", decode_request_id},
        {"n_past_after_restore", val_res->n_past},
        {"restore_slot_ms", val_res->restore_slot_ms},
    };
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());

    SRV_INF("hydra: DECODE slot=%d accepted, request_id=%d, restore=%.1fms\n",
            slot_id, decode_request_id, val_res->restore_slot_ms);
}

// SET_EXPERT_MODE (0x37): Read mode string, post task, return success.
static void hydra_handle_set_expert_mode(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    std::string mode(payload_len, '\0');
    if (payload_len > 0 && !hydra_recv_all(fd, mode.data(), payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_SET_EXPERT_MODE);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.expert_mode = std::move(mode);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || !res->success) {
        const std::string err = (res && !res->error.empty()) ? res->error : std::string();
        json err_j = {{"success", false}};
        if (!err.empty()) err_j["error"] = err;
        const std::string err_str = err_j.dump();
        hydra_write_res(fd, HYDRA_STATUS_ERROR, (uint32_t)err_str.size(), 0);
        hydra_send_all(fd, err_str.data(), err_str.size());
        return;
    }

    // Report the ACTUAL mode applied (may be "solo" even though "combined" was
    // requested, if this engine never dual-loaded combined experts) — the
    // Coordinator's ReportsSolo() reads this key to detect the fallback.
    json meta_j = {{"success", true}, {"mode", res->expert_mode_applied}};
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// SWAP_QUANT (0x38): Read quant_key + tensor_pattern, post task, return success.
static void hydra_handle_swap_quant(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    if (payload_len < sizeof(uint16_t)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    uint16_t quant_key_len = 0;
    if (!hydra_recv_all(fd, &quant_key_len, sizeof(quant_key_len))) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    std::string quant_key(quant_key_len, '\0');
    if (quant_key_len > 0 && !hydra_recv_all(fd, quant_key.data(), quant_key_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    const uint64_t pattern_len = payload_len - sizeof(uint16_t) - quant_key_len;
    std::string tensor_pattern(pattern_len, '\0');
    if (pattern_len > 0 && !hydra_recv_all(fd, tensor_pattern.data(), pattern_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_SWAP_QUANT);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.quant_key = std::move(quant_key);
    task.hydra_action.tensor_pattern = std::move(tensor_pattern);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 30);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    if (!res || !res->success) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    json meta_j = {{"success", true}};
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, HYDRA_STATUS_OK, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// PIPELINE_ATTACH (0x46): M-Perf.9 (#289) / issue #287 — two-engine "work
// together" routing scaffolding. The C# Coordinator sends the peer address
// and the --override-tensor regex; the engine should load the assigned
// tensor slice from its OWN local model (no weight transfer). This opcode
// is stubbed for now (returns NOT_IMPLEMENTED) — full implementation is
// tracked under issue #287.
static void hydra_handle_pipeline_attach(int fd, int slot_id, uint64_t payload_len, const hydra_rpc_ctx & ctx) {
    std::string json_body(payload_len, '\0');
    if (payload_len > 0 && !hydra_recv_all(fd, json_body.data(), payload_len)) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    server_task task(SERVER_TASK_TYPE_HYDRA_ENGINE_PIPELINE_ATTACH);
    task.id = ctx.queue_tasks->get_new_id();
    task.hydra_action.id_slot = slot_id;
    task.hydra_action.request_json = std::move(json_body);
    const int task_id = task.id;
    ctx.queue_results->add_waiting_task_id(task_id);
    ctx.queue_tasks->wait_until_no_sleep();
    ctx.queue_tasks->post(std::move(task));

    std::unordered_set<int> task_ids = {task_id};
    auto res_ptr = ctx.queue_results->recv_with_timeout(task_ids, 5);
    ctx.queue_results->remove_waiting_task_id(task_id);
    if (!res_ptr) {
        hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        return;
    }

    auto * res = dynamic_cast<server_task_result_hydra_engine*>(res_ptr.get());
    // Stubbed: server returns NOT_IMPLEMENTED until issue #287 lands.
    // Propagate that status to the client so the Coordinator can
    // distinguish "not yet built" from a real error and fall back to solo.
    const uint8_t status = (res && res->rpc_status == HYDRA_STATUS_NOT_IMPLEMENTED)
        ? HYDRA_STATUS_NOT_IMPLEMENTED : HYDRA_STATUS_ERROR;
    json meta_j;
    if (res && !res->error.empty()) meta_j["error"] = res->error;
    meta_j["success"] = res && res->success;
    const std::string meta_str = meta_j.dump();
    hydra_write_res(fd, status, (uint32_t)meta_str.size(), 0);
    hydra_send_all(fd, meta_str.data(), meta_str.size());
}

// ── Per-connection loop ───────────────────────────────────────────────────────
// Persistent: one TCP connection handles many sequential requests.

static void hydra_handle_connection(int fd, const hydra_rpc_ctx & ctx) {
    // Set receive timeout to prevent hung connections on stalled clients
    struct timeval tv;
    tv.tv_sec  = 120; // 2 min inactivity timeout
    tv.tv_usec = 0;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    while (true) {
        uint8_t hdr[HYDRA_REQ_HEADER_SIZE];
        if (!hydra_recv_all(fd, hdr, HYDRA_REQ_HEADER_SIZE)) break;

        uint16_t magic = 0;
        memcpy(&magic, hdr + 0, 2);
        if (magic != HYDRA_MAGIC) {
            SRV_WRN("hydra rpc: bad magic 0x%04x — closing connection\n", (unsigned)magic);
            break;
        }

        const uint8_t op = hdr[2];
        // hdr[3] = flags (reserved, unused in M1)
        uint16_t key_len = 0, trace_len = 0;
        uint64_t payload_len = 0;
        memcpy(&key_len,     hdr + 4,  2);
        memcpy(&payload_len, hdr + 6,  8);
        memcpy(&trace_len,   hdr + 14, 2);

        std::string key(key_len, '\0');
        std::string trace_id(trace_len, '\0');
        if (!hydra_recv_all(fd, key.data(),      key_len))   break;
        if (!hydra_recv_all(fd, trace_id.data(), trace_len)) break;

        // Slot-key parsing: engine-level opcodes (INFO, CONFIGURE, SET_EXPERT_MODE,
        // SWAP_QUANT) don't need a valid slot — use slot_id = 0 when the key is
        // empty or invalid. Slot-level opcodes (STATE_GET, STATE_PUT, STATE_META,
        // PREFILL, DECODE) still require a valid integer key.
        int slot_id = -1;
        bool is_engine_level_op = (op == HYDRA_OP_INFO || op == HYDRA_OP_CONFIGURE ||
                                   op == HYDRA_OP_SET_EXPERT_MODE || op == HYDRA_OP_SWAP_QUANT);
        if (key.empty() && is_engine_level_op) {
            slot_id = 0;
        } else {
            try { slot_id = std::stoi(key); }
            catch (...) {
                SRV_WRN("hydra rpc: invalid slot key '%s'\n", key.c_str());
                hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
                continue;
            }
        }

        // Dispatch to handler via task queue (no direct slot access)
        switch (op) {
            case HYDRA_OP_STATE_GET:
                SRV_DBG("hydra rpc: STATE_GET  slot=%d trace=%s\n", slot_id, trace_id.c_str());
                hydra_handle_state_get(fd, slot_id, ctx);
                break;
            case HYDRA_OP_STATE_PUT:
                SRV_DBG("hydra rpc: STATE_PUT  slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_state_put(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_STATE_META:
                SRV_DBG("hydra rpc: STATE_META slot=%d trace=%s\n", slot_id, trace_id.c_str());
                hydra_handle_state_meta(fd, slot_id, ctx);
                break;
            case HYDRA_OP_CONFIGURE:
                SRV_DBG("hydra rpc: CONFIGURE slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_configure(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_INFO:
                SRV_DBG("hydra rpc: INFO slot=%d trace=%s\n", slot_id, trace_id.c_str());
                hydra_handle_info(fd, slot_id, ctx);
                break;
            case HYDRA_OP_PREFILL:
                SRV_DBG("hydra rpc: PREFILL slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_prefill(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_DECODE:
                SRV_DBG("hydra rpc: DECODE slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_decode(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_SET_EXPERT_MODE:
                SRV_DBG("hydra rpc: SET_EXPERT_MODE slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_set_expert_mode(fd, slot_id, payload_len, ctx);
                break;
            case HYDRA_OP_SWAP_QUANT:
                SRV_DBG("hydra rpc: SWAP_QUANT slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_swap_quant(fd, slot_id, payload_len, ctx);
                break;
            // M-Perf.9 (#289) / issue #287: PIPELINE_ATTACH (0x46) is the
            // two-engine "work together" attach. Stubbed: full impl in #287.
            case HYDRA_OP_PIPELINE_ATTACH:
                SRV_DBG("hydra rpc: PIPELINE_ATTACH slot=%d payload=%" PRIu64 " trace=%s\n",
                        slot_id, payload_len, trace_id.c_str());
                hydra_handle_pipeline_attach(fd, slot_id, payload_len, ctx);
                break;
            default:
                SRV_WRN("hydra rpc: unknown op 0x%02x — ignoring\n", (unsigned)op);
                hydra_write_res(fd, HYDRA_STATUS_ERROR, 0, 0);
        }
    }
    ::close(fd);
}

// ── Unified RPC server implementation ───────────────────────────────────────
//
// `#36` Phase 1: the merged server lives in `tools/llama-engine/hydra_rpc/`
// (fork-isolated). `server_context::start_rpc_server` is a thin adapter that
// builds the settings and delegates to `hydra_rpc::start()`. The Hydra
// protocol entry `hydra_handle_connection` is reached through the
// `hydra_rpc_bridge` trampoline (defined below) — the bridge takes a
// `void*` so the new module can stay decoupled from this file's includes.

#include "../llama-engine/hydra_rpc/hydra_rpc.h"

void server_context::start_rpc_server(int port,
                                       std::vector<ggml_backend *> backends) {
    if (port <= 0) return;

    // Hydra #43: MUST outlive this function. `hydra_rpc::start()` below
    // stores `&ctx` as a raw pointer inside `hydra_rpc::state()`, a
    // process-lifetime singleton that every subsequent RPC connection reads
    // (from a bounded-thread-pool worker thread) to recover queue_tasks /
    // queue_results. An automatic-storage `ctx` here would dangle the
    // instant this function returns — a stack-use-after-return that "works"
    // until the freed stack slot gets reused, then silently corrupts the
    // RPC response path. `start_rpc_server` only ever runs once per process
    // (hydra_rpc::start() itself guards double-start), so `static` gives it
    // exactly the lifetime the singleton needs.
    static hydra_rpc_ctx ctx{};
    if (impl) {
        ctx.queue_tasks   = &impl->queue_tasks;
        ctx.queue_results = &impl->queue_results;
    }

    hydra_rpc::settings s;
    s.port       = port;
    s.backends   = std::move(backends);
    s.hydra_ctx  = (ctx.queue_tasks && ctx.queue_results) ? &ctx : nullptr;
    s.pool_size  = 2;
    s.max_queue  = 64;
    s.host       = "0.0.0.0";

    if (!hydra_rpc::start(s)) {
        SRV_ERR("hydra rpc: start() failed on port %d\n", port);
        return;
    }

    if (s.hydra_ctx) {
        SRV_INF("hydra rpc: unified server on 0.0.0.0:%d (ggml-RPC + Hydra protocol)\n", port);
    } else {
        SRV_INF("hydra rpc: unified server on 0.0.0.0:%d (ggml-RPC only)\n", port);
    }
}

// `hydra_rpc_bridge` — extern "C" trampoline. `hydra_rpc.cpp` calls this
// when the first byte on a new connection is not `RPC_CMD_HELLO`. It
// re-enters the C++ entry point with the typed `hydra_rpc_ctx &`.
//
// Forward-declared with the matching signature so the new
// `tools/llama-engine/hydra_rpc/hydra_rpc.cpp` module can take its
// address without including this heavy header.
extern "C" void hydra_rpc_bridge(int fd, const void * ctx);
extern "C" void hydra_rpc_bridge(int fd, const void * ctx) {
    hydra_handle_connection(fd, *static_cast<const hydra_rpc_ctx *>(ctx));
}

#else
// Windows: RPC server not implemented — target hardware is Linux-only for M0.
void server_context::start_rpc_server(int port, std::vector<ggml_backend *>) {
    if (port > 0) {
        SRV_WRN("hydra rpc: not supported on Windows (port %d ignored)\n", port);
    }
    GGML_UNUSED(port);
}
#endif // !_WIN32

// --- WS3.5 moved helper methods ---

    bool server_context_impl::apply_t3_rebuild() {
        bool is_first_load = !ctx_tgt;

        // Track the last override_tensor string that was actually
        // applied so we can detect "nothing changed" on subsequent
        // calls and skip the expensive unload+reload cycle.
        static std::string old_override_applied;

        common_params old_params = params_base;
        common_params swapped_params = params_base;

        // Read the staged T3 statics and apply them to swapped_params.
        if (llama_hydra_get_pending_n_gpu_layers() >= 0) {
            swapped_params.n_gpu_layers = llama_hydra_get_pending_n_gpu_layers();
        }
        // n_cpu_moe is informational only — the actual MoE expert
        // offload is done via override_tensor (parsed below into
        // tensor_buft_overrides). The standard common_params struct
        // has no n_cpu_moe field; we just log the staged value for
        // operator visibility.
        if (llama_hydra_get_pending_n_cpu_moe() >= 0) {
            SRV_INF("hydra: T3 rebuild: staged n_cpu_moe=%d (informational; expert routing via override_tensor)\n",
                    llama_hydra_get_pending_n_cpu_moe());
        }
        const char * path = llama_hydra_get_pending_model_path();
        if (path && *path) {
            swapped_params.model.path = path;
        }
        const char * mode = llama_hydra_get_pending_split_mode();
        if (mode && *mode) {
            std::string m(mode);
            if (m == "none")      swapped_params.split_mode = LLAMA_SPLIT_MODE_NONE;
            else if (m == "layer") swapped_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            else if (m == "row")   swapped_params.split_mode = LLAMA_SPLIT_MODE_ROW;
            else SRV_WRN("hydra: T3 split_mode='%s' unknown; keeping current\n", m.c_str());
        }
        const size_t n_split = llama_hydra_get_pending_tensor_split_count();
        if (n_split > 0) {
            const float * split = llama_hydra_get_pending_tensor_split();
            // common_params::tensor_split is a fixed-size array.
            const size_t cap = sizeof(swapped_params.tensor_split) /
                                sizeof(swapped_params.tensor_split[0]);
            const size_t n = n_split < cap ? n_split : cap;
            for (size_t i = 0; i < n; i++) {
                swapped_params.tensor_split[i] = split[i];
            }
            // Zero the rest so the engine doesn't see stale values.
            for (size_t i = n; i < cap; i++) {
                swapped_params.tensor_split[i] = 0.0f;
            }
        }
        const char * override = llama_hydra_get_pending_override_tensor();
        if (override && *override) {
            // Wire-shape: comma-separated "pattern=buft" pairs (e.g.
            // "blk.*.ffn_*_exps.weight=CPU"). The C++ side stores
            // these as a vector<llama_model_tensor_buft_override>.
            // Buft names are looked up
            // via ggml_backend_dev_buffer_type() + ggml_backend_buft_name()
            // (mirrors common/arg.cpp:parse_tensor_buffer_overrides).
            ggml_backend_load_all();
            std::map<std::string, ggml_backend_buffer_type_t> buft_list;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                auto * buft = ggml_backend_dev_buffer_type(dev);
                if (buft) {
                    buft_list[std::string(ggml_backend_buft_name(buft))] = buft;
                }
            }
            // CPU is the common case (MoE expert routing) — also lookup
            // explicitly since some backends may not register the CPU buft.
            buft_list["CPU"] = ggml_backend_cpu_buffer_type();

            // Keep pattern strings alive for the lifetime of the
            // process — entry.pattern is a const char* that must not
            // dangle.  Matches the safe pattern in common/arg.cpp.
            static std::list<std::string> buft_override_patterns;

            std::vector<llama_model_tensor_buft_override> staged;

            const std::string ovr(override);
            size_t start = 0;
            while (start < ovr.size()) {
                size_t comma = ovr.find(',', start);
                std::string part = ovr.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                size_t eq = part.find('=');
                if (eq != std::string::npos) {
                    std::string pattern = part.substr(0, eq);
                    std::string buft_name = part.substr(eq + 1);
                    auto it = buft_list.find(buft_name);
                    if (it != buft_list.end()) {
                        buft_override_patterns.push_back(pattern);
                        llama_model_tensor_buft_override entry;
                        entry.pattern = buft_override_patterns.back().c_str();
                        entry.buft = it->second;
                        staged.push_back(entry);
                    } else {
                        SRV_WRN("%s", "hydra: T3 rebuild: override_tensor buft name not in registered list; skipping pattern\n");
                    }
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }

            // Install the staged patterns *in place of* the base ones instead of
            // appending to them.
            //
            // common_params_parse_ex() (common/arg.cpp) unconditionally pads this
            // vector out to llama_max_tensor_buft_overrides() entries of
            // {nullptr, nullptr}, so by the time we get here the real CLI overrides
            // sit at the head and the rest is terminator padding. push_back() would
            // land *behind* that padding, which breaks twice over:
            //   1. common_model_params_to_llama() asserts that back().pattern is
            //      nullptr, so the engine aborts before the model loads;
            //   2. even without that assert, llama_model_loader stops scanning at
            //      the first nullptr pattern, so appended entries are never read —
            //      the override would be silently dropped and the MoE experts would
            //      land on the GPU.
            // Replacing also matches the sibling fields handled above: model.path,
            // split_mode, n_gpu_layers and tensor_split are all overwritten by the
            // staged T3 config rather than merged into it.
            const size_t ntbo = llama_max_tensor_buft_overrides();
            if (staged.empty()) {
                // Nothing resolved (every buft name was unknown). Wiping the base
                // overrides here would silently change how the model is placed, so
                // keep them and make the no-op explicit.
                SRV_WRN("%s", "hydra: T3 rebuild: staged override_tensor resolved to no usable patterns; keeping base overrides\n");
            } else {
                if (staged.size() + 1 > ntbo) {
                    SRV_WRN("hydra: T3 rebuild: %zu override_tensor patterns exceed the %zu-entry limit; keeping the first %zu\n",
                            staged.size(), ntbo, ntbo - 1);
                    staged.resize(ntbo - 1);
                }
                // assign() re-establishes the full terminator padding, so everything
                // from staged.size() onward is {nullptr, nullptr}.
                swapped_params.tensor_buft_overrides.assign(ntbo, llama_model_tensor_buft_override{ nullptr, nullptr });
                for (size_t i = 0; i < staged.size(); ++i) {
                    swapped_params.tensor_buft_overrides[i] = staged[i];
                }
            }
        }

        // Early-exit: if the model and all T3-relevant params are
        // identical to what is already loaded, skip the expensive
        // unload+reload cycle.  Without this, every COMPLETION
        // request that carries hydra_config triggers a full model
        // swap even when nothing changed (the coordinator sends the
        // same config on every decode request).
        if (!is_first_load) {
            const char * cur_override = llama_hydra_get_pending_override_tensor();
            bool params_unchanged =
                swapped_params.model.path == old_params.model.path &&
                swapped_params.n_gpu_layers == old_params.n_gpu_layers &&
                swapped_params.split_mode == old_params.split_mode &&
                ((cur_override == nullptr && old_override_applied.empty()) ||
                 (cur_override && old_override_applied == cur_override));
            if (params_unchanged) {
                // T3 overrides (override_tensor, split_mode) were staged by
                // the COMPLETION hydra_config path. But the model reload is
                // being skipped. Clear the staged override so the next decode
                // uses the current tensor placement (not the staged override).
                llama_hydra_set_override_tensor(ctx_tgt, nullptr);
                SRV_INF("%s", "hydra: T3 rebuild: model and params unchanged — skipping reload, cleared staged overrides\n");
                return true;
            }
        }

        // COMBINED-mode teardown BEFORE the model reload — see
        // hydra_teardown_combined_before_reload() above.
        const bool was_combined = hydra_combined_head_attached || hydra_combined_static;
        if (!is_first_load && was_combined) {
            hydra_teardown_combined_before_reload();
        }

        // Register any new RPC peer devices before load_model() so the
        // peer's device exists in the global ggml backend registry when
        // common_init_from_params() tries to place tensors per
        // tensor_split/split_mode. Only genuinely new endpoints are
        // registered (hydra_register_rpc_servers tracks already-registered
        // endpoints to avoid unsafe repeated registration).
        if (!g_pending_rpc_servers.empty()) {
            json rpc_arr = json::array();
            for (const auto & s : g_pending_rpc_servers) {
                rpc_arr.push_back(s);
            }
            hydra_register_rpc_servers(rpc_arr);
            g_pending_rpc_servers.clear();
        }

        // Full model reload. load_model() handles the unload of the
        // current model, the load of the new model, the new context
        // creation, the MTP/draft paths, and the slot rebuild.
        // NOTE: load_model() does `params_base = params` internally
        // (line 844), so after a successful load params_base reflects
        // swapped_params — no explicit reassignment needed by us.
        //
        // #507: Skip the fit_params probe during T3 rebuild. The probe
        // does a full model-structure load with no_alloc=true to measure
        // GPU memory — expensive (~45-90s) and unnecessary here because:
        // (a) we just freed VRAM by destroying the old model, (b) the new
        // model's requirements are known (same or smaller), (c) a controlled
        // inference server has predictable VRAM. Disabling saves ~1 min.
        swapped_params.fit_params = false;
        if (!load_model(swapped_params)) {
            if (is_first_load) {
                SRV_WRN("%s", "hydra: T3 first load failed — engine stays empty\n");
                return false;
            }
            SRV_ERR("hydra: T3 reload to '%s' failed (load_model returned false); "
                    "rolling back to old model\n",
                    swapped_params.model.path.c_str());
            if (!load_model(old_params)) {
                SRV_ERR("%s", "hydra: T3 rollback also failed — engine in unrecoverable state\n");
                GGML_ABORT("hydra: T3 rollback failed (cannot reload old model). "
                           "Engine exiting to prevent serving with corrupted state.");
            }
            SRV_INF("hydra: T3 rollback succeeded — restored old model '%s'\n",
                    old_params.model.path.c_str());
            return false;
        }

        // Record the override_tensor that was just applied so the
        // next call can skip the reload if nothing changed.
        {
            const char * cur = llama_hydra_get_pending_override_tensor();
            old_override_applied = cur ? cur : "";
        }

        // T3 reload confirmed. Log model identity for traceability.
        SRV_INF("hydra: T3 reload confirmed model_alias='%s' tokenizer='%s' model_name='%s' quant='%s' caps=0x%x model_path='%s'\n",
                swapped_params.model_alias.empty() ? "?" : swapped_params.model_alias.begin()->c_str(),
                model_tgt ? llama_model_get_tokenizer_model(model_tgt) : "",
                model_tgt ? llama_model_get_display_name(model_tgt) : "",
                model_tgt ? llama_model_get_quant_label(model_tgt) : "",
                model_tgt ? llama_model_get_capabilities_bitfield(model_tgt) : 0,
                swapped_params.model.path.c_str());

        // COMBINED-mode reattach AFTER the model reload — see
        // hydra_reattach_combined_after_reload() above.
        if (was_combined) {
            hydra_reattach_combined_after_reload();
        }

        SRV_INF("hydra: T3 rebuild applied (model='%s', split_mode=%d, n_gpu_layers=%d, slots=%zu)\n",
                params_base.model.path.c_str(), (int) params_base.split_mode,
                params_base.n_gpu_layers, slots.size());
        return true;
    }

    void server_context_impl::hydra_repad_tensor_buft_overrides(common_params & p, const char * ctx_label) {
        const size_t ntbo = llama_max_tensor_buft_overrides();
        if (p.tensor_buft_overrides.size() + 1 > ntbo) {
            SRV_WRN("hydra: %s: %zu tensor_buft_overrides exceed the %zu-entry limit; keeping the first %zu\n",
                    ctx_label, p.tensor_buft_overrides.size(), ntbo, ntbo - 1);
            p.tensor_buft_overrides.resize(ntbo - 1);
        }
        p.tensor_buft_overrides.resize(ntbo, llama_model_tensor_buft_override{ nullptr, nullptr });
    }

    void server_context_impl::hydra_reattach_combined_after_reload() {
        SRV_INF("%s", "hydra: re-attaching COMBINED on new model\n");
        if (hydra_combined_static) {
            llama_hydra_set_expert_mode(ctx_tgt, 1);
        } else if (!hydra_peer.empty() && !hydra_combined_pattern.empty()) {
            if (llama_hydra_peer_reachable(hydra_peer.c_str())) {
                ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
                if (rpc_reg) {
                    using add_server_fn_t = ggml_backend_reg_t (*)(const char *);
                    auto add_server_fn = (add_server_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
                    ggml_backend_reg_t peer_reg = add_server_fn ? add_server_fn(hydra_peer.c_str()) : nullptr;
                    ggml_backend_dev_t  peer_dev = (peer_reg && ggml_backend_reg_dev_count(peer_reg) > 0) ? ggml_backend_reg_dev_get(peer_reg, 0) : nullptr;
                    if (peer_dev) {
                        int32_t n_bound = llama_hydra_rebind_combined_experts(
                                ctx_tgt, hydra_peer.c_str(), peer_dev, hydra_combined_pattern.c_str());
                        if (n_bound > 0) {
                            hydra_combined_head_attached = true;
                            llama_hydra_set_expert_mode(ctx_tgt, 1);
                            SRV_INF("hydra: COMBINED re-attached on peer %s (%d layers bound)\n",
                                    hydra_peer.c_str(), n_bound);
                        } else {
                            SRV_WRN("hydra: rebind returned %d; staying solo\n", n_bound);
                        }
                    } else {
                        SRV_WRN("hydra: peer %s has no device; staying solo\n", hydra_peer.c_str());
                    }
                } else {
                    SRV_WRN("%s\n", "hydra: RPC backend not available; staying solo");
                }
            } else {
                SRV_WRN("hydra: peer %s unreachable; staying solo\n", hydra_peer.c_str());
            }
        }
    }

    void server_context_impl::hydra_teardown_combined_before_reload() {
        SRV_INF("hydra: tearing down COMBINED before model reload (was head_attached=%d, static=%d)\n",
                (int) hydra_combined_head_attached, (int) hydra_combined_static);
        llama_hydra_set_expert_mode(ctx_tgt, 0);
        if (!hydra_current_peer.empty()) {
            ctx_tgt->hydra_remove_combined_rpc_backend(hydra_current_peer.c_str());
        }
        llama_hydra_clear_combined_bindings(ctx_tgt, hydra_peer.c_str());
        hydra_combined_head_attached = false;
    }

    void server_context_impl::hydra_register_rpc_servers(const json & servers_arr) {
        static std::set<std::string> registered;

        if (!servers_arr.is_array() || servers_arr.empty()) {
            return;
        }

        ggml_backend_load_all();
        ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
        if (!rpc_reg) {
            SRV_WRN("%s", "hydra: rpc_servers: RPC backend not available\n");
            return;
        }

        typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
        auto add_server_fn = (ggml_backend_rpc_add_server_t)
            ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
        if (!add_server_fn) {
            SRV_WRN("%s", "hydra: rpc_servers: ggml_backend_rpc_add_server not found\n");
            return;
        }

        for (const auto & v : servers_arr) {
            if (!v.is_string()) continue;
            const std::string endpoint = v.get<std::string>();
            if (endpoint.empty()) continue;
            if (registered.count(endpoint)) {
                SRV_DBG("hydra: rpc_servers: endpoint '%s' already registered, skipping\n",
                        endpoint.c_str());
                continue;
            }
            ggml_backend_reg_t reg = add_server_fn(endpoint.c_str());
            if (reg) {
                ggml_backend_register(reg);
                registered.insert(endpoint);
                SRV_INF("hydra: rpc_servers: registered endpoint '%s'\n", endpoint.c_str());
            } else {
                SRV_WRN("hydra: rpc_servers: failed to register endpoint '%s'\n",
                        endpoint.c_str());
            }
        }
    }

    bool server_context_impl::apply_t2_rebuild(const std::string & pending_json) {
        if (!ctx_tgt || !model_tgt) return false;

        json cfg;
        try {
            cfg = json::parse(pending_json);
        } catch (const std::exception & e) {
            SRV_WRN("hydra: T2 apply: invalid JSON in pending_config: %s\n", e.what());
            return false;
        }

        // Snapshot the old params for rollback. params_base is the
        // canonical "what's in effect" state; restoring it plus a
        // recreate-cycle is the rollback path.
        common_params old_params = params_base;

        // Update params_base with the T2 keys. Each is optional;
        // absence means "leave unchanged".
        if (cfg.contains("n_ctx") && cfg["n_ctx"].is_number_integer()) {
            const int32_t n_ctx = cfg["n_ctx"].get<int32_t>();
            // Clamp to the model's training ctx. The wire spec does
            // not require a reject-on-too-large (the engine's own
            // check below does that); we clamp and report.
            const int32_t max_ctx = (int32_t) llama_model_n_ctx_train(model_tgt);
            if (n_ctx > max_ctx) {
                SRV_WRN("hydra: T2 n_ctx=%d exceeds model_n_ctx_train=%d; clamping\n",
                        n_ctx, max_ctx);
                params_base.n_ctx = max_ctx;
            } else {
                params_base.n_ctx = n_ctx;
            }
        }
        if (cfg.contains("cache_type_k") && cfg["cache_type_k"].is_string()) {
            const std::string & s = cfg["cache_type_k"].get_ref<const std::string &>();
            ggml_type t = hydra_parse_cache_type(s);
            if (t == GGML_TYPE_COUNT) {
                SRV_WRN("hydra: T2 cache_type_k='%s' unparseable; ignoring\n", s.c_str());
            } else {
                params_base.cache_type_k = t;
            }
        }
        if (cfg.contains("cache_type_v") && cfg["cache_type_v"].is_string()) {
            const std::string & s = cfg["cache_type_v"].get_ref<const std::string &>();
            ggml_type t = hydra_parse_cache_type(s);
            if (t == GGML_TYPE_COUNT) {
                SRV_WRN("hydra: T2 cache_type_v='%s' unparseable; ignoring\n", s.c_str());
            } else {
                params_base.cache_type_v = t;
            }
        }
        if (cfg.contains("rope_freq_base") && cfg["rope_freq_base"].is_number()) {
            params_base.rope_freq_base = cfg["rope_freq_base"].get<float>();
        }
        if (cfg.contains("rope_freq_scale") && cfg["rope_freq_scale"].is_number()) {
            params_base.rope_freq_scale = cfg["rope_freq_scale"].get<float>();
        }
        if (cfg.contains("yarn_ext_factor") && cfg["yarn_ext_factor"].is_number()) {
            params_base.yarn_ext_factor = cfg["yarn_ext_factor"].get<float>();
        }
        if (cfg.contains("yarn_attn_factor") && cfg["yarn_attn_factor"].is_number()) {
            params_base.yarn_attn_factor = cfg["yarn_attn_factor"].get<float>();
        }
        if (cfg.contains("yarn_beta_fast") && cfg["yarn_beta_fast"].is_number()) {
            params_base.yarn_beta_fast = cfg["yarn_beta_fast"].get<float>();
        }
        if (cfg.contains("yarn_beta_slow") && cfg["yarn_beta_slow"].is_number()) {
            params_base.yarn_beta_slow = cfg["yarn_beta_slow"].get<float>();
        }
        if (cfg.contains("yarn_orig_ctx") && cfg["yarn_orig_ctx"].is_number_integer()) {
            params_base.yarn_orig_ctx = cfg["yarn_orig_ctx"].get<int32_t>();
        }

        // Free the live context. KV cache is destroyed; this is the
        // T2 cost. The model is kept (T2 is context-only).
        llama_free(ctx_tgt);
        if (ctx_dft) {
            llama_free(ctx_dft.get());
            ctx_dft.reset();
        }

        // Build new cparams from the updated params_base. This is
        // the same call site load_model() uses internally.
        auto cparams = common_context_params_to_llama(params_base);

        // Recreate the context with the new cparams.
        ctx_tgt = llama_new_context_with_model(model_tgt, cparams);
        if (!ctx_tgt) {
            // Rollback: rebuild with the old params_base. The old
            // params must work (we just freed and recreated the
            // context with them). If they don't, the engine is in
            // a bad state — abort.
            SRV_WRN("hydra: T2 rebuild failed with n_ctx=%d cache_type=%d/%d; "
                    "rolling back to old params\n",
                    params_base.n_ctx, (int) params_base.cache_type_k,
                    (int) params_base.cache_type_v);
            params_base = old_params;
            auto cparams_old = common_context_params_to_llama(params_base);
            ctx_tgt = llama_new_context_with_model(model_tgt, cparams_old);
            if (!ctx_tgt) {
                GGML_ABORT("hydra: T2 rollback failed (cannot rebuild context with old params). "
                           "Engine exiting to prevent serving with corrupted state.");
            }
            return false;
        }

        // Re-init per-slot samplers. The old samplers were bound to
        // the now-freed context; common_sampler_init() on the new
        // model picks up the (possibly changed) sampling config.
        for (auto & slot : slots) {
            slot.smpl.reset(common_sampler_init(model_tgt, params_base.sampling));
        }

        n_ctx = llama_n_ctx(ctx_tgt);
        SRV_INF("hydra: T2 rebuild applied (n_ctx=%d, cache=%d/%d, slots=%zu)\n",
                n_ctx, (int) params_base.cache_type_k,
                (int) params_base.cache_type_v, slots.size());
        return true;
    }

    ggml_type server_context_impl::hydra_parse_cache_type(const std::string & s) {
        if (s.empty()) return GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            ggml_type t = (ggml_type) i;
            if (strcmp(ggml_type_name(t), s.c_str()) == 0) return t;
        }
        return GGML_TYPE_COUNT;
    }

    bool server_context_impl::apply_pending_hydra_config() {
        const bool is_first_load = !ctx_tgt;
        if (is_first_load) {
            if (!first_load_pending) {
                return false;
            }
            // Don't check hydra_has_pending_config — ctx_tgt doesn't exist yet.
            // The T3 statics were staged by hydra_apply_t3_mutators() in the
            // CONFIGURE handler. Set a default tier for the rebuild path.
        } else if (!ctx_tgt->hydra_has_pending_config()) {
            return false;
        }

        // 1. Drain timeout — skipped for first load (no ctx_tgt timestamp).
        std::string tier;
        std::string pending_json;

        if (is_first_load) {
            tier = "T3";
            // pending_json stays empty — T3 statics are staged in global
            // overrides, not in pending_config (ctx_tgt doesn't exist yet).
        } else {
            constexpr time_t k_drain_timeout_default = 300;
            time_t now = std::time(nullptr);
            time_t elapsed = now - ctx_tgt->hydra_get_pending_config_set_at();
            int env_timeout = 0;
            if (const char * e = getenv("HYDRA_COORD_PROFILE_SWITCH_DRAIN_TIMEOUT")) {
                env_timeout = atoi(e);
            }
            time_t drain_timeout = env_timeout > 0 ? env_timeout : k_drain_timeout_default;
            if (elapsed > drain_timeout) {
                SRV_WRN("hydra: pending config drain timeout (elapsed=%lld, limit=%lld) — discarding, "
                        "tier='%s' payload_size=%zu\n",
                        (long long) elapsed, (long long) drain_timeout,
                        ctx_tgt->hydra_get_pending_config_tier().c_str(),
                        ctx_tgt->hydra_get_pending_config().size());
                ctx_tgt->hydra_clear_pending_config();
                llama_hydra_clear_pending_t3();
                return false;
            }

            tier = ctx_tgt->hydra_get_pending_config_tier();
            pending_json = ctx_tgt->hydra_get_pending_config();
            SRV_INF("hydra: applying pending config (tier='%s', age=%llds, payload_size=%zu)\n",
                    tier.c_str(), (long long) elapsed, pending_json.size());
        }

        bool ok = true;

        // 2. T2 work: free + rebuild context with the new cparams.
        //    Skipped when tier is T3 (T3's load_model() handles both).
        if (tier == "T2") {
            if (!apply_t2_rebuild(pending_json)) {
                SRV_ERR("%s", "hydra: T2 rebuild failed; engine continues with old context\n");
                ok = false;
            }
        }

        // 3. T3 work: full model reload with the staged T3 statics.
        //    load_model() handles the unload+reload cycle. COMBINED-mode
        //    expert bindings are torn down before the reload and re-
        //    attached after, in the same pattern as SET_EXPERT_MODE.
        if (tier == "T3") {
            if (!apply_t3_rebuild()) {
                SRV_ERR("%s", "hydra: T3 rebuild failed; engine continues with old model\n");
                ok = false;
            } else {
                // P1-6: T3 model changed — the cached server_context_meta
                // (model_path, split_mode, tensor_split, chat_params, …)
                // is now stale.  Refresh it on the task-queue thread
                // (safe — runs during the drain window when no slots are
                // processing and no new requests are being dispatched).
                if (routes_ptr) {
                    routes_ptr->refresh_meta();
                }
                // P0-1 (#49): after deferred first-load, apply staged capabilities
                // so ENGINE_INFO(0x41) and COMBINED-mode logic work correctly.
                if (is_first_load) {
                    hydra_rpc_backend_active = bootstrap_rpc_active;
                    hydra_peer               = bootstrap_peer;
                    hydra_peer_reachable     = bootstrap_peer_reachable;
                    hydra_combined_pattern   = bootstrap_pattern;
                    hydra_split_mode         = bootstrap_split_mode;
                    if (bootstrap_combined_static) {
                        hydra_combined_static = true;
                        SRV_INF("%s", "P0-1: deferred first-load — combined_static mode activated\n");
                    }
                    // Register local tensors and enable shared-backend compute
                    // lock so the model can serve inbound RPC requests.
                    if (model_tgt && ctx_tgt) {
                        llama_hydra_register_local_tensors_for_rpc(ctx_tgt);
                        llama_hydra_enable_shared_backend_compute_lock();
                    }
                    // Update the RPC server's compute backends now that the
                    // model is loaded. The RPC server was started with empty
                    // backends (head-bootstrap mode); now populate it.
                    if (ctx_tgt) {
                        std::vector<ggml_backend_t> backends(8);
                        size_t n = llama_hydra_get_compute_backends(ctx_tgt, backends.data(), backends.size());
                        if (n > backends.size()) {
                            backends.resize(n);
                            n = llama_hydra_get_compute_backends(ctx_tgt, backends.data(), backends.size());
                        }
                        backends.resize(n);
                        hydra_rpc::update_backends(backends);
                        SRV_INF("P0-1: updated RPC backends to %zu compute device(s)\n", backends.size());
                    }
                    SRV_INF("%s", "hydra-engine ready — model loaded via CONFIGURE T3\n");
                }
            }
        }

        // 4. Clear the staged state regardless of success. On failure
        //    the rollback in apply_t{2,3}_rebuild has restored the
        //    previous state; clearing the staged state prevents the
        //    next slot-free moment from re-attempting the same rebuild.
        if (is_first_load) {
            first_load_pending = false;
        } else {
            ctx_tgt->hydra_clear_pending_config();
        }
        llama_hydra_clear_pending_t3();
        return ok;
    }

    void server_context_impl::hydra_apply_t3_mutators(llama_context * ctx, const json & cfg, std::vector<std::string> & deferred_keys) {
        if (cfg.contains("n_gpu_layers") && cfg["n_gpu_layers"].is_number_integer()) {
            llama_hydra_set_pending_n_gpu_layers(cfg["n_gpu_layers"].get<int32_t>());
            deferred_keys.push_back("n_gpu_layers");
        }
        if (cfg.contains("n_cpu_moe") && cfg["n_cpu_moe"].is_number_integer()) {
            llama_hydra_set_pending_n_cpu_moe(cfg["n_cpu_moe"].get<int32_t>());
            deferred_keys.push_back("n_cpu_moe");
        }
        if (cfg.contains("override_tensor") && cfg["override_tensor"].is_string()) {
            llama_hydra_set_override_tensor(ctx, cfg["override_tensor"].get<std::string>().c_str());
            deferred_keys.push_back("override_tensor");
        }
        if (cfg.contains("split_mode") && cfg["split_mode"].is_string()) {
            std::vector<float> split;
            if (cfg.contains("tensor_split") && cfg["tensor_split"].is_array()) {
                for (const auto & v : cfg["tensor_split"]) {
                    if (v.is_number()) split.push_back(v.get<float>());
                }
            }
            llama_hydra_set_split_mode(ctx, cfg["split_mode"].get<std::string>().c_str(),
                                       split.empty() ? nullptr : split.data(), split.size());
            deferred_keys.push_back("split_mode");
            if (!split.empty()) deferred_keys.push_back("tensor_split");
        } else if (cfg.contains("tensor_split") && cfg["tensor_split"].is_array()) {
            // tensor_split without split_mode is meaningless; record it as
            // deferred and let the apply step surface the missing mode.
            deferred_keys.push_back("tensor_split");
        }
        if (cfg.contains("model") && cfg["model"].is_object() &&
            cfg["model"].contains("path") && cfg["model"]["path"].is_string()) {
            llama_hydra_set_pending_model_path(cfg["model"]["path"].get<std::string>().c_str());
            deferred_keys.push_back("model.path");
        } else if (cfg.contains("model") && cfg["model"].is_string()) {
            // legacy shorthand: {"model": "/path/to.gguf"}
            llama_hydra_set_pending_model_path(cfg["model"].get<std::string>().c_str());
            deferred_keys.push_back("model");
        }
        // hydra_config flat key: {"model_path": "/path/to.gguf"}
        if (cfg.contains("model_path") && cfg["model_path"].is_string()) {
            llama_hydra_set_pending_model_path(cfg["model_path"].get<std::string>().c_str());
            deferred_keys.push_back("model_path");
        }
        // hydra_config: {"rpc_servers": ["host1:port1", "host2:port2"]}
        // Stored in a static for apply_t3_rebuild() to consume before
        // load_model(). The actual ggml backend registration happens in
        // hydra_register_rpc_servers() called from apply_t3_rebuild().
        if (cfg.contains("rpc_servers") && cfg["rpc_servers"].is_array()) {
            g_pending_rpc_servers.clear();
            for (const auto & v : cfg["rpc_servers"]) {
                if (v.is_string()) {
                    g_pending_rpc_servers.push_back(v.get<std::string>());
                }
            }
            deferred_keys.push_back("rpc_servers");
        }
    }

    bool server_context_impl::hydra_apply_t1_config(common_params & params, llama_context * ctx, const json & cfg, std::map<std::string, json> & params_applied) {
        // sampling.* — set on the common_params, which the next launch_slot
        // will pick up when re-initializing the slot's common_sampler.
        if (cfg.contains("sampling") && cfg["sampling"].is_object()) {
            const json & s = cfg["sampling"];
            #define COPY_FLOAT(field) \
                if (s.contains(#field) && s[#field].is_number()) { \
                    params.sampling.field = s[#field].get<float>(); \
                    params_applied["sampling." #field] = params.sampling.field; \
                }
            #define COPY_INT(field) \
                if (s.contains(#field) && s[#field].is_number()) { \
                    params.sampling.field = s[#field].get<int32_t>(); \
                    params_applied["sampling." #field] = params.sampling.field; \
                }
            COPY_FLOAT(temp)
            COPY_FLOAT(top_p)
            COPY_FLOAT(min_p)
            COPY_FLOAT(penalty_repeat)
            COPY_INT(top_k)
            COPY_INT(seed)
            #undef COPY_FLOAT
            #undef COPY_INT
        }
        // n_predict
        if (cfg.contains("n_predict") && cfg["n_predict"].is_number_integer()) {
            params.n_predict = cfg["n_predict"].get<int32_t>();
            params_applied["n_predict"] = params.n_predict;
        } else if (cfg.contains("n_predict") && !cfg["n_predict"].is_number_integer()) {
            SRV_WRN("%s", "hydra: CONFIGURE n_predict must be an integer\n");
            return false;
        }
        // n_keep
        if (cfg.contains("n_keep") && cfg["n_keep"].is_number_integer()) {
            params.n_keep = cfg["n_keep"].get<int32_t>();
            params_applied["n_keep"] = params.n_keep;
        } else if (cfg.contains("n_keep") && !cfg["n_keep"].is_number_integer()) {
            SRV_WRN("%s", "hydra: CONFIGURE n_keep must be an integer\n");
            return false;
        }
        // seed (top-level — sets the sampler's seed via common_params::sampling).
        // common_params itself has no top-level seed; common_params_sampling does.
        if (cfg.contains("seed") && cfg["seed"].is_number_unsigned()) {
            params.sampling.seed = cfg["seed"].get<uint32_t>();
            params_applied["seed"] = params.sampling.seed;
        } else if (cfg.contains("seed") && cfg["seed"].is_number_integer()) {
            params.sampling.seed = (uint32_t) cfg["seed"].get<int32_t>();
            params_applied["seed"] = params.sampling.seed;
        } else if (cfg.contains("seed") && !cfg["seed"].is_number()) {
            SRV_WRN("%s", "hydra: CONFIGURE seed must be a number\n");
            return false;
        }
        // antiprompt — full replacement (matches the existing semantics
        // of CLI --reverse-prompt)
        if (cfg.contains("antiprompt") && cfg["antiprompt"].is_array()) {
            std::vector<std::string> new_antiprompt;
            for (const auto & v : cfg["antiprompt"]) {
                if (!v.is_string()) {
                    SRV_WRN("%s", "hydra: CONFIGURE antiprompt entries must be strings\n");
                    return false;
                }
                new_antiprompt.push_back(v.get<std::string>());
            }
            params.antiprompt = std::move(new_antiprompt);
            params_applied["antiprompt"] = params.antiprompt;
        } else if (cfg.contains("antiprompt") && !cfg["antiprompt"].is_array()) {
            SRV_WRN("%s", "hydra: CONFIGURE antiprompt must be an array of strings\n");
            return false;
        }
        // state_chunk_size — apply via the existing llama_hydra API (clamps
        // and echoes the post-clamp value)
        if (cfg.contains("state_chunk_size") && cfg["state_chunk_size"].is_number_unsigned()) {
            const size_t bytes = cfg["state_chunk_size"].get<size_t>();
            if (ctx) {
                llama_hydra_set_state_chunk_size(ctx, bytes);
            }
            const size_t applied = ctx ? llama_hydra_get_state_chunk_size(ctx) : llama_hydra_clamp_state_chunk_size(bytes);
            params_applied["state_chunk_size"] = (uint64_t) applied;
        } else if (cfg.contains("state_chunk_size") && !cfg["state_chunk_size"].is_number_unsigned()) {
            SRV_WRN("%s", "hydra: CONFIGURE state_chunk_size must be a non-negative integer\n");
            return false;
        }
        return true;
    }

    int server_context_impl::hydra_classify_config_key(const std::string & key) {
        // T1: sampling nested keys
        if (key == "sampling.temp"           ||
            key == "sampling.top_p"          ||
            key == "sampling.top_k"          ||
            key == "sampling.min_p"          ||
            key == "sampling.penalty_repeat" ||
            key == "sampling.seed") {
            return 1;
        }
        // T1: top-level fields
        if (key == "n_predict" ||
            key == "n_keep"    ||
            key == "seed"      ||
            key == "antiprompt" ||
            key == "state_chunk_size") {
            return 1;
        }
        // T2: context-level (KV cache / RoPE / ctx)
        if (key == "n_ctx"        ||
            key == "cache_type_k" ||
            key == "cache_type_v" ||
            key.rfind("rope_", 0) == 0) {
            return 2;
        }
        // T3: model-level (offload / placement / model)
        if (key == "n_gpu_layers"    ||
            key == "n_cpu_moe"       ||
            key == "override_tensor" ||
            key == "split_mode"      ||
            key == "tensor_split"    ||
            key == "model_path"      ||  // hydra_config: absolute GGUF path
            key == "rpc_servers"     ||  // hydra_config: RPC peer endpoints to register
            key == "model.path"      ||
            key == "model") {        // legacy alias for { "model": { "path": ... } }
            return 3;
        }
        return 0;  // unknown
    }

    const char * server_context_impl::hydra_tier_label(int tier) {
        switch (tier) {
            case 1: return "T1";
            case 2: return "T2";
            case 3: return "T3";
            default: return "T1";  // 0 (no recognized keys) → degenerate T1
        }
    }
