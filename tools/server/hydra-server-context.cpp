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
