// SPDX-License-Identifier: MIT
//
// Hydra: bounded thread pool for the merged RPC server (`tools/llama-engine/hydra_rpc/`).
//
// Fork-isolated — lives under `tools/llama-engine/`, NOT `tools/server/`, so that
// the ggml-RPC + Hydra protocol dispatch loop is clearly a Hydra extension and
// survives upstream rebases without merge conflicts.
//
// Replaces the per-conn `std::thread::detach()` that the pre-#30 unified server
// used (see `ddvnguyen/llama.cpp#36` C7 — bounded thread pool is a hard
// constraint, not an optimization).
//
// Design (`#36`):
//   - `bounded_thread_pool<N>` is a template — N is the worker thread count
//     (compile-time constant for size 2 per the design).
//   - The pool owns N worker threads.
//   - `enqueue(f)` BLOCKS if the queue is full (back-pressure).
//   - `try_enqueue(f)` returns `false` if the queue is full (drop on overflow,
//     not the inference path's normal case — only the RPC server's per-conn
//     dispatch uses this).
//   - `stop()` joins all workers; safe to call multiple times.
//   - Workers are not interruptible mid-task — a task that ignores shutdown
//     will run to completion. This matches the upstream ggml-rpc semantics
//     where a graph_compute can take seconds.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace hydra_rpc {

template <std::size_t N>
class bounded_thread_pool {
    static_assert(N > 0, "bounded_thread_pool requires at least one worker");

public:
    using task_type = std::function<void()>;

    bounded_thread_pool(std::size_t max_queue = 64) : max_queue_(max_queue) {
        workers_.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    // Non-copyable, non-movable — the worker threads hold `this`.
    bounded_thread_pool(const bounded_thread_pool &) = delete;
    bounded_thread_pool & operator=(const bounded_thread_pool &) = delete;
    bounded_thread_pool(bounded_thread_pool &&) = delete;
    bounded_thread_pool & operator=(bounded_thread_pool &&) = delete;

    ~bounded_thread_pool() {
        stop();
    }

    // Back-pressure: blocks the caller if the queue is full. Use this on the
    // hot path (RPC server's accept loop) when we want the producer to wait
    // for a slot rather than drop.
    void enqueue(task_type f) {
        if (!f) return;
        std::unique_lock<std::mutex> lk(mu_);
        cv_full_.wait(lk, [this] { return queue_.size() < max_queue_ || stopping_; });
        if (stopping_) return;
        queue_.push(std::move(f));
        lk.unlock();
        cv_empty_.notify_one();
    }

    // Drop on overflow: returns false without blocking. Use this for
    // untrusted / DoS-prone producers where blocking would amplify the
    // problem. The unified RPC server uses this in the per-conn handler
    // — an attacker flooding connections should see drops, not back-pressure
    // that fills the accept queue.
    bool try_enqueue(task_type f) {
        if (!f) return false;
        std::unique_lock<std::mutex> lk(mu_);
        if (stopping_) return false;
        if (queue_.size() >= max_queue_) {
            return false;
        }
        queue_.push(std::move(f));
        lk.unlock();
        cv_empty_.notify_one();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_empty_.notify_all();
        cv_full_.notify_all();
        for (auto & t : workers_) {
            if (t.joinable()) t.join();
        }
        // Drain any unstarted tasks.
        std::lock_guard<std::mutex> lk(mu_);
        while (!queue_.empty()) queue_.pop();
    }

    std::size_t size() const { return N; }
    std::size_t queue_size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return queue_.size();
    }
    std::size_t max_queue_size() const { return max_queue_; }

private:
    void worker_loop() {
        while (true) {
            task_type task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_empty_.wait(lk, [this] { return !queue_.empty() || stopping_; });
                if (stopping_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop();
            }
            cv_full_.notify_one();
            // Tasks run outside the lock — a slow task does not block
            // siblings from being dequeued. If a task throws, the worker
            // thread is the only thing that catches it (we do not propagate
            // exceptions across threads, matching upstream ggml-rpc's
            // fail-stop semantics in normal ops).
            try {
                task();
            } catch (...) {
                // Swallow — the task is the dispatch path, not the inference
                // path. Failures inside the dispatch are reported back to
                // the client (close fd or RPC error response); they are
                // never fatal to the engine.
            }
        }
    }

    mutable std::mutex mu_;
    std::condition_variable cv_empty_;
    std::condition_variable cv_full_;
    std::queue<task_type> queue_;
    const std::size_t max_queue_;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace hydra_rpc
