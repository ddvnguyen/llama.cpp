// hydra#713: shared recv-with-EAGAIN-retry helper for non-blocking sockets.
//
// Both the M2 state-stream path (llama_io_read_socket::refill in
// llama-context.cpp) and the RPC framing path (hydra_recv_all in
// server-context.cpp) need identical EAGAIN/EWOULDBLOCK handling: poll + retry
// with a bounded deadline, drain on POLLHUP/POLLERR before declaring EOF, and
// EINTR retry on every syscall.  A single copy here avoids silent drift
// between the two call sites.
//
// Header-only, POSIX only (`#if !defined(_WIN32)`).
//
// timeout_ms is a per-recv-call budget (not a per-transfer budget):
// each call to hydra_recv_with_retry waits at most timeout_ms for the
// requested n bytes.  The caller's outer loop (hydra_recv_all, refill)
// may invoke this repeatedly for large transfers.

#ifndef LLAMA_HYDRA_SOCKET_RETRY_H
#define LLAMA_HYDRA_SOCKET_RETRY_H

#if !defined(_WIN32)

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

// Attempt a non-blocking recv with bounded poll-retry on EAGAIN/EWOULDBLOCK.
//
// Returns:
//   >0  — bytes read (1..n).  recv may return a short read (r < n), so the
//         caller's loop must keep reading until n bytes are in or EOF/error.
//    0  — clean EOF (peer closed after draining any buffered data)
//   -1  — hard error or timeout; errno is set:
//           ETIMEDOUT  — timeout_ms elapsed with no data
//           ECONNRESET / EPIPE / etc. — peer-level failure
//           EBADF      — bad fd
//
// On POLLHUP or POLLERR the helper performs one final recv to drain any
// data the peer wrote before closing.  Only if that recv also returns 0
// (EOF) or a hard error does the function return.  This prevents
// discarding buffered data when the peer writes its final bytes then
// closes (a common pattern for large STATE_PUT transfers up to 800 MB).
inline ssize_t hydra_recv_with_retry(int fd, void * buf, size_t n, int timeout_ms) {
    char * p = reinterpret_cast<char *>(buf);

    // Fast path: attempt recv immediately — no poll overhead for the common case.
    ssize_t r = ::recv(fd, p, n, 0);
    if (r > 0) {
        return r;
    }
    if (r == 0) {
        return 0;  // clean EOF
    }
    // r < 0 — check errno before entering the retry loop.
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
        return -1;  // hard error (ECONNRESET, EBADF, …)
    }

    // EAGAIN: poll + retry loop with a bounded wall-clock deadline.
    // The deadline is relative to the FIRST EAGAIN, not to the original call,
    // so the caller's per-call budget is respected.
    //
    // Elapsed time is measured with a monotonic clock (steady_clock), not by
    // subtracting the poll slice each iteration.  Slice-subtraction burns the
    // whole budget on events that consumed no wall time (EINTR, a spurious
    // POLLIN that still yields EAGAIN), which would time out a healthy
    // transfer prematurely.  Charging real elapsed time also makes the EINTR
    // handling actually correct rather than merely bounded.
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(static_cast<long long>(timeout_ms > 0 ? timeout_ms : 30000));

    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            errno = ETIMEDOUT;
            return -1;
        }
        const auto rem_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const int wait_ms = static_cast<int>(std::min<long long>(rem_ms, 1000LL));

        struct pollfd pfd = { fd, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, wait_ms);
        if (pr < 0) {
            if (errno == EINTR) {
                // Interrupted before any data: real elapsed time is already
                // charged against the deadline at the top of the loop.
                continue;
            }
            return -1;  // real poll error (EBADF, EINVAL, …)
        }
        if (pr == 0) {
            // Poll slice expired with no data: loop back — the deadline check
            // accounts for the time that actually passed.
            continue;
        }

        // poll returned > 0: at least one event is ready.

        if (pfd.revents & (POLLHUP | POLLERR)) {
            // On Linux, POLLIN is often set together with POLLHUP when the
            // peer wrote final bytes then closed.  We must drain buffered
            // data before declaring EOF — otherwise up to 800 MB of a
            // legitimate STATE_PUT transfer is silently discarded.
            r = ::recv(fd, p, n, 0);
            if (r > 0) {
                return r;
            }
            if (r == 0) {
                return 0;  // true EOF after drain
            }
            // r < 0: recv error after HUP/ERR — propagate.
            return -1;
        }

        // POLLIN ready — attempt the actual recv.
        r = ::recv(fd, p, n, 0);
        if (r > 0) {
            return r;
        }
        if (r == 0) {
            return 0;  // clean EOF
        }
        // r < 0
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            // Still no data (spurious readiness / interrupted): retry within
            // the deadline.  Real elapsed time is charged at the top of the loop.
            continue;
        }
        return -1;  // hard error (ECONNRESET, etc.)
    }
}

#endif // !_WIN32
#endif // LLAMA_HYDRA_SOCKET_RETRY_H
