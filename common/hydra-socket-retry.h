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

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

// Attempt a non-blocking recv with bounded poll-retry on EAGAIN/EWOULDBLOCK.
//
// Returns:
//   >0  — bytes read (always == n on success, caller loop handles short reads)
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
    const int64_t deadline_ms =
        static_cast<int64_t>(timeout_ms) > 0 ? timeout_ms : 30000;
    // We track elapsed time via poll slices rather than a clock to avoid
    // clock-resolution issues on all platforms; the loop simply counts
    // down `remaining_ms`.
    int remaining_ms = deadline_ms;

    for (;;) {
        if (remaining_ms <= 0) {
            errno = ETIMEDOUT;
            return -1;
        }
        const int wait_ms = std::min(remaining_ms, 1000);
        struct pollfd pfd = { fd, POLLIN, 0 };
        int pr = ::poll(&pfd, 1, wait_ms);
        if (pr < 0) {
            if (errno == EINTR) {
                // EINTR on poll: subtract the slice we waited and retry.
                remaining_ms -= wait_ms;
                continue;
            }
            return -1;  // real poll error (EBADF, EINVAL, …)
        }
        if (pr == 0) {
            // Timeout slice expired — subtract and loop.
            remaining_ms -= wait_ms;
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
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Still EAGAIN — subtract the poll slice and loop.
            remaining_ms -= wait_ms;
            continue;
        }
        if (errno == EINTR) {
            // EINTR on recv: subtract the poll slice and retry (don't
            // count this as a "no data" cycle against the deadline).
            remaining_ms -= wait_ms;
            continue;
        }
        return -1;  // hard error (ECONNRESET, etc.)
    }
}

#endif // !_WIN32
#endif // LLAMA_HYDRA_SOCKET_RETRY_H
