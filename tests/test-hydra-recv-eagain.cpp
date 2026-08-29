// hydra#713: EAGAIN/EWOULDBLOCK retry in hydra_recv_with_retry().
//
// Bug: when recv() returns -1/EAGAIN mid-state-stream (non-blocking socket
// under backpressure), the old code treated it identically to EOF -> short
// buffer passed to state_seq_set_data -> "unexpectedly reached end of buffer"
// -> silent corrupt KV pool.
//
// Fix: hydra_recv_with_retry() (common/hydra-socket-retry.h) polls + retries
// on EAGAIN/EWOULDBLOCK with a bounded deadline, drains buffered data on
// POLLHUP before declaring EOF, and retries recv() on EINTR.
//
// This test exercises the REAL function directly (no copy-paste mirror).
// No model or GPU needed -- pure socket test.
#include "hydra-socket-retry.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

static int g_failures = 0;

static void expect(const char * what, bool ok) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

// Helper: read exactly n bytes using hydra_recv_with_retry in a loop.
// Returns true on success, false on error/timeout/EOF.
static bool recv_all(int fd, void * buf, size_t n) {
    char * p = reinterpret_cast<char *>(buf);
    while (n > 0) {
        ssize_t r = hydra_recv_with_retry(fd, p, n, 30000);
        if (r > 0) {
            p += r; n -= (size_t)r;
        } else {
            return false;
        }
    }
    return true;
}

int main() {
    fprintf(stderr, "test-hydra-recv-eagain: running\n");

    // ── Case A: delayed write with EAGAIN retry.  Writer sleeps 200 ms then
    //    sends; reader's first recv() returns EAGAIN, poll() blocks until
    //    POLLIN, then recv() succeeds.
    {
        int sv[2];
        expect("A: socketpair", socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        int flags = fcntl(sv[0], F_GETFL, 0);
        fcntl(sv[0], F_SETFL, flags | O_NONBLOCK);

        const char payload[] = "hello hydra#713";
        std::thread writer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            ssize_t w = ::send(sv[1], payload, sizeof(payload), 0);
            expect("A: send", w == (ssize_t)sizeof(payload));
        });

        char buf[sizeof(payload)] = {};
        bool ok = recv_all(sv[0], buf, sizeof(payload));
        expect("A: recv retries past EAGAIN", ok);
        expect("A: payload matches", ok && memcmp(buf, payload, sizeof(payload)) == 0);

        writer.join();
        ::close(sv[0]);
        ::close(sv[1]);
    }

    // ── Case B: deterministic fast-path (no EAGAIN).  Data is written to the
    //    socket BEFORE the first recv() call, so the reader sees data
    //    immediately -- the common case for small payloads.  This validates
    //    that the fast path (no poll overhead) works correctly.
    {
        int sv[2];
        expect("B: socketpair", socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

        const char payload[] = "immediate-send";
        // Write BEFORE setting non-blocking -- data is in kernel buffer.
        ssize_t w = ::send(sv[1], payload, sizeof(payload), 0);
        expect("B: send", w == (ssize_t)sizeof(payload));

        // Now set non-blocking and read -- should succeed on first recv().
        int flags = fcntl(sv[0], F_GETFL, 0);
        fcntl(sv[0], F_SETFL, flags | O_NONBLOCK);

        char buf[sizeof(payload)] = {};
        bool ok = recv_all(sv[0], buf, sizeof(payload));
        expect("B: immediate data (fast path, no EAGAIN)", ok);
        expect("B: payload matches", ok && memcmp(buf, payload, sizeof(payload)) == 0);

        ::close(sv[0]);
        ::close(sv[1]);
    }

    // ── Case C: EOF (writer closes without sending).  recv must return false,
    //    not hang.
    {
        int sv[2];
        expect("C: socketpair", socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        int flags = fcntl(sv[0], F_GETFL, 0);
        fcntl(sv[0], F_SETFL, flags | O_NONBLOCK);

        ::close(sv[1]); // EOF

        char buf[8] = {};
        bool ok = recv_all(sv[0], buf, sizeof(buf));
        expect("C: EOF returns false (no hang)", !ok);

        ::close(sv[0]);
    }

    // ── Case D: timeout on slow peer.  Custom 100 ms timeout; writer delays
    //    500 ms -- must time out cleanly.
    {
        int sv[2];
        expect("D: socketpair", socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        int flags = fcntl(sv[0], F_GETFL, 0);
        fcntl(sv[0], F_SETFL, flags | O_NONBLOCK);

        std::thread writer([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            ::send(sv[1], "late", 4, 0);
        });

        char buf[8] = {};
        ssize_t r = hydra_recv_with_retry(sv[0], buf, sizeof(buf), 100);
        expect("D: slow peer -> timeout returns -1", r == -1);
        expect("D: errno is ETIMEDOUT", errno == ETIMEDOUT);

        writer.join();
        ::close(sv[0]);
        ::close(sv[1]);
    }

    // ── Case E: POLLHUP with buffered data.  Writer writes a large payload
    //    (64 KB -- larger than typical socket buffer to ensure some kernel
    //    buffering) then immediately close()s.  On Linux, poll() returns
    //    POLLHUP|POLLIN when the peer closes with data still in the buffer.
    //    The reader MUST receive all bytes before seeing EOF -- not fail
    //    with a spurious error.  This is the root-cause bug from the t2
    //    review (discarding buffered data on legitimate STATE_PUT transfers
    //    up to 800 MB).
    {
        int sv[2];
        expect("E: socketpair", socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        int flags = fcntl(sv[0], F_GETFL, 0);
        fcntl(sv[0], F_SETFL, flags | O_NONBLOCK);

        // Build a 64 KB payload.
        const size_t payload_sz = 64 * 1024;
        std::vector<char> payload(payload_sz);
        for (size_t i = 0; i < payload_sz; i++) {
            payload[i] = (char)(i & 0xFF);
        }

        // Writer: send large payload then immediately close.
        std::thread writer([&] {
            ssize_t w = ::send(sv[1], payload.data(), payload_sz, 0);
            // May be partial if kernel buffer is small; loop to be safe.
            size_t sent = (w > 0) ? (size_t)w : 0;
            while (sent < payload_sz) {
                w = ::send(sv[1], payload.data() + sent, payload_sz - sent, 0);
                if (w <= 0) break;
                sent += (size_t)w;
            }
            ::close(sv[1]);
        });

        // Read all bytes then expect EOF.
        std::vector<char> buf(payload_sz);
        bool ok = recv_all(sv[0], buf.data(), payload_sz);
        expect("E: all bytes received before EOF", ok);
        expect("E: payload matches", ok && memcmp(buf.data(), payload.data(), payload_sz) == 0);

        // After draining, the next recv should return 0 (EOF).
        char junk[8];
        ssize_t r = hydra_recv_with_retry(sv[0], junk, sizeof(junk), 1000);
        expect("E: EOF after full drain", r == 0);

        writer.join();
        ::close(sv[0]);
    }

    // ── Case F: POLLHUP set together with POLLIN (the common Linux pattern).
    //    Write a small payload, close the writer, then read.  poll() may
    //    return POLLIN|POLLHUP in a single call.  The reader must get the
    //    data, not discard it.
    {
        int sv[2];
        expect("F: socketpair", socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        int flags = fcntl(sv[0], F_GETFL, 0);
        fcntl(sv[0], F_SETFL, flags | O_NONBLOCK);

        const char payload[] = "pollhup-with-data";
        // Write then close in same thread -- fast, no delay.
        ::send(sv[1], payload, sizeof(payload), 0);
        ::close(sv[1]);

        char buf[sizeof(payload)] = {};
        bool ok = recv_all(sv[0], buf, sizeof(payload));
        expect("F: data received despite POLLHUP", ok);
        expect("F: payload matches", ok && memcmp(buf, payload, sizeof(payload)) == 0);

        ::close(sv[0]);
    }

    if (g_failures == 0) {
        fprintf(stderr, "test-hydra-recv-eagain: all checks passed\n");
    } else {
        fprintf(stderr, "test-hydra-recv-eagain: %d check(s) FAILED\n", g_failures);
    }
    return g_failures;
}
