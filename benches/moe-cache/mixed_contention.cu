// mixed_contention.cu - decide the pacing lever's premise.
//
// The look-ahead staging DMA competes with the demand gather for one PCIe gen4 x4
// fabric. Every earlier probe measured SOLO rates (CE 6.10, contiguous SM 6.11,
// 1.78 MiB slabs @ 8 blocks 5.91). None measured DMA concurrent with gather-pattern
// reads, so the displacement coefficient is unknown: does a staged byte cost a
// demand byte 1:1, or does the fabric have real concurrent headroom?
//
// A = the gather's own pattern: 8 blocks, each streaming 1.78 MiB slabs from
//     host-mapped memory (matches the measured 5.91 GB/s geometry).
// B = the copy engine: cudaMemcpyAsync H2D of 1.9 MiB chunks on a second stream,
//     sized to the width-1 staging rate (~1/6 of A's volume).
//
// Reports T_A solo, T_B solo, and T_A while B runs concurrently, then the
// displacement coefficient: 0.0 = B rode completely free, 1.0 = each B byte cost
// exactly one A byte.
//
// build: nvcc -O3 -arch=sm_86 -o mixed_contention mixed_contention.cu
// run:   CUDA_VISIBLE_DEVICES=1 ./mixed_contention

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static const size_t SLAB_BYTES   = 1782579;   // 1.78 MiB, the measured geometry
static const int    BLOCKS       = 8;
static const int    PASSES       = 20;
static const size_t COPY_BYTES   = 1920000;   // ~1.9 MiB, a width-1 lane
static const int    COPIES       = 25;        // ~47 MiB, the width-1 staging volume

__global__ void slab_read(const uint4 * __restrict__ src, uint4 * __restrict__ sink,
                          size_t words_per_slab, int passes) {
    const int slab = blockIdx.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int p = 0; p < passes; ++p) {
        const uint4 * base = src + ((size_t) slab + (size_t) p * BLOCKS) * words_per_slab;
        for (size_t i = threadIdx.x; i < words_per_slab; i += blockDim.x) {
            const uint4 v = base[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    // never true in practice; keeps the reads from being optimised away
    if (acc.x == 0xdeadbeefu && acc.y == 0xfeedfaceu) sink[threadIdx.x] = acc;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    const size_t words_per_slab = SLAB_BYTES / sizeof(uint4);
    const size_t slabs = (size_t) BLOCKS * PASSES;
    const size_t host_words = slabs * words_per_slab;
    const size_t host_bytes = host_words * sizeof(uint4);
    const size_t a_bytes = (size_t) BLOCKS * PASSES * SLAB_BYTES;
    const size_t b_bytes = (size_t) COPIES * COPY_BYTES;

    void * h_src = nullptr, * h_copy = nullptr;
    CK(cudaHostAlloc(&h_src, host_bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CK(cudaHostAlloc(&h_copy, COPY_BYTES * 2, cudaHostAllocDefault));
    for (size_t i = 0; i < host_bytes / sizeof(unsigned); ++i) ((unsigned *) h_src)[i] = (unsigned) (i * 2654435761u);
    for (size_t i = 0; i < COPY_BYTES * 2 / sizeof(unsigned); ++i) ((unsigned *) h_copy)[i] = (unsigned) i;

    void * d_src = nullptr, * d_sink = nullptr, * d_copy = nullptr;
    CK(cudaHostGetDevicePointer(&d_src, h_src, 0));
    CK(cudaMalloc(&d_sink, 4096));
    CK(cudaMalloc(&d_copy, COPY_BYTES * 4));

    cudaStream_t s_copy;
    CK(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking));

    // ---- A solo ----
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    slab_read<<<BLOCKS, 256>>>((const uint4 *) d_src, (uint4 *) d_sink, words_per_slab, PASSES);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float t_a_solo = 0.f; CK(cudaEventElapsedTime(&t_a_solo, e0, e1));

    // ---- B solo ----
    CK(cudaEventRecord(e0));
    for (int i = 0; i < COPIES; ++i) CK(cudaMemcpyAsync(d_copy, h_copy, COPY_BYTES, cudaMemcpyHostToDevice, s_copy));
    CK(cudaEventRecord(e1, s_copy));
    CK(cudaEventSynchronize(e1));
    float t_b_solo = 0.f; CK(cudaEventElapsedTime(&t_b_solo, e0, e1));

    // ---- mixed: record A's time while B streams concurrently ----
    CK(cudaEventRecord(e0));
    for (int i = 0; i < COPIES; ++i) CK(cudaMemcpyAsync(d_copy, h_copy, COPY_BYTES, cudaMemcpyHostToDevice, s_copy));
    slab_read<<<BLOCKS, 256>>>((const uint4 *) d_src, (uint4 *) d_sink, words_per_slab, PASSES);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float t_a_mixed = 0.f; CK(cudaEventElapsedTime(&t_a_mixed, e0, e1));
    CK(cudaStreamSynchronize(s_copy));

    const double a_gbs_solo  = a_bytes / (t_a_solo  / 1000.0) / 1e9;
    const double b_gbs_solo  = b_bytes / (t_b_solo  / 1000.0) / 1e9;
    const double a_gbs_mixed = a_bytes / (t_a_mixed / 1000.0) / 1e9;
    const double a_lost_bytes = (t_a_mixed - t_a_solo) / 1000.0 * (a_bytes / (t_a_solo / 1000.0));
    const double displacement = a_lost_bytes / (double) b_bytes;

    printf("A (gather pattern, 8 blocks x 1.78 MiB slab) : %.1f MiB\n", a_bytes / 1048576.0);
    printf("B (copy engine, 1.9 MiB chunks)             : %.1f MiB\n\n", b_bytes / 1048576.0);
    printf("T_A solo   = %8.2f ms   A = %6.2f GB/s\n", t_a_solo,  a_gbs_solo);
    printf("T_B solo   = %8.2f ms   B = %6.2f GB/s\n", t_b_solo,  b_gbs_solo);
    printf("T_A mixed  = %8.2f ms   A = %6.2f GB/s   (B running concurrently)\n", t_a_mixed, a_gbs_mixed);
    printf("\nA slowdown            = %+7.2f ms  (%+.1f%%)\n", t_a_mixed - t_a_solo,
           100.0 * (t_a_mixed - t_a_solo) / t_a_solo);
    printf("B bytes that could be moved in that slowdown at A's solo rate = %.1f MiB\n",
           a_lost_bytes / 1048576.0);
    printf("DISPLACEMENT COEFFICIENT = %.2f  (0 = B rode free, 1 = each B byte cost one A byte)\n",
           displacement);
    printf("\nInterpretation: >= ~1.0 means the fabric is effectively serial and immediate-issue\n"
           "staging displaces demand 1:1, so pacing into the idle gaps converts the whole tax.\n"
           "< ~0.5 means real concurrent headroom exists and pacing's upside is smaller.\n");

    CK(cudaFreeHost(h_src)); CK(cudaFreeHost(h_copy));
    CK(cudaFree(d_sink)); CK(cudaFree(d_copy));
    CK(cudaStreamDestroy(s_copy));
    return 0;
}
