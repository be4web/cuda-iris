extern "C" {
#include "reduction.h"
#include <stdint.h>
}

__global__ void histogram_block_kernel(int img_w, uint8_t *img, int *histo)
{
    __shared__ int tmp[256];

#pragma unroll
    for (int i = 0; i < 8; i++)
        tmp[8 * threadIdx.y + i] = 0;

    __syncthreads();

    const int x = blockIdx.x * 32;
    const int y = blockIdx.y * 32 + threadIdx.y;

    img += y * img_w + x;
    histo += (blockIdx.y * (img_w / 32) + blockIdx.x) * 256;
#pragma unroll
    for (int i = 0; i < 32; i++)
        atomicAdd(tmp + img[i], 1);

    __syncthreads();

#pragma unroll
    for (int i = 0; i < 8; i++)
        histo[8 * threadIdx.y + i] = tmp[8 * threadIdx.y + i];
}

__global__ void histogram_reduce_kernel(int cnt, int *histo)
{
    histo += threadIdx.x * 2 * 256;

#pragma unroll
    for (int i = 0; i < 256; i++)
        histo[i] += histo[256 + i];

    for (int s = 2; threadIdx.x % s == 0 && threadIdx.x * 2 + s < cnt; s <<= 1) {
        __syncthreads();

#pragma unroll
        for (int i = 0; i < 256; i++)
            histo[i] += histo[256 * s + i];
    }
}

extern "C" void cu_histogram(int img_w, int img_h, void *gm_img, void *gm_tmp, int *histo)
{
    dim3 blocks(img_w / 32, img_h / 32);
    dim3 threads(1, 32);

    histogram_block_kernel<<<blocks, threads>>>(img_w, (uint8_t *)gm_img, (int *)gm_tmp);

    int blk_cnt = ((img_w / 32) * (img_h / 32)) / 2;
    histogram_reduce_kernel<<<1, blk_cnt>>>(blk_cnt, (int *)gm_tmp);

    cudaMemcpy(histo, gm_tmp, 256 * sizeof(int), cudaMemcpyDeviceToHost);
}

