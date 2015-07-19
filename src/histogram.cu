extern "C" {
#include "histogram.h"
#include <stdint.h>
}

__global__ void histogram_block_kernel(int img_p, uint8_t *img, int *histo)
{
    __shared__ int tmp[256];

#pragma unroll
    for (int i = 0; i < 8; i++)
        tmp[8 * threadIdx.y + i] = 0;

    __syncthreads();

    const int x = blockIdx.x * 32;
    const int y = blockIdx.y * 32 + threadIdx.y;

    img += y * img_p + x;
    histo += (blockIdx.y * (img_p / 32) + blockIdx.x) * 256;

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

extern "C" void cu_histogram(int img_w, int img_h, int img_p, void *gm_img, void *gm_tmp, int *histo)
{
    dim3 blocks(img_w / 32, img_h / 32);
    dim3 threads(1, 32);

    histogram_block_kernel<<<blocks, threads>>>(img_p, (uint8_t *)gm_img, (int *)gm_tmp);

    int blk_cnt = ((img_w / 32) * (img_h / 32)) / 2;
    histogram_reduce_kernel<<<1, blk_cnt>>>(blk_cnt, (int *)gm_tmp);

    cudaMemcpy(histo, gm_tmp, 256 * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void mean_kernel(int img_p, float *img, float *mean)
{
    __shared__ float blk_mean;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        blk_mean = 0.0;

    __syncthreads();

    const int x = blockIdx.x * 512 + threadIdx.x * 16;
    const int y = blockIdx.y * 32 + threadIdx.y;

    img += y * img_p + x;
    mean += blockIdx.y * (img_p / 512) + blockIdx.x;

    float tmp = 0.0;

#pragma unroll
    for (int i = 0; i < 16; i++)
        tmp += img[i];

#pragma unroll
    for (int off = 32 / 2; off > 0; off /= 2)
        tmp += __shfl_down(tmp, off);

    if (threadIdx.x == 0)
        atomicAdd(&blk_mean, tmp);

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
        *mean = blk_mean / (512.0 * 32.0);
}

__global__ void mad_kernel(int img_p, float *img, float *mad, float mean)
{
    __shared__ float blk_mad;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        blk_mad = 0.0;

    __syncthreads();

    const int x = blockIdx.x * 512 + threadIdx.x * 16;
    const int y = blockIdx.y * 32 + threadIdx.y;

    img += y * img_p + x;
    mad += blockIdx.y * (img_p / 512) + blockIdx.x;

    float tmp = 0.0;

#pragma unroll
    for (int i = 0; i < 16; i++) {
        float diff = img[i] - mean;
        tmp += (diff < 0.0) ? -diff : diff;
    }

#pragma unroll
    for (int off = 32 / 2; off > 0; off /= 2)
        tmp += __shfl_down(tmp, off);

    if (threadIdx.x == 0)
        atomicAdd(&blk_mad, tmp);

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
        *mad = blk_mad / (512.0 * 32.0);
}

extern "C" float cu_mad(int img_w, int img_h, int img_p, void *gm_img, void *gm_tmp)
{
    dim3 blocks(img_w / 512, img_h / 32);
    dim3 threads(32, 32);

    float mean, mad, *tmp;
    int blk_cnt, i;

    blk_cnt = blocks.x * blocks.y;
    tmp = (float *)malloc(blk_cnt * sizeof(float));

    mean_kernel<<<blocks, threads>>>(img_p, (float *)gm_img, (float *)gm_tmp);
    cudaMemcpy(tmp, gm_tmp, blk_cnt * sizeof(float), cudaMemcpyDeviceToHost);

    mean = 0.0;
    for (i = 0; i < blk_cnt; i++)
        mean += tmp[i];
    mean /= (float)blk_cnt;

    mad_kernel<<<blocks, threads>>>(img_p, (float *)gm_img, (float *)gm_tmp, mean);
    cudaMemcpy(tmp, gm_tmp, blk_cnt * sizeof(float), cudaMemcpyDeviceToHost);

    mad = 0.0;
    for (i = 0; i < blk_cnt; i++)
        mad += tmp[i];
    mad /= (float)blk_cnt;

    free(tmp);
    return mad;
}
