extern "C" {
#include "unroll.h"
#include <stdint.h>
}

#define PI 3.141592654

#define UNROLL_W32 (CU_UNROLL_W / 32)

texture<int, cudaTextureType2D> img_tex;

__global__ void unroll_kernel(int out_p, int *gm_out, float center_x, float center_y, float *inner_rad, float *outer_rad, int cut_p, uint8_t *gm_cut)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int prev_rad = x / UNROLL_W32;
    int next_rad = prev_rad + 1;

    int inner = inner_rad[prev_rad] * (float)(next_rad * UNROLL_W32 - x) + inner_rad[next_rad & 0x1f] * (float)(x - prev_rad * UNROLL_W32);
    int outer = outer_rad[prev_rad] * (float)(next_rad * UNROLL_W32 - x) + outer_rad[next_rad & 0x1f] * (float)(x - prev_rad * UNROLL_W32);
    inner /= UNROLL_W32;
    outer /= UNROLL_W32;

    float phi = (float)x * 2.0 * PI / (float)CU_UNROLL_W;
    float rad = (float)y * (outer - inner) / (float)CU_UNROLL_H + inner;

    gm_out[out_p * y + x] = tex2D(img_tex, center_x + rad * cosf(phi), center_y + rad * sinf(phi));

    // debug
    gm_cut[(int)(center_y + rad * sinf(phi)) * cut_p + (int)(center_x + rad * cosf(phi))] = 120;
}

extern "C" void cu_unroll(int img_w, int img_h, int img_p, void *gm_img, void *gm_out, int center_x, int center_y, float *inner_rad, float *outer_rad, void *gm_cut)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, img_tex, gm_img, desc, img_w, img_h, img_p * 4);

    void *gm_inner_rad, *gm_outer_rad;
    cudaMalloc(&gm_inner_rad, 32 * sizeof(float));
    cudaMalloc(&gm_outer_rad, 32 * sizeof(float));
    cudaMemcpy(gm_inner_rad, inner_rad, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gm_outer_rad, outer_rad, 32 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(CU_UNROLL_W / 8, CU_UNROLL_H / 8);
    dim3 threads(8, 8);

    unroll_kernel<<<blocks, threads>>>(CU_UNROLL_W, (int *)gm_out, (float)center_x, (float)center_y, (float *)gm_inner_rad, (float *)gm_outer_rad, img_w, (uint8_t *)gm_cut);

    cudaFree(gm_inner_rad);
    cudaFree(gm_outer_rad);
}
