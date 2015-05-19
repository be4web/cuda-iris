extern "C" {
#include "pixel.h"
#include <stdint.h>
}

__global__ void color_to_gray_kernel(int img_w, uint32_t *color, uint8_t *gray, int coeff_r, int coeff_g, int coeff_b)
{
    const int x = blockIdx.x * 8 + threadIdx.x;
    const int y = blockIdx.y * 8 + threadIdx.y;
    const int p = img_w * y + x;

    int div = coeff_r + coeff_g + coeff_b;
    uint32_t clr = color[p];

    gray[p] = ((clr & 0xff) * coeff_r + ((clr >> 8) & 0xff) * coeff_g + ((clr >> 16) & 0xff) * coeff_b) / div;
}

extern "C" void cu_color_to_gray(int img_w, int img_h, void *gm_color, void *gm_gray)
{
    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    color_to_gray_kernel<<<blocks, threads>>>(img_w, (uint32_t *)gm_color, (uint8_t *)gm_gray, 1, 1, 1);
}

__global__ void cart_to_polar_kernel(int img_w, int16_t *hori, int16_t *vert, float *rad, float *phi)
{
    const int x = blockIdx.x * 8 + threadIdx.x;
    const int y = blockIdx.y * 8 + threadIdx.y;
    const int p = img_w * y + x;

    int grad_x = hori[p],
        grad_y = vert[p];

    //rad[p] = (unsigned int)(grad_x * grad_x + grad_y * grad_y) >> 5;
    rad[p] = sqrt((float)(grad_x * grad_x + grad_y * grad_y));
    phi[p] = atan2((double)grad_y, (double)grad_x);
}

extern "C" void cu_cart_to_polar(int img_w, int img_h, void *gm_hori, void *gm_vert, void *gm_rad, void *gm_phi)
{
    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    cart_to_polar_kernel<<<blocks, threads>>>(img_w, (int16_t *)gm_hori, (int16_t *)gm_hori, (float *)gm_rad, (float *)gm_phi);
}

__global__ void pixel_substitute_kernel(int img_w, uint8_t *in, uint8_t *out, uint8_t *sub)
{
    const int x = blockIdx.x * 8 + threadIdx.x;
    const int y = blockIdx.y * 8 + threadIdx.y;
    const int p = img_w * y + x;

    out[p] = sub[in[p]];
}

extern "C" void cu_pixel_substitute(int img_w, int img_h, void *gm_in, void *gm_out, uint8_t *sub)
{
    void *gm_sub;
    cudaMalloc(&gm_sub, 256);
    cudaMemcpy(gm_sub, sub, 256, cudaMemcpyHostToDevice);

    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    pixel_substitute_kernel<<<blocks, threads>>>(img_w, (uint8_t *)gm_in, (uint8_t *)gm_out, (uint8_t *)gm_sub);
}
