extern "C" {
#include "pixel.h"
#include <stdint.h>
}

__global__ void color_to_gray_kernel(int color_p, uint32_t *color, int gray_p, uint8_t *gray, int coeff_r, int coeff_g, int coeff_b)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int div = coeff_r + coeff_g + coeff_b;
    uint32_t clr = color[color_p * y + x];

    gray[gray_p * y + x] = ((clr & 0xff) * coeff_r + ((clr >> 8) & 0xff) * coeff_g + ((clr >> 16) & 0xff) * coeff_b) / div;
}

extern "C" void cu_color_to_gray(int img_w, int img_h, int color_p, void *gm_color, int gray_p, void *gm_gray)
{
    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    color_to_gray_kernel<<<blocks, threads>>>(color_p, (uint32_t *)gm_color, gray_p, (uint8_t *)gm_gray, 1, 1, 1);
}

__global__ void cart_to_polar_kernel(int hori_p, int16_t *hori, int vert_p, int16_t *vert, int abs_p, float *abs, int phi_p, float *phi)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int grad_x = hori[hori_p * y + x],
        grad_y = vert[vert_p * y + x];

    //rad[p] = (unsigned int)(grad_x * grad_x + grad_y * grad_y) >> 5;
    abs[abs_p * y + x] = sqrtf((float)(grad_x * grad_x + grad_y * grad_y));
    phi[phi_p * y + x] = atan2f((float)grad_y, (float)grad_x);
}

extern "C" void cu_cart_to_polar(int img_w, int img_h, int hori_p, void *gm_hori, int vert_p, void *gm_vert, int abs_p, void *gm_abs, int phi_p, void *gm_phi)
{
    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    cart_to_polar_kernel<<<blocks, threads>>>(hori_p, (int16_t *)gm_hori, vert_p, (int16_t *)gm_vert, abs_p, (float *)gm_abs, phi_p, (float *)gm_phi);
}

__global__ void pixel_substitute_kernel(int in_p, uint8_t *in, int out_p, uint8_t *out, uint8_t *sub)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    out[out_p * y + x] = sub[in[in_p * y + x]];
}

extern "C" void cu_pixel_substitute(int img_w, int img_h, int in_p, void *gm_in, int out_p, void *gm_out, uint8_t *sub)
{
    void *gm_sub;
    cudaMalloc(&gm_sub, 256);
    cudaMemcpy(gm_sub, sub, 256, cudaMemcpyHostToDevice);

    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    pixel_substitute_kernel<<<blocks, threads>>>(in_p, (uint8_t *)gm_in, out_p, (uint8_t *)gm_out, (uint8_t *)gm_sub);

    cudaFree(&gm_sub);
}

#define GAUSS_STD_DEV 0.3 // gaussian standard deviation

__global__ void centered_gradient_normalization_kernel(int abs_p, float *abs, int phi_p, float *phi, int norm_p, float *norm, int center_x, int center_y)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    float grad_abs = abs[abs_p * y + x],
          grad_phi = phi[phi_p * y + x];

    float center_phi = atan2f((float)(y - center_y), (float)(x - center_x));

    float phi_diff = grad_phi - center_phi;

    // phi difference is normalized using gaussian function
    float norm_phi_diff = expf(-(phi_diff * phi_diff) / GAUSS_STD_DEV);

    norm[norm_p * y + x] = grad_abs * norm_phi_diff;
}

extern "C" void cu_centered_gradient_normalization(int img_w, int img_h, int abs_p, void *gm_abs, int phi_p, void *gm_phi, int norm_p, void *gm_norm, int center_x, int center_y)
{
    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    centered_gradient_normalization_kernel<<<blocks, threads>>>(abs_p, (float *)gm_abs, phi_p, (float *)gm_phi, norm_p, (float *)gm_norm, center_x, center_y);
}

__global__ void image_resize_kernel(int src_p, uint8_t *src, int dst_p, uint8_t *dst, int img_w, int img_h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const float x_ratio = (float) img_w / (gridDim.x * blockDim.x);
	const float y_ratio = (float) img_h / (gridDim.y * blockDim.y);
	const int px = (int) floor((float) x * x_ratio), py = (int) floor((float) y * y_ratio);

	dst[x + y * dst_p] = src[src_p * py + px];	
}

extern "C" void cu_image_resize(int img_w, int img_h, int src_p, void *gm_src, int dst_w, int dst_h, int dst_p, void *gm_dst) {
	dim3 blocks(dst_w / 8, dst_h / 8);
	dim3 threads(8, 8);
	image_resize_kernel<<<blocks, threads>>>(src_p, (uint8_t *) gm_src, dst_p, (uint8_t *) gm_dst, img_w, img_h);

}
