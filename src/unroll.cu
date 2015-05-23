extern "C" {
#include "unroll.h"
#include <stdint.h>
}

#define PI 3.141592654

texture<int, cudaTextureType2D> img_tex;

__global__ void unroll_kernel(float center_x, float center_y, float inner_rad, float outer_rad, int *gm_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float phi = (float)x * 2.0 * PI / (float)CU_UNROLL_W;
    float rad = (float)y * (outer_rad - inner_rad) / (float)CU_UNROLL_H + inner_rad;

    gm_out[CU_UNROLL_W * y + x] = tex2D(img_tex, center_x + rad * cosf(phi), center_y + rad * sinf(phi));
}

extern "C" void cu_unroll(int img_w, int img_h, int pitch, int center_x, int center_y, int inner_rad, int outer_rad, void *gm_img, void *gm_out)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, img_tex, gm_img, desc, img_w, img_h, pitch);

    dim3 blocks(CU_UNROLL_W / 8, CU_UNROLL_H / 8);
    dim3 threads(8, 8);

    unroll_kernel<<<blocks, threads>>>((float)center_x, (float)center_y, (float)inner_rad, (float)outer_rad, (int *)gm_out);
}
