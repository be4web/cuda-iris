extern "C" {
#include "convolve.h"
#include <stdint.h>
}

// http://developer.download.nvidia.com/assets/cuda/files/convolutionSeparable.pdf

/*!
 * USR: user specified identifier for this convolution
 * IN_T: input pixel type
 * OUT_T: output pixel type
 * MTX_T: matrix scalar type (this type is used for intermediate results!)
 * MTX_S: matrix size
 * BLOCK_W: block width (block x dimension); note: BLOCK_W must be >= (MTX_S / 2) to fill apron
 * BLOCK_H: block height (block y dimension)
 * STEPS: number of convolution steps performed (number of output pixels written) by each thread
 */
#define DECL_CU_CONVOLUTION_ROW(USR, IN_T, OUT_T, MTX_T, MTX_S, BLOCK_W, BLOCK_H, STEPS)                               \
__global__ void convol_row_k_##USR##MTX_S(IN_T *in, OUT_T *out, MTX_T *mtx, MTX_T div, int img_w)                      \
{                                                                                                                      \
    __shared__ IN_T bl_d[(BLOCK_H)][((STEPS) + 2) * (BLOCK_W)];                                                        \
                                                                                                                       \
    /* offset to left edge of apron */                                                                                 \
    const int off_x = (blockIdx.x * (STEPS) - 1) * (BLOCK_W) + threadIdx.x;                                            \
    const int off_y = blockIdx.y * (BLOCK_H) + threadIdx.y;                                                            \
                                                                                                                       \
    in += off_y * img_w + off_x;                                                                                       \
    out += off_y * img_w + off_x;                                                                                      \
                                                                                                                       \
    /* left apron */                                                                                                   \
    bl_d[threadIdx.y][threadIdx.x] = (off_x >= 0) ? in[0] : in[-off_x];                                                \
                                                                                                                       \
    /* main data */                                                                                                    \
_Pragma("unroll")                                                                                                      \
    for (int i = 1; i <= (STEPS); i++)                                                                                 \
        bl_d[threadIdx.y][threadIdx.x + i * (BLOCK_W)] = in[i * (BLOCK_W)];                                            \
                                                                                                                       \
    /* right apron */                                                                                                  \
    bl_d[threadIdx.y][threadIdx.x + ((STEPS) + 1) * (BLOCK_W)] = (img_w - off_x > ((STEPS) + 1) * (BLOCK_W)) ?         \
                                                             in[((STEPS) + 1) * (BLOCK_W)] : in[img_w - off_x - 1];    \
                                                                                                                       \
    __syncthreads();                                                                                                   \
                                                                                                                       \
_Pragma("unroll")                                                                                                      \
    for (int i = 1; i <= (STEPS); i++) {                                                                               \
        MTX_T sum = 0;                                                                                                 \
                                                                                                                       \
_Pragma("unroll")                                                                                                      \
        for (int j = -(MTX_S / 2); j <= ((MTX_S - 1) / 2); j++)                                                        \
            sum += mtx[(MTX_S / 2) + j] *  bl_d[threadIdx.y][threadIdx.x + i * (BLOCK_W) + j];                         \
                                                                                                                       \
        out[i * (BLOCK_W)] = sum / div;                                                                                \
    }                                                                                                                  \
}                                                                                                                      \
                                                                                                                       \
static void cu_convolve_row_##USR##MTX_S(void *gm_in, void *gm_out, void *gm_mtx, MTX_T div, int img_w, int img_h)     \
{                                                                                                                      \
    dim3 blocks(img_w / ((BLOCK_W) * (STEPS)), img_h / (BLOCK_H));                                                     \
    dim3 threads((BLOCK_W), (BLOCK_H));                                                                                \
                                                                                                                       \
    convol_row_k_##USR##MTX_S<<<blocks, threads>>>((IN_T *)gm_in, (OUT_T *)gm_out, (MTX_T *)gm_mtx, div, img_w);       \
}

/*!
 * USR: user specified identifier for this convolution
 * IN_T: input pixel type
 * OUT_T: output pixel type
 * MTX_T: matrix scalar type (this type is used for intermediate results!)
 * MTX_S: matrix size
 * BLOCK_W: block width (block x dimension)
 * BLOCK_H: block height (block y dimension); note: BLOCK_H must be >= (MTX_S / 2) to fill apron
 * STEPS: number of convolution steps performed (number of output pixels written) by each thread
 */
#define DECL_CU_CONVOLUTION_COL(USR, IN_T, OUT_T, MTX_T, MTX_S, BLOCK_W, BLOCK_H, STEPS)                               \
__global__ void convol_col_k_##USR##MTX_S(IN_T *in, OUT_T *out, MTX_T *mtx, MTX_T div, int img_w, int img_h)           \
{                                                                                                                      \
    __shared__ IN_T bl_d[(BLOCK_W)][((STEPS) + 2) * (BLOCK_H)]; /* +1 */                                               \
                                                                                                                       \
    /* offset to upper edge of apron */                                                                                \
    const int off_x = blockIdx.x * (BLOCK_W) + threadIdx.x;                                                            \
    const int off_y = (blockIdx.y * (STEPS) - 1) * (BLOCK_H) + threadIdx.y;                                            \
                                                                                                                       \
    in += off_y * img_w + off_x;                                                                                       \
    out += off_y * img_w + off_x;                                                                                      \
                                                                                                                       \
    /* upper apron */                                                                                                  \
    bl_d[threadIdx.x][threadIdx.y] = (off_y >= 0) ? in[0] : in[-off_y * img_w];                                        \
                                                                                                                       \
    /* main data */                                                                                                    \
_Pragma("unroll")                                                                                                      \
    for (int i = 1; i <= (STEPS); i++)                                                                                 \
        bl_d[threadIdx.x][threadIdx.y + i * (BLOCK_H)] = in[i * (BLOCK_H) * img_w];                                    \
                                                                                                                       \
    /* lower apron */                                                                                                  \
    bl_d[threadIdx.x][threadIdx.y + ((STEPS) + 1) * (BLOCK_H)] = (img_h - off_y > ((STEPS) + 1) * (BLOCK_H)) ?         \
                                              in[((STEPS) + 1) * (BLOCK_H) * img_w] : in[(img_h - off_y - 1) * img_w]; \
                                                                                                                       \
    __syncthreads();                                                                                                   \
                                                                                                                       \
_Pragma("unroll")                                                                                                      \
    for (int i = 1; i <= (STEPS); i++) {                                                                               \
        MTX_T sum = 0;                                                                                                 \
                                                                                                                       \
_Pragma("unroll")                                                                                                      \
        for (int j = -(MTX_S / 2); j <= ((MTX_S - 1) / 2); j++)                                                        \
            sum += mtx[(MTX_S / 2) + j] *  bl_d[threadIdx.x][threadIdx.y + i * (BLOCK_H) + j];                         \
                                                                                                                       \
        out[i * (BLOCK_H) * img_w] = sum / div;                                                                        \
    }                                                                                                                  \
}                                                                                                                      \
                                                                                                                       \
static void cu_convolve_col_##USR##MTX_S(void *gm_in, void *gm_out, void *gm_mtx, MTX_T div, int img_w, int img_h)     \
{                                                                                                                      \
    dim3 blocks(img_w / (BLOCK_W), img_h / ((BLOCK_H) * (STEPS)));                                                     \
    dim3 threads((BLOCK_W), (BLOCK_H));                                                                                \
                                                                                                                       \
    convol_col_k_##USR##MTX_S<<<blocks, threads>>>((IN_T *)gm_in, (OUT_T *)gm_out, (MTX_T *)gm_mtx, div, img_w, img_h);\
}

#define WARP_SIZE 32

DECL_CU_CONVOLUTION_ROW(gauss, uint8_t, uint8_t, int, 3, 1, (WARP_SIZE / 1), 8)
DECL_CU_CONVOLUTION_COL(gauss, uint8_t, uint8_t, int, 3, (WARP_SIZE / 1), 1, 8)

DECL_CU_CONVOLUTION_ROW(gauss, uint8_t, uint8_t, int, 5, 2, (WARP_SIZE / 2), 8)
DECL_CU_CONVOLUTION_COL(gauss, uint8_t, uint8_t, int, 5, (WARP_SIZE / 2), 2, 8)

DECL_CU_CONVOLUTION_ROW(gauss, uint8_t, uint8_t, int, 7, 4, (WARP_SIZE / 4), 4)
DECL_CU_CONVOLUTION_COL(gauss, uint8_t, uint8_t, int, 7, (WARP_SIZE / 4), 4, 4)

DECL_CU_CONVOLUTION_ROW(gauss, uint8_t, uint8_t, int, 9, 4, (WARP_SIZE / 4), 4)
DECL_CU_CONVOLUTION_COL(gauss, uint8_t, uint8_t, int, 9, (WARP_SIZE / 4), 4, 4)

DECL_CU_CONVOLUTION_ROW(gauss, uint8_t, uint8_t, int, 11, 8, (WARP_SIZE / 8), 4)
DECL_CU_CONVOLUTION_COL(gauss, uint8_t, uint8_t, int, 11, (WARP_SIZE / 8), 8, 4)

__constant__ int gauss_mtx_3[3] = { 1, 2, 1 };
__constant__ int gauss_mtx_5[5] = { 1, 4, 6, 4, 1 };
__constant__ int gauss_mtx_7[7] = { 1, 6, 15, 20, 15, 6, 1 };
__constant__ int gauss_mtx_9[9] = { 1, 8, 28, 56, 70, 56, 28, 8, 1 };
__constant__ int gauss_mtx_11[11] = { 1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1 };

extern "C" int cu_gauss_filter(int rad, int img_w, int img_h, void *gm_in, void *gm_out, void *gm_tmp)
{
    void *mtx;

    switch (rad) {
        case 3:
            cudaGetSymbolAddress(&mtx, gauss_mtx_3);
            cu_convolve_row_gauss3(gm_in, gm_tmp, mtx, 4, img_w, img_h);
            cu_convolve_col_gauss3(gm_tmp, gm_out, mtx, 4, img_w, img_h);
            break;

        case 5:
            cudaGetSymbolAddress(&mtx, gauss_mtx_5);
            cu_convolve_row_gauss5(gm_in, gm_tmp, mtx, 16, img_w, img_h);
            cu_convolve_col_gauss5(gm_tmp, gm_out, mtx, 16, img_w, img_h);
            break;

        case 7:
            cudaGetSymbolAddress(&mtx, gauss_mtx_7);
            cu_convolve_row_gauss7(gm_in, gm_tmp, mtx, 64, img_w, img_h);
            cu_convolve_col_gauss7(gm_tmp, gm_out, mtx, 64, img_w, img_h);
            break;

        case 9:
            cudaGetSymbolAddress(&mtx, gauss_mtx_9);
            cu_convolve_row_gauss9(gm_in, gm_tmp, mtx, 256, img_w, img_h);
            cu_convolve_col_gauss9(gm_tmp, gm_out, mtx, 256, img_w, img_h);
            break;

        case 11:
            cudaGetSymbolAddress(&mtx, gauss_mtx_11);
            cu_convolve_row_gauss11(gm_in, gm_tmp, mtx, 1024, img_w, img_h);
            cu_convolve_col_gauss11(gm_tmp, gm_out, mtx, 1024, img_w, img_h);
            break;

        default:
            return -1;
    }

    return 0;
}

DECL_CU_CONVOLUTION_ROW(gaussf, float, float, float, 11, 6, (WARP_SIZE / 6), 4)
DECL_CU_CONVOLUTION_COL(gaussf, float, float, float, 11, (WARP_SIZE / 6), 6, 4)

__constant__ float gauss_mtx_f11[11] = { 1.0, 10.0, 45.0, 120.0, 210.0, 252.0, 210.0, 120.0, 45.0, 10.0, 1.0 };

extern "C" void cu_gauss_filter_f11(int img_w, int img_h, void *gm_in, void *gm_out, void *gm_tmp)
{
    void *mtx;

    cudaGetSymbolAddress(&mtx, gauss_mtx_f11);
    cu_convolve_row_gaussf11(gm_in, gm_tmp, mtx, 1024.0, img_w, img_h);
    cu_convolve_col_gaussf11(gm_tmp, gm_out, mtx, 1024.0, img_w, img_h);
}

DECL_CU_CONVOLUTION_ROW(sobel, uint8_t, int16_t, int, 3, 1, (WARP_SIZE / 1), 8);
DECL_CU_CONVOLUTION_COL(sobel, int16_t, int16_t, int, 3, (WARP_SIZE / 1), 1, 8);

__constant__ int sobel_mtx_1[3] = { -1, 0, 1 };
__constant__ int sobel_mtx_2[3] = { 1, 2, 1 };

extern "C" int cu_sobel_filter(int img_w, int img_h, void *gm_in, void *gm_hori, void *gm_vert, void *gm_tmp)
{
    void *mtx1, *mtx2;

    cudaGetSymbolAddress(&mtx1, sobel_mtx_1);
    cudaGetSymbolAddress(&mtx2, sobel_mtx_2);

    cu_convolve_row_sobel3(gm_in, gm_tmp, mtx1, 1, img_w, img_h);
    cu_convolve_col_sobel3(gm_tmp, gm_hori, mtx2, 1, img_w, img_h);

    cu_convolve_row_sobel3(gm_in, gm_tmp, mtx2, 1, img_w, img_h);
    cu_convolve_col_sobel3(gm_tmp, gm_vert, mtx1, 1, img_w, img_h);

    return 0;
}

/*!
 * USR: user specified identifier for this convolution
 * IN_T: input pixel type
 * OUT_T: output pixel type
 * MTX_T: matrix scalar type (this type is used for intermediate results!)
 * MTX_S: matrix size
 * BLOCK_W: block width (block x dimension); note: BLOCK_W must be >= (MTX_S / 2) to fill apron
 * BLOCK_H: block height (block y dimension); note: BLOCK_H must be >= (MTX_S / 2) to fill apron
 * STEPS_X: number of convolution steps performed (number of output pixels written) by each thread in x direction
 * STEPS_Y: number of convolution steps performed (number of output pixels written) by each thread in y direction
 */
#define DECL_CU_CONVOLUTION(USR, IN_T, OUT_T, MTX_T, MTX_S, BLOCK_W, BLOCK_H, STEPS_X, STEPS_Y)                                                                \
__global__ void convol_kernel_##USR(IN_T *in, OUT_T *out, MTX_T *mtx, MTX_T div, int img_w, int img_h)                                                         \
{                                                                                                                                                              \
    __shared__ IN_T bl_d[((STEPS_X) + 2) * (BLOCK_W)][((STEPS_Y) + 2) * (BLOCK_H)];                                                                            \
                                                                                                                                                               \
    /* offset to upper left corner of apron */                                                                                                                 \
    const int off_x = (blockIdx.x * (STEPS_X) - 1) * (BLOCK_W) + threadIdx.x;                                                                                  \
    const int off_y = (blockIdx.y * (STEPS_Y) - 1) * (BLOCK_H) + threadIdx.y;                                                                                  \
                                                                                                                                                               \
    in += off_y * img_w + off_x;                                                                                                                               \
    out += off_y * img_w + off_x;                                                                                                                              \
                                                                                                                                                               \
    /* upper and lower apron */                                                                                                                                \
    {                                                                                                                                                          \
        IN_T *ua_in = in - img_w * ((off_y >= 0) ? 0 : off_y);                                                                                                 \
        IN_T *la_in = in + img_w * ((img_h - off_y > ((STEPS_Y) + 1) * (BLOCK_H)) ? ((STEPS_Y) + 1) * (BLOCK_H) : (img_h - off_y - 1));                        \
                                                                                                                                                               \
        /* upper left and lower left apron */                                                                                                                  \
        bl_d[threadIdx.x][threadIdx.y] = ua_in[(off_x >= 0) ? 0 : -off_x];                                                                                     \
        bl_d[threadIdx.x][threadIdx.y + ((STEPS_Y) + 1) * (BLOCK_H)] = la_in[(off_x >= 0) ? 0 : -off_x];                                                       \
                                                                                                                                                               \
        /* upper mid and lower mid apron */                                                                                                                    \
_Pragma("unroll")                                                                                                                                              \
        for (int x = 1; x <= (STEPS_X); x++) {                                                                                                                 \
            bl_d[threadIdx.x + x * (BLOCK_W)][threadIdx.y] = ua_in[x * (BLOCK_W)];                                                                             \
            bl_d[threadIdx.x + x * (BLOCK_W)][threadIdx.y + ((STEPS_Y) + 1) * (BLOCK_H)] = la_in[x * (BLOCK_W)];                                               \
        }                                                                                                                                                      \
                                                                                                                                                               \
        /* upper right and lower right apron */                                                                                                                \
        bl_d[threadIdx.x + ((STEPS_X) + 1) * (BLOCK_W)][threadIdx.y] =                                                                                         \
                                                       ua_in[(img_w - off_x > ((STEPS_X) + 1) * (BLOCK_W)) ? ((STEPS_X) + 1) * (BLOCK_W) : img_w - off_x - 1]; \
        bl_d[threadIdx.x + ((STEPS_X) + 1) * (BLOCK_W)][threadIdx.y + ((STEPS_Y) + 1) * (BLOCK_H)] =                                                           \
                                                       la_in[(img_w - off_x > ((STEPS_X) + 1) * (BLOCK_W)) ? ((STEPS_X) + 1) * (BLOCK_W) : img_w - off_x - 1]; \
    }                                                                                                                                                          \
                                                                                                                                                               \
    /* left and right apron */                                                                                                                                 \
    {                                                                                                                                                          \
        IN_T *la_in = in - ((off_x >= 0) ? 0 : off_x);                                                                                                         \
        IN_T *ra_in = in + ((img_w - off_x > ((STEPS_X) + 1) * (BLOCK_W)) ? ((STEPS_X) + 1) * (BLOCK_W) : img_w - off_x - 1);                                  \
                                                                                                                                                               \
_Pragma("unroll")                                                                                                                                              \
        for (int y = 1; y <= (STEPS_Y); y++) {                                                                                                                 \
            bl_d[threadIdx.x][threadIdx.y + y * (BLOCK_H)] = la_in[y * (BLOCK_H) * img_w];                                                                     \
            bl_d[threadIdx.x + ((STEPS_X) + 1) * (BLOCK_W)][threadIdx.y + y * (BLOCK_H)] = ra_in[y * (BLOCK_H) * img_w];                                       \
        }                                                                                                                                                      \
    }                                                                                                                                                          \
                                                                                                                                                               \
    /* main data */                                                                                                                                            \
_Pragma("unroll")                                                                                                                                              \
    for (int x = 1; x <= (STEPS_X); x++) {                                                                                                                     \
                                                                                                                                                               \
_Pragma("unroll")                                                                                                                                              \
        for (int y = 1; y <= (STEPS_Y); y++)                                                                                                                   \
            bl_d[threadIdx.x + x * (BLOCK_W)][threadIdx.y + y * (BLOCK_H)] = in[x * (BLOCK_W) + y * (BLOCK_H) * img_w];                                        \
    }                                                                                                                                                          \
                                                                                                                                                               \
    __syncthreads();                                                                                                                                           \
                                                                                                                                                               \
_Pragma("unroll")                                                                                                                                              \
    for (int x = 1; x <= (STEPS_X); x++) {                                                                                                                     \
                                                                                                                                                               \
_Pragma("unroll")                                                                                                                                              \
        for (int y = 1; y <= (STEPS_Y); y++) {                                                                                                                 \
            MTX_T sum = 0;                                                                                                                                     \
                                                                                                                                                               \
_Pragma("unroll")                                                                                                                                              \
            for (int i = -(MTX_S / 2); i <= ((MTX_S - 1) / 2); i++) {                                                                                          \
                                                                                                                                                               \
_Pragma("unroll")                                                                                                                                              \
                for (int j = -(MTX_S / 2); j <= ((MTX_S - 1) / 2); j++)                                                                                        \
                    sum += mtx[(MTX_S / 2) + i + MTX_S * ((MTX_S / 2) + j)] *  bl_d[threadIdx.x + x * (BLOCK_W) + i][threadIdx.y + y * (BLOCK_H) + j];         \
            }                                                                                                                                                  \
                                                                                                                                                               \
            out[x * (BLOCK_W) + y * (BLOCK_H) * img_w] = sum / div;                                                                                            \
        }                                                                                                                                                      \
    }                                                                                                                                                          \
}                                                                                                                                                              \
                                                                                                                                                               \
static void cu_convolve_##USR(void *gm_in, void *gm_out, void *gm_mtx, MTX_T div, int img_w, int img_h)                                                        \
{                                                                                                                                                              \
    dim3 blocks(img_w / ((BLOCK_W) * (STEPS_X)), img_h / ((BLOCK_H) * (STEPS_Y)));                                                                             \
    dim3 threads((BLOCK_W), (BLOCK_H));                                                                                                                        \
                                                                                                                                                               \
    convol_kernel_##USR<<<blocks, threads>>>((IN_T *)gm_in, (OUT_T *)gm_out, (MTX_T *)gm_mtx, div, img_w, img_h);                                              \
}

DECL_CU_CONVOLUTION(wavelet_65, uint8_t, float, float, 65, 32, 32, 4, 4)

__constant__ float wavelet_65[65][65];

extern "C" void cu_wavelet_filter_65(int img_w, int img_h, void *gm_in, void *gm_out, const float *wave_mtx, float div)
{
    void *mtx;
    cudaGetSymbolAddress(&mtx, wavelet_65);

    cudaMemcpyToSymbol(wavelet_65, wave_mtx, 65 * 65 * sizeof(float));
    cu_convolve_wavelet_65(gm_in, gm_out, mtx, div, img_w, img_h);
}

DECL_CU_CONVOLUTION_ROW(sf, uint8_t, float, float, 65, 32, 4, 4)
DECL_CU_CONVOLUTION_COL(sf, float, float, float, 65, 4, 32, 4)

__constant__ float mtx_f65[65];

extern "C" void cu_convolve_row_f65(int img_w, int img_h, void *gm_in, void *gm_out, const float *mtx, float div)
{
    void *mtx_s;
    cudaGetSymbolAddress(&mtx_s, mtx_f65);

    cudaMemcpyToSymbol(mtx_f65, mtx, 65 * sizeof(float));
    cu_convolve_row_sf65(gm_in, gm_out, mtx_s, div, img_w, img_h);
}

extern "C" void cu_convolve_col_f65(int img_w, int img_h, void *gm_in, void *gm_out, const float *mtx, float div)
{
    void *mtx_s;
    cudaGetSymbolAddress(&mtx_s, mtx_f65);

    cudaMemcpyToSymbol(mtx_f65, mtx, 65 * sizeof(float));
    cu_convolve_col_sf65(gm_in, gm_out, mtx_s, div, img_w, img_h);
}
