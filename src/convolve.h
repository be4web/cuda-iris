#ifndef CU_CONVOLVE_H
#define CU_CONVOLVE_H

/*!
 * \file convolve.h
 * convolution operations
 */

/*!
 * Apply a Gaussian blur filter to an 8 bit gray-scale image
 *
 * rad: radius of gaussian blur
 * img_w: width of image
 * img_h: height of image
 * gm_in: graphic memory for input data (8 bit wide)
 * gm_out: graphic memory for output data (8 bit wide)
 * gm_tmp: graphic memory for temporary image storage (8 bit wide)
 */
int cu_gauss_filter(int rad, int img_w, int img_h, void *gm_in, void *gm_out, void *gm_tmp);

/*!
 * Apply a Sobel filter to an 8 bit gray-scale image
 *
 * img_w: width of image
 * img_h: height of image
 * gm_in: graphic memory for input data (8 bit wide)
 * gm_hori: graphic memory for horizontal gradient data (16 bit wide)
 * gm_vert: graphic memory for vertical gradient data (16 bit wide)
 * gm_tmp: graphic memory for temporary image storage (16 bit wide)
 */
int cu_sobel_filter(int img_w, int img_h, void *gm_in, void *gm_hori, void *gm_vert, void *gm_tmp);

/*!
 * Apply a Gaussian blur filter with a diameter of 11 pixel to a single precision floating point gray-scale image
 *
 * img_w: width of image
 * img_h: height of image
 * gm_in: graphic memory for input data (single precision floating point)
 * gm_out: graphic memory for output data (single precision floating point)
 * gm_tmp: graphic memory for temporary image storage (32 bit wide)
 */
void cu_gauss_filter_f11(int img_w, int img_h, void *gm_in, void *gm_out, void *gm_tmp);

/*!
 * Apply a separable 65x65 matrix filter to an 8-bit gray-scale image
 *
 * img_w: width of image
 * img_h: height of image
 * gm_in: graphic memory for input data (8 bit wide)
 * gm_out: graphic memory for output data (single precision floating point)
 * mtx: separated matrix vectors (65 entries)
 * div: division factor
 */
void cu_convolve_row_f65(int img_w, int img_h, void *gm_in, void *gm_out, const float *mtx, float div);
void cu_convolve_col_f65(int img_w, int img_h, void *gm_in, void *gm_out, const float *mtx, float div);

#endif // CU_CONVOLVE_H
