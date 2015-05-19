#ifndef CU_CONVOLVE_H
#define CU_CONVOLVE_H

/*!
 * \file
 * convolution operations
 */

/*!
 * rad: radius of gaussian blur
 * img_w: width of image
 * img_h: height of image
 * gm_in: graphic memory for input data (8 bit wide)
 * gm_out: graphic memory for output data (8 bit wide)
 * gm_tmp: graphic memory for temporary image storage (8 bit wide)
 */
int cu_gauss_filter(int rad, int img_w, int img_h, void *gm_in, void *gm_out, void *gm_tmp);

/*!
 * img_w: width of image
 * img_h: height of image
 * gm_in: graphic memory for input data (8 bit wide)
 * gm_hori: graphic memory for horizontal gradient data (16 bit wide)
 * gm_vert: graphic memory for vertical gradient data (16 bit wide)
 * gm_tmp: graphic memory for temporary image storage (16 bit wide)
 */
int cu_sobel_filter(int img_w, int img_h, void *gm_in, void *gm_hori, void *gm_vert, void *gm_tmp);

#endif // CU_CONVOLVE_H
