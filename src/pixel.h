#ifndef CU_PIXEL_H
#define CU_PIXEL_H

/*!
 * \file
 * pixel-wise operations
 */

#include <stdint.h>

/*!
 * img_w: image width
 * img_h: image height
 * gm_color: graphic memory for color input data (32 bit wide)
 * gm_gray: graphic memory for gray output data (8 bit wide)
 */
void cu_color_to_gray(int img_w, int img_h, void *gm_color, void *gm_gray);

/*!
 * img_w: image width
 * img_h: image height
 * gm_hori: graphic memory for horizontal gradient data (16 bit wide)
 * gm_vert: graphic memory for vertical gradient data (16 bit wide)
 * gm_rad: graphic memory for gradient value (16 bit wide)
 * gm_phi: graphic memory for gradient angle (single precision floating point)
 */
void cu_cart_to_polar(int img_w, int img_h, void *gm_hori, void *gm_vert, void *gm_rad, void *gm_phi);

void cu_pixel_substitute(int img_w, int img_h, void *gm_in, void *gm_out, uint8_t *sub);

void cu_centered_gradient_normalization(int img_w, int img_h, int center_x, int center_y, void *gm_abs, void *gm_phi, void *gm_norm);

#endif // CU_PIXEL_H
