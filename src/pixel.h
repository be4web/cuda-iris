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
 * color_p: pitch of the color input data (in samples)
 * gm_color: graphic memory for color input data (32 bit wide)
 * gray_p: pitch of the gray output data
 * gm_gray: graphic memory for gray output data (8 bit wide)
 * coeff_r:
 * coeff_g:
 * coeff_b:
 */
void cu_color_to_gray(int img_w, int img_h, int color_p, void *gm_color, int gray_p, void *gm_gray, int coeff_r, int coeff_g, int coeff_b);

/*!
 * img_w: image width
 * img_h: image height
 * hori_p: pitch of the horizontal gradient data (in samples)
 * gm_hori: graphic memory for horizontal gradient data (16 bit wide)
 * vert_p: pitch of the vertical gradient data (in samples)
 * gm_vert: graphic memory for vertical gradient data (16 bit wide)
 * abs_p: pitch of the gradient value data (in samples)
 * gm_abs: graphic memory for gradient value data (single precision floating point)
 * phi_p: pitch of the gradient angle data (in samples)
 * gm_phi: graphic memory for gradient angle data (single precision floating point)
 */
void cu_cart_to_polar(int img_w, int img_h, int hori_p, void *gm_hori, int vert_p, void *gm_vert, int abs_p, void *gm_abs, int phi_p, void *gm_phi);

void cu_pixel_substitute(int img_w, int img_h, int in_p, void *gm_in, int out_p, void *gm_out, uint8_t *sub);

void cu_centered_gradient_normalization(int img_w, int img_h, int abs_p, void *gm_abs, int phi_p, void *gm_phi, int norm_p, void *gm_norm, int center_x, int center_y);

void cu_image_resize(int img_w, int img_h, int src_p, void *gm_src, int dst_w, int dst_h, int dst_p, void *gm_dst);

#endif // CU_PIXEL_H
