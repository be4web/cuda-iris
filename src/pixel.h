#ifndef CU_PIXEL_H
#define CU_PIXEL_H

/*!
 * \file pixel.h
 * pixel-wise operations
 */

#include <stdint.h>

/*!
 * Convert color image to gray-scale image
 *
 * img_w: image width
 * img_h: image height
 * color_p: pitch of the color input data (in samples)
 * gm_color: graphic memory for color input data (32 bit wide)
 * gray_p: pitch of the gray output data
 * gm_gray: graphic memory for gray output data (8 bit wide)
 * coeff_r: coefficient for red channel
 * coeff_g: coefficient for green channel
 * coeff_b: coefficient for blue channel
 */
void cu_color_to_gray(int img_w, int img_h, int color_p, void *gm_color, int gray_p, void *gm_gray, int coeff_r, int coeff_g, int coeff_b);

/*!
 * Convert cartesian gradient data (i.e. x and y direction of gradient for each pixel) to polar gradient data (absolute value and angle for each pixel)
 *
 * img_w: image width
 * img_h: image height
 * hori_p: pitch of the horizontal gradient data (in samples)
 * gm_hori: graphic memory for horizontal gradient data (16 bit wide)
 * vert_p: pitch of the vertical gradient data (in samples)
 * gm_vert: graphic memory for vertical gradient data (16 bit wide)
 * abs_p: pitch of the gradient value data (in samples)
 * gm_abs: graphic memory for absolute gradient value data (single precision floating point)
 * phi_p: pitch of the gradient angle data (in samples)
 * gm_phi: graphic memory for gradient angle data (single precision floating point)
 */
void cu_cart_to_polar(int img_w, int img_h, int hori_p, void *gm_hori, int vert_p, void *gm_vert, int abs_p, void *gm_abs, int phi_p, void *gm_phi);

/*!
 * Substitute pixels in gray-scale image
 *
 * img_w: image width
 * img_h: image height
 * in_p: pitch of the input data
 * gm_in: graphic memory for input data (8 bit wide)
 * out_p: pitch of the output data
 * gm_out: graphic memory for output data (8 bit wide)
 * sub: substitution rules (array with 256 bytes)
 */
void cu_pixel_substitute(int img_w, int img_h, int in_p, void *gm_in, int out_p, void *gm_out, uint8_t *sub);

/*!
 * Normalize gradient data according to center (accepts gradient data in polar coordinates)
 *
 * img_w: image width
 * img_h: image height
 * abs_p: pitch of the gradient value data (in samples)
 * gm_abs: graphic memory for absolute gradient value data (single precision floating point)
 * phi_p: pitch of the gradient angle data (in samples)
 * gm_phi: graphic memory for gradient angle data (single precision floating point)
 * norm_p: pitch of the normalized gradient data (in samples)
 * gm_norm: graphic memory for normalized gradient data (single precision floating point)
 * center_x: x coordinate of the center
 * center_y: y coordinate of the center
 */
void cu_centered_gradient_normalization(int img_w, int img_h, int abs_p, void *gm_abs, int phi_p, void *gm_phi, int norm_p, void *gm_norm, int center_x, int center_y);

/*!
 * Resize image gray-scale image
 *
 * img_w: image width of source image
 * img_h: image height of source image
 * src_p: pitch of the input data
 * gm_src: graphic memory for input data (8 bit wide)
 * dst_w: image width of destination image
 * dst_h: image height of the destination image
 * dst_p: pitch of the output data
 * gm_dst: graphic memory for output data (8 bit wide)
 */
void cu_image_resize(int img_w, int img_h, int src_p, void *gm_src, int dst_w, int dst_h, int dst_p, void *gm_dst);

#endif // CU_PIXEL_H
