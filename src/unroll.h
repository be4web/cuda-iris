#ifndef UNROLL_H
#define UNROLL_H

/*!
 * \file unroll.h
 * iris unrolling
 */

#define CU_UNROLL_W 1024
#define CU_UNROLL_H 256

/*!
 * Unroll iris located at given center and with given inner and outer boundary
 *
 * img_w: image width of input image
 * img_h: image height of input image
 * img_p: pitch of the input data (in samples)
 * gm_img: graphic memory for input data (32 bit wide)
 * gm_out: graphic memory for output data (32 bit wide)
 * center_x: x coordinate of center
 * center_y: y coordinate of center
 * inner_rad: array of 32 floats, describing inner iris boundary
 * outer_rad: array of 32 floats, describing outer iris boundary
 * gm_cut: graphic memory to be filled with the cut mask (for debug purposes only)
 */
void cu_unroll(int img_w, int img_h, int img_p, void *gm_img, void *gm_out, int center_x, int center_y, float *inner_rad, float *outer_rad, void *gm_cut);

#endif // UNROLL_H
