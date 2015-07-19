#ifndef CU_HOUGH_H
#define CU_HOUGH_H

/*!
 * \file hough.h
 * convolution operations
 */

/*!
 * Apply a Hough transform to gradient data in polar coordinates
 *
 * img_w: width of image
 * img_h: height of image
 * abs_p: pitch of the gradient value data (in samples)
 * gm_abs: graphic memory for absolute gradient value data (single precision floating point)
 * phi_p: pitch of the gradient angle data (in samples)
 * gm_phi: graphic memory for gradient angle data (single precision floating point)
 * hough_p: pitch of hough data (in samples)
 * gm_hough: graphic memory for hough data (32 bit integer)
 * min_rad: minimum radius (in pixels)
 * max_rad: maximum radius (in pixels)
 */
void cu_hough(int img_w, int img_h, int abs_p, void *gm_abs, int phi_p, void *gm_phi, int hough_p, void *gm_hough, float min_rad, float max_rad);

/*!
 * Apply a Sobel filter to an 8 bit gray-scale image
 *
 * img_w: width of image
 * img_h: height of image
 * hough_p: pitch of hough data (in samples)
 * gm_hough: graphic memory for hough data (32 bit integer)
 * center_x: pointer to an integer that will receive the x coordinate of the center
 * center_y: pointer to an integer that will receive the y coordinate of the center
 */
void cu_center_detection(int img_w, int img_h, void *gm_hough, int *center_x, int *center_y);

#endif // CU_HOUGH_H
