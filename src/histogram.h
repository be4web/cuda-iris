#ifndef CU_REDUCTION_H
#define CU_REDUCTION_H

/*!
 * \file histogram.h
 * image histogram and mean absolute deviation
 */

/*!
 * Generate histogram of an 8-bit gray-scale image
 *
 * img_w: image width of input image
 * img_h: image height of input image
 * img_p: pitch of the image data
 * gm_img: graphic memory for image data (8 bit wide)
 * gm_tmp: graphic memory for temporary data (32 bit wide)
 * histo: output buffer for histogram data (256 integers)
 */
void cu_histogram(int img_w, int img_h, int img_p, void *gm_img, void *gm_tmp, int *histo);

/*!
 * Calculate the mean absolute deviation of a single precision floating point gray-scale image
 *
 * img_w: image width of input image
 * img_h: image height of input image
 * img_p: pitch of the image data (in samples)
 * gm_img: graphic memory for image data (single precision floating point)
 * gm_tmp: graphic memory for temporary data (32 bit wide)
 *
 * \return mean abolute deviation of the image
 */
float cu_mad(int img_w, int img_h, int img_p, void *gm_img, void *gm_tmp);

#endif // CU_REDUCTION_H
