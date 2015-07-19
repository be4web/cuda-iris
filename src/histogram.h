#ifndef CU_REDUCTION_H
#define CU_REDUCTION_H

/*!
 * \file
 * image histogram
 */

void cu_histogram(int img_w, int img_h, int img_p, void *gm_img, void *gm_tmp, int *histo);

float cu_mad(int img_w, int img_h, int img_p, void *gm_img, void *gm_tmp);

#endif // CU_REDUCTION_H
