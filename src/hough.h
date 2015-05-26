#ifndef CU_HOUGH_H
#define CU_HOUGH_H

void cu_hough(int img_w, int img_h, int pitch, void *gm_abs, void *gm_phi, void *gm_hough);
void cu_center_detection(int img_w, int img_h, void *gm_hough, int *center_x, int *center_y);

#endif // CU_HOUGH_H
