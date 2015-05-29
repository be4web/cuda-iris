#ifndef CU_HOUGH_H
#define CU_HOUGH_H

void cu_hough(int img_w, int img_h, int abs_p, void *gm_abs, int phi_p, void *gm_phi, int hough_p, void *gm_hough, float min_rad, float max_rad);
void cu_center_detection(int img_w, int img_h, void *gm_hough, int *center_x, int *center_y);

#endif // CU_HOUGH_H
