#ifndef UNROLL_H
#define UNROLL_H

#define CU_UNROLL_W 1024
#define CU_UNROLL_H 256

void cu_unroll(int img_w, int img_h, int img_p, void *gm_img, void *gm_out, int center_x, int center_y, float *inner_rad, float *outer_rad, void *gm_cut);

#endif // UNROLL_H
