#ifndef UNROLL_H
#define UNROLL_H

#define CU_UNROLL_W 704
#define CU_UNROLL_H 256

void cu_unroll(int img_w, int img_h, int pitch, int center_x, int center_y, int inner_rad, int outer_rad, void *gm_img, void *gm_out);

#endif // UNROLL_H
