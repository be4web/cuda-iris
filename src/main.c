#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <cuda_runtime.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#include <math.h>

#include "pixel.h"
#include "convolve.h"
#include "reduction.h"
#include "hough.h"
#include "unroll.h"

#define PI 3.141592654

int main(int argc, char *argv[])
{
    GdkPixbuf *img;
    GError *err = NULL;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <image file>\n", argv[0]);
        return 1;
    }

    if ((img = gdk_pixbuf_new_from_file(argv[1], &err)) == NULL) {
        fprintf(stderr, "error loading\n");
        return 2;
    }

    int img_w = gdk_pixbuf_get_width(img),
        img_h = gdk_pixbuf_get_height(img);
        //img_s = gdk_pixbuf_get_rowstride(img),
        //img_c = gdk_pixbuf_get_n_channels(img),
        //img_b = gdk_pixbuf_get_bits_per_sample(img);

    uint8_t *img_d = gdk_pixbuf_get_pixels(img);

    void *gm_img, *gm_sobel_h, *gm_sobel_v, *gm_sobel_1, *gm_sobel_2, *gm_tmp, *gm_sobel_abs, *gm_sobel_phi;
    size_t sobel_pitch;

    cudaMalloc(&gm_img, img_w * img_h);
    cudaMalloc(&gm_sobel_h, img_w * img_h * 4);
    cudaMalloc(&gm_sobel_v, img_w * img_h * 4);
    cudaMalloc(&gm_sobel_1, img_w * img_h * 4);
    cudaMalloc(&gm_sobel_2, img_w * img_h * 4);
    cudaMalloc(&gm_tmp, img_w * img_h * 4);

    cudaMallocPitch(&gm_sobel_abs, &sobel_pitch, img_w * sizeof(float), img_h);
    cudaMallocPitch(&gm_sobel_phi, &sobel_pitch, img_w * sizeof(float), img_h);

    cudaMemcpy(gm_tmp, img_d, img_w * img_h * 4, cudaMemcpyHostToDevice);

    cu_color_to_gray(img_w, img_h, gm_tmp, gm_img);

    //! Gauss Filter:
    if (cu_gauss_filter(11, img_w, img_h, gm_img, gm_img, gm_tmp) < 0)
        fprintf(stderr, "error applying gauss filter\n");

    {
        uint8_t *gray_d = malloc(img_w * img_h);
        cudaMemcpy(gray_d, gm_img, img_w * img_h, cudaMemcpyDeviceToHost);

        FILE *file = fopen("gauss.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
        int p;
        for (p = 0; p < img_w * img_h; p++)
            fputc(gray_d[p], file);
        fclose(file);
    }

    //! Sobel Filter:
    cu_sobel_filter(img_w, img_h, gm_img, gm_sobel_h, gm_sobel_v, gm_tmp);

    {
        int16_t *sobel_h, *sobel_v;
        sobel_h = malloc(img_w * img_h * 2);
        sobel_v = malloc(img_w * img_h * 2);

        cudaMemcpy(sobel_h, gm_sobel_h, img_w * img_h * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(sobel_v, gm_sobel_v, img_w * img_h * 2, cudaMemcpyDeviceToHost);

        FILE *file = fopen("sobel.ppm", "w");
        fprintf(file, "P6\n%d %d\n255\n", img_w, img_h);
        int p;
        for (p = 0; p < img_w * img_h; p++) {
            fputc((sobel_h[p] >> 1) + 128, file);
            fputc(0, file);
            fputc((sobel_v[p] >> 1) + 128, file);
        }
        fclose(file);
    }

    //! Transformation der Gradienten (Sobel) von kartesisch nach polar:
    cu_cart_to_polar(img_w, img_h, gm_sobel_h, gm_sobel_v, gm_sobel_1, gm_sobel_2);

    cudaMemcpy2D(gm_sobel_abs, sobel_pitch, gm_sobel_1, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(gm_sobel_phi, sobel_pitch, gm_sobel_2, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyDeviceToDevice);

    //! Hough Transformation:
    cu_hough(img_w, img_h, sobel_pitch, gm_sobel_abs, gm_sobel_phi, gm_tmp);

    {
        int *hough_d;
        hough_d = malloc(img_w * img_h * 4);

        cudaMemcpy(hough_d, gm_tmp, img_w * img_h * 4, cudaMemcpyDeviceToHost);
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(cudaGetLastError()));

        FILE *file = fopen("hough.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
        int p, v;
        for (p = 0; p < img_w * img_h; p++) {
            v = hough_d[p] >> 16;
            fputc((v > 255) ? 255 : v, file);
        }
        fclose(file);
    }

    int center_x, center_y;

    //! Berechnung des Zentrums:
    cu_center_detection(img_w, img_h, gm_tmp, &center_x, &center_y);
    printf("center: (%d, %d)\n", center_x, center_y);

    /*
    {
        center_x = 0;
        center_y = 0;

        {
            int x, y, c = 0;
            for (x = 0; x < img_w; x++)
                for (y = 0; y < img_h; y++)
                    if (hough_d[y * img_w + x] > (1 << 22)) {
                        center_x += x;
                        center_y += y;
                        c++;
                    }

            if (c == 0) {
                fprintf(stderr, "error: no center detected\n");
                return 1;
            }

            center_x /= c;
            center_y /= c;
        }

        printf("center: (%d, %d)\n", center_x, center_y);

        int *hough_radius;
        hough_radius = get_hough_radius(sobel_h, sobel_v, img_w, img_h, center_x, center_y);

        {
            FILE *file = fopen("radius.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", MAX_RAD, 200);
            int x, y;
            for (y = 0; y < 200; y++)
                for (x = 0; x < MAX_RAD; x++)
                    fputc((hough_radius[x] > (200 - y)) ? 255 : 0, file);
            fclose(file);
        }

        inner_rad = 0;
        outer_rad = 0;

        {
            int rad;
            for (rad = MIN_RAD; rad < MAX_RAD; rad++)
                if (hough_radius[rad] > hough_radius[outer_rad])
                    outer_rad = rad;

            for (rad = MIN_RAD; rad < outer_rad * 3 / 4; rad++)
                if (hough_radius[rad] > hough_radius[inner_rad])
                    inner_rad = rad;
        }
    }
    */

    float inner_rad[32], outer_rad[32];

    //! Berechnung des inneren und aeusseren Radius:
    {
        cu_centered_gradient_normalization(img_w, img_h, center_x, center_y, gm_sobel_1, gm_sobel_2, gm_tmp);

        float *norm;
        norm = malloc(img_w * img_h * 4);

        cudaMemcpy(norm, gm_tmp, img_w * img_h * 4, cudaMemcpyDeviceToHost);

        {
            FILE *file = fopen("sobel_norm.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
            int p;
            for (p = 0; p < img_w * img_h; p++)
                fputc((int)norm[p], file);
            fclose(file);
        }

        void *gm_norm, *gm_norm_unr, *gm_tmp_unr;
        size_t pitch;
        cudaMallocPitch(&gm_norm, &pitch, img_w * sizeof(float), img_h);
        cudaMalloc(&gm_norm_unr, CU_UNROLL_W * CU_UNROLL_H * 4);
        cudaMalloc(&gm_tmp_unr, CU_UNROLL_W * CU_UNROLL_H * 4);

        cudaMemcpy2D(gm_norm, pitch, gm_tmp, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyDeviceToDevice);

        int i;
        for (i = 0; i < 32; i++) {
            inner_rad[i] = 6.0;
            outer_rad[i] = 120.0;
        }

        cu_unroll(img_w, img_h, pitch, center_x, center_y, inner_rad, outer_rad, gm_norm, gm_norm_unr, gm_tmp);

        cu_gauss_filter_f11(CU_UNROLL_W, CU_UNROLL_H, gm_norm_unr, gm_norm_unr, gm_tmp_unr);

        float *norm_unr;
        norm_unr = malloc(CU_UNROLL_W * CU_UNROLL_H * 4);

        cudaMemcpy(norm_unr, gm_norm_unr, CU_UNROLL_W * CU_UNROLL_H * 4, cudaMemcpyDeviceToHost);

        {
            FILE *file = fopen("sobel_norm_unrolled.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++)
                fputc((int)norm_unr[p], file);
            fclose(file);
        }

        uint8_t *norm_unr_maxima;
        norm_unr_maxima = malloc(CU_UNROLL_W * CU_UNROLL_H);

        int x, y;
        for (x = 0; x < CU_UNROLL_W; x++) {
            float prev, act, next;
            act = norm_unr[x];
            next = norm_unr[CU_UNROLL_W + x];

            for (y = 1; y < CU_UNROLL_H - 1; y++) {
                prev = act;
                act = next;
                next = norm_unr[(y + 1) * CU_UNROLL_W + x];

                norm_unr_maxima[y * CU_UNROLL_W + x] = (act > 5.0 && act > prev && act > next) ? 255 : 0;
            }

            norm_unr_maxima[x] = 0;
            norm_unr_maxima[(CU_UNROLL_H - 1) * CU_UNROLL_W + x] = 0;
        }

        {
            FILE *file = fopen("sobel_norm_maxima.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++)
                fputc(norm_unr_maxima[p], file);
            fclose(file);
        }

        int *route_1, *route_2, *act_route;
        int cost_1, cost_2;

        route_1 = NULL;
        route_2 = NULL;

        act_route = malloc(CU_UNROLL_W * sizeof(int));

        for (y = 0; y < CU_UNROLL_H; y++) {
            int on_route = 1,
                cost = 0,
                act_y = y;

            for (x = 0; on_route && x < CU_UNROLL_W;) {
                int dx, next_y;
                on_route = 0;

                for (dx = 0; dx < 40 && x + dx < CU_UNROLL_W; dx++) {
                    int dh = dx / 4 + 1;

                    for (next_y = (act_y - dh > 0) ? act_y - dh : 0; next_y <= act_y + dh && next_y < CU_UNROLL_H; next_y++)
                        if (norm_unr_maxima[next_y * CU_UNROLL_W + x + dx]) {
                            int i;
                            for (i = 0; i < dx + 1; i++)
                                act_route[x + i] = (act_y * (dx + 1 - i) + next_y * i) / (dx + 1);

                            act_y = next_y;
                            cost += dx;
                            x += dx + 1;
                            on_route = 1;
                            break;
                        }

                    if (on_route)
                        break;

                    cost += 10;
                    dh = dx / 2 + 2;

                    for (next_y = (act_y - dh > 0) ? act_y - dh : 0; next_y <= act_y + dh && next_y < CU_UNROLL_H; next_y++)
                        if (norm_unr_maxima[next_y * CU_UNROLL_W + x + dx]) {
                            int i;
                            for (i = 0; i < dx + 1; i++)
                                act_route[x + i] = (act_y * (dx + 1 - i) + next_y * i) / (dx + 1);

                            act_y = next_y;
                            cost += dx;
                            x += dx + 1;
                            on_route = 1;
                            break;
                        }

                    if (on_route)
                        break;
                }
            }

            if (x + 40 >= CU_UNROLL_W && y == act_y) {
                for (; x < CU_UNROLL_W; x++)
                    act_route[x] = act_y;

                printf("trace at y=%d, cost: %d\n", y, cost);

                if (route_1 == NULL) {
                    route_1 = act_route;
                    cost_1 = cost;
                    act_route = malloc(CU_UNROLL_W * sizeof(int));
                }
                else if (route_2 == NULL) {
                    route_2 = act_route;
                    cost_2 = cost;
                    act_route = malloc(CU_UNROLL_W * sizeof(int));
                }
                else {
                    if (cost_1 > cost_2) {
                        if (cost < cost_1) {
                            route_1 = route_2;
                            cost_1 = cost_2;
                            route_2 = act_route;
                            cost_2 = cost;
                            act_route = malloc(CU_UNROLL_W * sizeof(int));
                        }
                    }
                    else if (cost < cost_2) {
                        route_2 = act_route;
                        cost_2 = cost;
                        act_route = malloc(CU_UNROLL_W * sizeof(int));
                    }
                }
            }
        }

        if (route_1 == NULL || route_2 == NULL) {
            fprintf(stderr, "error: no iris boundaries detected\n");
            return 2;
        }

        {
            FILE *file = fopen("sobel_norm_traced.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int x, y;
            for (y = 0; y < CU_UNROLL_H; y++)
                for (x = 0; x < CU_UNROLL_W; x++)
                    fputc((route_1[x] == y || route_2[x] == y) ? 255 : 0, file);
            fclose(file);
        }

        for (i = 0; i < 32; i++) {
            inner_rad[i] = route_1[i * CU_UNROLL_W / 32] * (120 - 6) / CU_UNROLL_H + 6;
            outer_rad[i] = route_2[i * CU_UNROLL_W / 32] * (120 - 6) / CU_UNROLL_H + 6;
        }
    }

    void *gm_iris;
    cudaMalloc(&gm_iris, CU_UNROLL_W * CU_UNROLL_H * 4);

    //! Aufrollen der Iris:
    {
        void *gm_img_pitch;
        size_t pitch;

        cudaMallocPitch(&gm_img_pitch, &pitch, img_w * 4, img_h);
        cudaMemcpy2D(gm_img_pitch, pitch, img_d, img_w * 4, img_w * 4, img_h, cudaMemcpyHostToDevice);

        cudaMemset(gm_tmp, 0, img_w * img_h);

        cu_unroll(img_w, img_h, pitch, center_x, center_y, inner_rad, outer_rad, gm_img_pitch, gm_iris, gm_tmp);

        uint8_t *iris_d = malloc(CU_UNROLL_W * CU_UNROLL_H * 4);
        cudaMemcpy(iris_d, gm_iris, CU_UNROLL_W * CU_UNROLL_H * 4, cudaMemcpyDeviceToHost);

        {
            FILE *file = fopen("iris.ppm", "w");
            fprintf(file, "P6\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++) {
                fputc(iris_d[p * 4], file);
                fputc(iris_d[p * 4 + 1], file);
                fputc(iris_d[p * 4 + 2], file);
            }
            fclose(file);
        }

        {
            uint8_t *gray_d = malloc(img_w * img_h);
            cudaMemcpy(gray_d, gm_tmp, img_w * img_h, cudaMemcpyDeviceToHost);

            int i;
            for (i = 0; i < 32; i++) {
                float phi = (float)i * 2.0 * PI / 32.0;
                gray_d[(center_y + (int)(inner_rad[i] * sin(phi))) * img_w + center_x + (int)(inner_rad[i] * cos(phi))] = 255;
                gray_d[(center_y + (int)(outer_rad[i] * sin(phi))) * img_w + center_x + (int)(outer_rad[i] * cos(phi))] = 255;
            }

            FILE *file = fopen("unroll_cut.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
            int p;
            for (p = 0; p < img_w * img_h; p++)
                fputc(gray_d[p], file);
            fclose(file);
        }
    }

    void *gm_iris_gray, *gm_iris_tmp;
    cudaMalloc(&gm_iris_gray, CU_UNROLL_W * CU_UNROLL_H);
    cudaMalloc(&gm_iris_tmp, CU_UNROLL_W * CU_UNROLL_H);

    cu_color_to_gray(CU_UNROLL_W, CU_UNROLL_H, gm_iris, gm_iris_gray);

    //! Histogram equalization der Iris
    {
        int *histo = malloc(sizeof(int) * 256);

        cu_histogram(CU_UNROLL_W, CU_UNROLL_H, gm_iris_gray, gm_iris_tmp, histo);

        {
            FILE *file = fopen("iris_histo.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", 256, 256);
            int x, y;
            for (y = 0; y < 256; y++)
                for (x = 0; x < 256; x++)
                    fputc((histo[x] / 8 > (256 - y)) ? 255 : 0, file);
            fclose(file);
        }

        int tmp[256];
        uint8_t sub[256];
        int cdf, i;
        cdf = 0;
        for (i = 0; i < 256; i++)
            tmp[i] = cdf += histo[i];
        //printf("sub:\n");
        for (i = 0; i < 256; i++) {
            sub[i] = tmp[i] * 255 / cdf;
            //printf("%d ", sub[i]);
        }
        //printf("\n\n");
        cu_pixel_substitute(CU_UNROLL_W, CU_UNROLL_H, gm_iris_gray, gm_iris_tmp, sub);

        {
            uint8_t *gray_d = malloc(CU_UNROLL_W * CU_UNROLL_H);
            cudaMemcpy(gray_d, gm_iris_tmp, CU_UNROLL_W * CU_UNROLL_H, cudaMemcpyDeviceToHost);

            FILE *file = fopen("iris_equal.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++)
                fputc(gray_d[p], file);
            fclose(file);
        }
    }

    return 0;
}
