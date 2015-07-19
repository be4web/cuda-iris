#include "iris.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <cuda_runtime.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#include <math.h>

#include "pixel.h"
#include "convolve.h"
#include "histogram.h"
#include "hough.h"
#include "unroll.h"
#include "wavelet.h"

#define PI 3.141592654
#define RESIZED_IMAGE_WIDTH 256

//#define DEBUG

int get_iris_features(const char *path, float *mad)
{
    //! Reading image file:
    GdkPixbuf *img;
    GError *err = NULL;

    if ((img = gdk_pixbuf_new_from_file(path, &err)) == NULL) {
        fprintf(stderr, "error loading image file `%s': %s\n", path, err->message);
        return -1;
    }

    int img_w = gdk_pixbuf_get_width(img),
        img_h = gdk_pixbuf_get_height(img),
        img_s = gdk_pixbuf_get_rowstride(img),
        img_c = gdk_pixbuf_get_n_channels(img),
        img_b = gdk_pixbuf_get_bits_per_sample(img);

#ifdef DEBUG
    fprintf(stderr, "image prop: w: %d, h: %d, s: %d, c: %d, b %d\n", img_w, img_h, img_s, img_c, img_b);
#endif

    uint8_t *img_d = gdk_pixbuf_get_pixels(img);

    void *gm_color, *gm_gray, *gm_resized, *gm_sobel_h, *gm_sobel_v, *gm_sobel_abs, *gm_sobel_phi, *gm_tmp;
    int pitch32;

    int resized_w = RESIZED_IMAGE_WIDTH,
        resized_h = (resized_w * img_h)/img_w,
        resized_p;

    {
        struct cudaDeviceProp prop;
        int tex_align;
        cudaGetDeviceProperties(&prop, 0);
        tex_align = prop.textureAlignment - 1;

        pitch32 = ((img_w * 4 + tex_align) & ~tex_align) >> 2;
#ifdef DEBUG
        fprintf(stderr, "texture alignment: %d, 32-bit width: %d => pitch: %d, per sample: %d\n", tex_align + 1, img_w * 4, pitch32 * 4, pitch32);
#endif

        resized_p = ((resized_w * 4 + tex_align) & ~tex_align) >> 2;
    }

    cudaMalloc(&gm_color, pitch32 * img_h * 4);
    cudaMalloc(&gm_gray, img_w * img_h);
    cudaMalloc(&gm_resized, resized_w * resized_h);
    cudaMalloc(&gm_sobel_h, resized_w * resized_h * 4);
    cudaMalloc(&gm_sobel_v, resized_w * resized_h * 4);
    cudaMalloc(&gm_sobel_abs, pitch32 * resized_w * 4);
    cudaMalloc(&gm_sobel_phi, pitch32 * resized_h * 4);
    cudaMalloc(&gm_tmp, pitch32 * resized_h * 4);

    int h;
    for (h = 0; h < img_h; h++)
        cudaMemcpy2D((uint8_t *)gm_color + pitch32 * 4 * h, 4, img_d + img_s * h, img_c, img_c, img_w, cudaMemcpyHostToDevice);

    //! Conversion to gray-scale image and resizing:
    cu_color_to_gray(img_w, img_h, pitch32, gm_color, img_w, gm_gray, 17, 2, 1);
    cu_image_resize (img_w, img_h, img_w, (uint8_t *) gm_gray, resized_w, resized_h, resized_w, gm_resized);

#ifdef DEBUG
    {
        uint8_t *gray_d = malloc(resized_w * resized_h);
        cudaMemcpy(gray_d, gm_resized, resized_w * resized_h, cudaMemcpyDeviceToHost);

        FILE *file = fopen("resized.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", resized_w, resized_h);
        int p;
        for (p = 0; p < resized_w * resized_h; p++)
            fputc(gray_d[p], file);
        fclose(file);

        free(gray_d);
    }
#endif

    //! Gaussian blur filter:
    if (cu_gauss_filter(11, resized_w, resized_h, gm_resized, gm_resized, gm_tmp) < 0)
        fprintf(stderr, "error applying gauss filter\n");

#ifdef DEBUG
    {
        uint8_t *gray_d = malloc(resized_w * resized_h);
        cudaMemcpy(gray_d, gm_resized, resized_w * resized_h, cudaMemcpyDeviceToHost);

        FILE *file = fopen("gauss.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", resized_w, resized_h);
        int p;
        for (p = 0; p < resized_w * resized_h; p++)
            fputc(gray_d[p], file);
        fclose(file);

        free(gray_d);
    }
#endif

    //! Sobel filter:
    cu_sobel_filter(resized_w, resized_h, gm_resized, gm_sobel_h, gm_sobel_v, gm_tmp);

#ifdef DEBUG
    {
        int16_t *sobel_h, *sobel_v;
        sobel_h = malloc(resized_w * resized_h * 2);
        sobel_v = malloc(resized_w * resized_h * 2);
        cudaMemcpy(sobel_h, gm_sobel_h, resized_w * resized_h * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(sobel_v, gm_sobel_v, resized_w * resized_h * 2, cudaMemcpyDeviceToHost);

        FILE *file = fopen("sobel.ppm", "w");
        fprintf(file, "P6\n%d %d\n255\n", resized_w, resized_h);
        int p;
        for (p = 0; p < resized_w * resized_h; p++) {
            fputc((sobel_h[p] >> 1) + 128, file);
            fputc(0, file);
            fputc((sobel_v[p] >> 1) + 128, file);
        }
        fclose(file);

        free(sobel_h);
        free(sobel_v);
    }
#endif

    //! Transform of gradient data (Sobel) from cartesian values to polar values:
    cu_cart_to_polar(resized_w, resized_h, resized_w, gm_sobel_h, resized_w, gm_sobel_v, resized_p, gm_sobel_abs, resized_p, gm_sobel_phi);

    //! Hough transform:
    cu_hough(resized_w, resized_h, resized_p, gm_sobel_abs, resized_p, gm_sobel_phi, resized_w, gm_tmp, 6.0, resized_h / 2);

#ifdef DEBUG
    {
        int *hough_d = malloc(img_w * img_h * 4);
        cudaMemcpy(hough_d, gm_tmp, resized_w * resized_h * 4, cudaMemcpyDeviceToHost);

        FILE *file = fopen("hough.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n",resized_w, resized_h);
        int p, v;
        for (p = 0; p < resized_w * resized_h; p++) {
            v = hough_d[p] >> 16;
            fputc((v > 255) ? 255 : v, file);
        }
        fclose(file);

        free(hough_d);
    }
#endif

    int center_x, center_y;

    //! Calculation of center of iris:
    cu_center_detection(resized_w, resized_h, gm_tmp, &center_x, &center_y);

#ifdef DEBUG
    fprintf(stderr, "center: (%d, %d)\n", center_x, center_y);
#endif

    float inner_rad[32], outer_rad[32];

    //! Calculation of inner and outer iris boundaries:
    {
        float min_rad, max_rad;
        max_rad = resized_h / 2;
        min_rad = 6.0;

        void *gm_norm, *gm_norm_unr, *gm_tmp_unr;
        cudaMalloc(&gm_norm, resized_p * resized_h * 4);
        cudaMalloc(&gm_norm_unr, CU_UNROLL_W * CU_UNROLL_H * 4);
        cudaMalloc(&gm_tmp_unr, CU_UNROLL_W * CU_UNROLL_H * 4);

        cu_centered_gradient_normalization(resized_w, resized_h, resized_p, gm_sobel_abs, resized_p, gm_sobel_phi, resized_p, gm_norm, center_x, center_y);

#ifdef DEBUG
        {
            float *norm = malloc(resized_p * resized_h * 4);
            cudaMemcpy(norm, gm_norm, resized_p * resized_h * 4, cudaMemcpyDeviceToHost);

            FILE *file = fopen("sobel_norm.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", resized_w, resized_h);
            int x, y;
            for (y = 0; y < resized_h; y++)
                for (x = 0; x < resized_w; x++)
                    fputc((int)norm[y * resized_p + x], file);
            fclose(file);

            free(norm);
        }
#endif

        int i;
        for (i = 0; i < 32; i++) {
            inner_rad[i] = min_rad;
            outer_rad[i] = max_rad;
        }

        cu_unroll(resized_w, resized_h, resized_p, gm_norm, gm_norm_unr, center_x, center_y, inner_rad, outer_rad, gm_tmp);

        cu_gauss_filter_f11(CU_UNROLL_W, CU_UNROLL_H, gm_norm_unr, gm_norm_unr, gm_tmp_unr);

        float *norm_unr;
        norm_unr = malloc(CU_UNROLL_W * CU_UNROLL_H * 4);

        cudaMemcpy(norm_unr, gm_norm_unr, CU_UNROLL_W * CU_UNROLL_H * 4, cudaMemcpyDeviceToHost);

#ifdef DEBUG
        {
            FILE *file = fopen("sobel_norm_unrolled.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++)
                fputc((int)norm_unr[p], file);
            fclose(file);
        }
#endif

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

#ifdef DEBUG
        {
            FILE *file = fopen("sobel_norm_maxima.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++)
                fputc(norm_unr_maxima[p], file);
            fclose(file);
        }
#endif

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

                for (dx = 0; dx < 60 && x + dx < CU_UNROLL_W; dx++) {
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

            if (x + 60 >= CU_UNROLL_W && y == act_y) {
                for (; x < CU_UNROLL_W; x++)
                    act_route[x] = act_y;

#ifdef DEBUG
                fprintf(stderr, "trace at y=%d, cost: %d\n", y, cost);
#endif

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
                            free(route_1);
                            route_1 = route_2;
                            cost_1 = cost_2;
                            route_2 = act_route;
                            cost_2 = cost;
                            act_route = malloc(CU_UNROLL_W * sizeof(int));
                        }
                    }
                    else if (cost < cost_2) {
                        free(route_2);
                        route_2 = act_route;
                        cost_2 = cost;
                        act_route = malloc(CU_UNROLL_W * sizeof(int));
                    }
                }
            }
        }

        free(act_route);

        if (route_1 == NULL || route_2 == NULL) {
            fprintf(stderr, "error: no iris boundaries detected\n");
            return -1;
        }

#ifdef DEBUG
        {
            FILE *file = fopen("sobel_norm_traced.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int x, y;
            for (y = 0; y < CU_UNROLL_H; y++)
                for (x = 0; x < CU_UNROLL_W; x++)
                    fputc((route_1[x] == y || route_2[x] == y) ? 255 : 0, file);
            fclose(file);
        }
#endif

        for (i = 0; i < 32; i++) {
            inner_rad[i] = route_1[i * CU_UNROLL_W / 32] * (max_rad - min_rad) / CU_UNROLL_H + 6;
            outer_rad[i] = route_2[i * CU_UNROLL_W / 32] * (max_rad - min_rad) / CU_UNROLL_H + 6;
        }

        free(route_1);
        free(route_2);

        free(norm_unr);
        free(norm_unr_maxima);

        cudaFree(gm_norm);
        cudaFree(gm_norm_unr);
        cudaFree(gm_tmp_unr);
    }

    center_x = ((float)img_w / resized_w) * center_x;
    center_y = ((float)img_h / resized_h) * center_y;

    int i;
    for (i = 0; i < 32; i++) {
        inner_rad[i] = ((float)img_w / resized_w) * inner_rad[i];
        outer_rad[i] = ((float)img_w / resized_w) * outer_rad[i];
    }

    void *gm_iris, *gm_iris_gray, *gm_iris_equ, *gm_iris_wave, *gm_iris_tmp;
    cudaMalloc(&gm_iris, CU_UNROLL_W * CU_UNROLL_H * 4);
    cudaMalloc(&gm_iris_gray, CU_UNROLL_W * CU_UNROLL_H);
    cudaMalloc(&gm_iris_equ, CU_UNROLL_W * CU_UNROLL_H);
    cudaMalloc(&gm_iris_wave, CU_UNROLL_W * CU_UNROLL_H * sizeof(float));
    cudaMalloc(&gm_iris_tmp, CU_UNROLL_W * CU_UNROLL_H * sizeof(float));

    //! Iris unrolling:
    {
        cudaMemset(gm_gray, 0, img_w * img_h);

        cu_unroll(img_w, img_h, pitch32, gm_color, gm_iris, center_x, center_y, inner_rad, outer_rad, gm_gray);

#ifdef DEBUG
        {
            uint8_t *iris_d = malloc(CU_UNROLL_W * CU_UNROLL_H * 4);
            cudaMemcpy(iris_d, gm_iris, CU_UNROLL_W * CU_UNROLL_H * 4, cudaMemcpyDeviceToHost);

            FILE *file = fopen("iris.ppm", "w");
            fprintf(file, "P6\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++) {
                fputc(iris_d[p * 4], file);
                fputc(iris_d[p * 4 + 1], file);
                fputc(iris_d[p * 4 + 2], file);
            }
            fclose(file);

            free(iris_d);
        }
#endif

#ifdef DEBUG
        {
            uint8_t *gray_d = malloc(img_w * img_h);
            cudaMemcpy(gray_d, gm_gray, img_w * img_h, cudaMemcpyDeviceToHost);

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

            free(gray_d);
        }
#endif
    }

    cu_color_to_gray(CU_UNROLL_W, CU_UNROLL_H, CU_UNROLL_W, gm_iris, CU_UNROLL_W, gm_iris_gray, 1, 4, 2);

    //! Histogram equalization
    {
        int *histo = malloc(sizeof(int) * 256);

        cu_histogram(CU_UNROLL_W, CU_UNROLL_H, CU_UNROLL_W, gm_iris_gray, gm_iris_tmp, histo);

#ifdef DEBUG
        {
            FILE *file = fopen("iris_histo.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", 256, 256);
            int x, y;
            for (y = 0; y < 256; y++)
                for (x = 0; x < 256; x++)
                    fputc((histo[x] / 8 > (256 - y)) ? 255 : 0, file);
            fclose(file);
        }
#endif

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
        cu_pixel_substitute(CU_UNROLL_W, CU_UNROLL_H, CU_UNROLL_W, gm_iris_gray, CU_UNROLL_W, gm_iris_equ, sub);

#ifdef DEBUG
        {
            uint8_t *gray_d = malloc(CU_UNROLL_W * CU_UNROLL_H);
            cudaMemcpy(gray_d, gm_iris_equ, CU_UNROLL_W * CU_UNROLL_H, cudaMemcpyDeviceToHost);

            FILE *file = fopen("iris_equal.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
            int p;
            for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++)
                fputc(gray_d[p], file);
            fclose(file);

            free(gray_d);
        }
#endif
    }

    //! Log Gabor filter
    {
        //float mad[32]; // feature vector (mean absolute deviation)
        float norm = 0.0;

        int f, o;
        for (f = 0; f < 16; f++)
            for (o = 0; o < 2; o++) {
                if (o == 0) {
                    cu_convolve_row_f65(CU_UNROLL_W, CU_UNROLL_H, gm_iris_equ, gm_iris_tmp, (const float *)log_gabor_1d[f], log_gabor_div[f]);
                    cu_convolve_col_f65(CU_UNROLL_W, CU_UNROLL_H, gm_iris_tmp, gm_iris_wave, (const float *)gauss65, 1.0);
                } else {
                    cu_convolve_row_f65(CU_UNROLL_W, CU_UNROLL_H, gm_iris_equ, gm_iris_tmp, (const float *)gauss65, 1.0);
                    cu_convolve_col_f65(CU_UNROLL_W, CU_UNROLL_H, gm_iris_tmp, gm_iris_wave, (const float *)log_gabor_1d[f], log_gabor_div[f]);
                }

#ifdef DEBUG
                {
                    float *wave = malloc(CU_UNROLL_W * CU_UNROLL_H * sizeof(float));
                    cudaMemcpy(wave, gm_iris_wave, CU_UNROLL_W * CU_UNROLL_H * sizeof(float), cudaMemcpyDeviceToHost);
                    char path[256];
                    snprintf(path, sizeof(path), "iris_wave_%03d_%d.ppm", f, o);
                    FILE *file = fopen(path, "w");
                    fprintf(file, "P5\n%d %d\n255\n", CU_UNROLL_W, CU_UNROLL_H);
                    int p;
                    for (p = 0; p < CU_UNROLL_W * CU_UNROLL_H; p++)
                        fputc((int)(wave[p] * 0.5) + 128, file);
                    fclose(file);
                    free(wave);
                }
#endif

                float act_mad;
                mad[f * 2 + o] = act_mad = cu_mad(CU_UNROLL_W, CU_UNROLL_H, CU_UNROLL_W, gm_iris_wave, gm_iris_tmp);
                norm += act_mad * act_mad;
            }

        norm = sqrt(norm);

        for (f = 0; f < 32; f++)
            mad[f] /= norm;
    }

    cudaFree(gm_iris);
    cudaFree(gm_iris_gray);
    cudaFree(gm_iris_equ);
    cudaFree(gm_iris_wave);
    cudaFree(gm_iris_tmp);

    cudaFree(&gm_color);
    cudaFree(&gm_gray);
    cudaFree(&gm_sobel_h);
    cudaFree(&gm_sobel_v);
    cudaFree(&gm_sobel_abs);
    cudaFree(&gm_sobel_phi);
    cudaFree(&gm_tmp);

    return 0;
}
