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
#define MIN_RAD 6
#define MAX_RAD 100

inline static int fill_accumulator(int16_t *sob_h, int16_t *sob_v, int img_w, int img_h, int x, int y, int rad)
{
    int p_dx, p_dy, rad_acc = 0;
    double phi;

    for (phi = -PI; phi < PI; phi += PI / 120.0) {
        p_dx = (double)rad * cos(phi);
        p_dy = (double)rad * sin(phi);

        if (x + p_dx >= 0 && x + p_dx < img_w && y + p_dy >= 0 && y + p_dy < img_h) {
            int hori = sob_h[(y + p_dy) * img_w + x + p_dx];
            int vert = sob_v[(y + p_dy) * img_w + x + p_dx];

            if (((hori < 0) ? -hori : hori) > 20 || ((vert < 0) ? -vert : vert) > 20) {
                double grad = atan2(vert, hori) - phi;

                if (-PI / 12.0 < grad && grad < PI / 12.0)
                    rad_acc++;
            }
        }
    }

    return rad_acc;
}

static int *get_hough_radius(int16_t *sob_h, int16_t *sob_v, int img_w, int img_h, int center_x, int center_y)
{
    int *radius = malloc(MAX_RAD * sizeof(int));
    memset(radius, 0, MAX_RAD * sizeof(int));

    int rad;
    for (rad = MIN_RAD; rad < MAX_RAD; rad += 1)
        radius[rad] = fill_accumulator(sob_h, sob_v, img_w, img_h, center_x, center_y, rad);

    return radius;
}

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

    int *histo = malloc(sizeof(int) * 256);
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

    //! Histogram equalization:
    {
        cu_histogram(img_w, img_h, gm_img, gm_tmp, histo);

        {
            FILE *file = fopen("histo.pgm", "w");
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
        cu_pixel_substitute(img_w, img_h, gm_img, gm_tmp, sub);

        {
            uint8_t *gray_d = malloc(img_w * img_h);
            cudaMemcpy(gray_d, gm_tmp, img_w * img_h, cudaMemcpyDeviceToHost);

            FILE *file = fopen("equal.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
            int p;
            for (p = 0; p < img_w * img_h; p++)
                fputc(gray_d[p], file);
            fclose(file);
        }
    }

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

    int16_t *sobel_h, *sobel_v;
    sobel_h = malloc(img_w * img_h * 2);
    sobel_v = malloc(img_w * img_h * 2);

    cudaMemcpy(sobel_h, gm_sobel_h, img_w * img_h * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(sobel_v, gm_sobel_v, img_w * img_h * 2, cudaMemcpyDeviceToHost);

    {
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

    int *hough_d;
    hough_d = malloc(img_w * img_h * 4);

    cudaMemcpy(hough_d, gm_tmp, img_w * img_h * 4, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy error: %s\n", cudaGetErrorString(cudaGetLastError()));

    {
        FILE *file = fopen("hough.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
        int p, v;
        for (p = 0; p < img_w * img_h; p++) {
            v = hough_d[p] >> 16;
            fputc((v > 255) ? 255 : v, file);
        }
        fclose(file);
    }

    int center_x, center_y, inner_rad, outer_rad;

    //! Berechnung von Zentrum, innerem und aeusserem Radius:
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

    //! Aufrollen der Iris:
    {
        void *gm_img_pitch, *gm_iris;
        size_t pitch;

        cudaMallocPitch(&gm_img_pitch, &pitch, img_w * 4, img_h);
        cudaMemcpy2D(gm_img_pitch, pitch, img_d, img_w * 4, img_w * 4, img_h, cudaMemcpyHostToDevice);

        cudaMalloc(&gm_iris, CU_UNROLL_W * CU_UNROLL_H * 4);

        cu_unroll(img_w, img_h, pitch, center_x, center_y, inner_rad, outer_rad, gm_img_pitch, gm_iris);

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
    }

    return 0;
}
