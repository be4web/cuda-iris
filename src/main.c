#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#include <math.h>

#include "pixel.h"
#include "convolve.h"
#include "reduction.h"

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
    cu_histogram(img_w, img_h, gm_img, gm_tmp, histo);

    uint8_t *gray_d = malloc(img_w * img_h);

    {
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

        cudaMemcpy(gray_d, gm_tmp, img_w * img_h, cudaMemcpyDeviceToHost);

        {
            FILE *file = fopen("equal.pgm", "w");
            fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
            int p;
            for (p = 0; p < img_w * img_h; p++)
                fputc(gray_d[p], file);
            fclose(file);
        }
    }

    if (cu_gauss_filter(11, img_w, img_h, gm_img, gm_img, gm_tmp) < 0)
        fprintf(stderr, "error applying gauss filter\n");

    cu_sobel_filter(img_w, img_h, gm_img, gm_sobel_h, gm_sobel_v, gm_tmp);

    int16_t *sobel_h, *sobel_v;
    sobel_h = malloc(img_w * img_h * 2);
    sobel_v = malloc(img_w * img_h * 2);

    cudaMemcpy(gray_d, gm_img, img_w * img_h, cudaMemcpyDeviceToHost);
    cudaMemcpy(sobel_h, gm_sobel_h, img_w * img_h * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(sobel_v, gm_sobel_v, img_w * img_h * 2, cudaMemcpyDeviceToHost);

    //! Transformation der Gradienten von kartesisch nach polar:
    // - auf der Grafikkarte (funktioniert nicht):
    /*
    {
        cu_cart_to_polar(img_w, img_h, gm_sobel_h, gm_sobel_v, gm_sobel_1, gm_sobel_2);

        cudaMemcpy2D(gm_sobel_abs, sobel_pitch, gm_sobel_1, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(gm_sobel_phi, sobel_pitch, gm_sobel_2, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyDeviceToDevice);
    }
    */

    // - auf der CPU (funktioniert):
    {
        float *abs, *phi;
        int x, y;

        abs = malloc(img_w * img_h * sizeof(float));
        phi = malloc(img_w * img_h * sizeof(float));

        for (y = 0; y < img_h; y++)
            for (x = 0; x < img_w; x++) {
                int hori = sobel_h[img_w * y + x],
                    vert = sobel_v[img_w * y + x];

                phi[img_w * y + x] = atan2(vert, hori);
                abs[img_w * y + x] = sqrt(hori* hori + vert * vert);
            }

        cudaMemcpy2D(gm_sobel_phi, sobel_pitch, phi, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyHostToDevice);
        cudaMemcpy2D(gm_sobel_abs, sobel_pitch, abs, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyHostToDevice);
    }

    cu_hough(img_w, img_h, sobel_pitch, gm_sobel_abs, gm_sobel_phi, gm_tmp);

    int *hough_d;
    hough_d = malloc(img_w * img_h * 4);

    cudaMemcpy(hough_d, gm_tmp, img_w * img_h * 4, cudaMemcpyDeviceToHost);
    printf("cudaMemcpy error: %s\n", cudaGetErrorString(cudaGetLastError()));

    {
        FILE *file = fopen("histo.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", 256, 256);
        int x, y;
        for (y = 0; y < 256; y++)
            for (x = 0; x < 256; x++)
                fputc((histo[x] / 8 > (256 - y)) ? 255 : 0, file);
        fclose(file);
    }

    {
        FILE *file = fopen("gauss.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
        int p;
        for (p = 0; p < img_w * img_h; p++)
            fputc(gray_d[p], file);
        fclose(file);
    }

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

    return 0;
}
