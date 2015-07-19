/*!
 * \file iris.c
 * Reference implementation for hough transform in C
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <gdk-pixbuf/gdk-pixbuf.h>

uint8_t *get_gaussian_blur(int width, int height, uint8_t *img)
{
    uint8_t *gauss;
    int x, y;

    gauss = malloc(width * height);

    for (x = 0; x < width; x++) {
        int xll = (x < 2) ? 0 : x - 2,
            xl = (x < 1) ? 0 : x - 1,
            xr = (x >= width - 1) ? width - 1 : x + 1,
            xrr = (x >= width - 2) ? width - 1 : x + 2;

        uint8_t ttll, ttl, ttm, ttr, ttrr,
                tll = img[xll], tl = img[xl], tm = img[x], tr = img[xr], trr = img[xrr],
                mll = img[xll], ml = img[xl], mm = img[x], mr = img[xr], mrr = img[xrr],
                bll = img[xll], bl = img[xl], bm = img[x], br = img[xr], brr = img[xrr],
                bbll = img[xll + width], bbl = img[xl + width], bbm = img[x + width], bbr = img[xr + width], bbrr = img[xrr + width];

        for (y = 0; y < height; y++) {
            ttll = tll; ttl = tl; ttm = tm; ttr = tr; ttrr = trr;
            tll = mll; tl = ml; tm = mm; tr = mr; trr = mrr;
            mll = bll; ml = bl; mm = bm; mr = br; mrr = brr;
            bll = bbll; bl = bbl; bm = bbm; br = bbr; brr = bbrr;

            int row = (y >= height - 2) ? height - 1 : y + 2;

            bbll = img[width * row + xll]; bbl = img[width * row + xl]; bbm = img[width * row + x]; bbr = img[width * row + xr]; bbrr = img[width * row + xrr];

            int val = ttll * 1 + ttl * 4 + ttm * 7 + ttr * 4 + ttrr * 1
                    + tll  * 4 + tl * 16 + tm * 26 + tr * 16 + trr  * 4
                    + mll  * 7 + ml * 26 + mm * 41 + mr * 26 + mrr  * 7
                    + bll  * 4 + bl * 16 + bm * 26 + br * 16 + brr  * 4
                    + bbll * 1 + bbl * 4 + bbm * 7 + bbr * 4 + bbrr * 1;

            gauss[y * width + x] = val / 273;
        }
    }

    return gauss;
}

int16_t *get_sobel(int width, int height, uint8_t *img)
{
    int16_t *sobel;
    int x, y;

    sobel = malloc(width * height * 2 * sizeof(int16_t));

    for (x = 0; x < width; x++) {
        sobel[x * 2] = 0;
        sobel[x * 2 + 1] = 0;
        sobel[(width * (height - 1) + x) * 2] = 0;
        sobel[(width * (height - 1) + x) * 2 + 1] = 0;
    }

    for (y = 0; y < height; y++) {
        sobel[(width * y) * 2] = 0;
        sobel[(width * y) * 2 + 1] = 0;
        sobel[(width * y + width - 1) * 2] = 0;
        sobel[(width * y + width - 1) * 2 + 1] = 0;
    }

    for (x = 1; x < width - 1; x++) {
        uint8_t tl, tm, tr,
                ml = img[x - 1], mm = img[x], mr = img[x + 1],
                bl = img[width + x - 1], bm = img[width + x], br = img[width + x + 1];

        for (y = 1; y < width - 1; y++) {
            tl = ml; tm = mm; tr = mr;
            ml = bl; mm = bm; mr = br;

            bl = img[width * (y + 1) + x - 1];
            bm = img[width * (y + 1) + x];
            br = img[width * (y + 1) + x + 1];

            int hori = (int)tl + 2 * (int)ml + (int)bl - (int)tr - 2 * (int)mr - (int)br,
                vert = (int)tl + 2 * (int)tm + (int)tr - (int)bl - 2 * (int)bm - (int)br;

            sobel[(width * y + x) * 2] = hori;
            sobel[(width * y + x) * 2 + 1] = vert;
        }
    }

    return sobel;
}

#define MARGIN 20
#define MIN_RAD 6
#define MAX_RAD 100
#define PI 3.141592654

inline static int fill_accumulator(int16_t *sob_d, int sob_w, int sob_h, int x, int y, int rad)
{
    int p_dx, p_dy, rad_acc = 0;
    double phi;

    for (phi = 0.0; phi < 2 * PI; phi += PI / 120.0) {
        p_dx = (double)rad * cos(phi);
        p_dy = (double)rad * sin(phi);

        if (x + p_dx >= 0 && x + p_dx < sob_w && y + p_dy >= 0 && y + p_dy < sob_h) {
            int hori = sob_d[((y + p_dy) * sob_w + x + p_dx) * 2];
            int vert = sob_d[((y + p_dy) * sob_w + x + p_dx) * 2 + 1];

            if (((hori < 0) ? -hori : hori) > 20 || ((vert < 0) ? -vert : vert) > 20) {
                double grad = PI + atan2(vert, hori) - phi;

                if (-PI / 12.0 < grad && grad < PI / 12.0)
                    rad_acc++;
            }
        }
    }

    return rad_acc;
}

static int *get_hough(int16_t *sob_d, int sob_w, int sob_h)
{
    int x, y, rad, *pp;

    int *acc_d = malloc(sob_w * sob_h * sizeof(int));
    memset(acc_d, 0, sob_w * sob_h * sizeof(int));

    for (x = MARGIN; x < sob_w - MARGIN; x++)
        for (y = MARGIN; y < sob_h - MARGIN; y++) {
            pp = &acc_d[y * sob_w + x];
            *pp = 0;

            for (rad = MIN_RAD; rad < MAX_RAD; rad += 2) {
                int rad_acc = fill_accumulator(sob_d, sob_w, sob_h, x, y, rad);
                (*pp) += rad_acc * rad_acc * rad_acc;
            }
        }

    return acc_d;
}

static int *get_hough_radius(int16_t *sob_d, int sob_w, int sob_h, int center_x, int center_y)
{
    int *radius = malloc(MAX_RAD * sizeof(int));
    memset(radius, 0, MAX_RAD * sizeof(int));

    int rad;
    for (rad = MIN_RAD; rad < MAX_RAD; rad += 1)
        radius[rad] = fill_accumulator(sob_d, sob_w, sob_h, center_x, center_y, rad);

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

    uint8_t *gray_d;
    gray_d = malloc(img_w * img_h);

    int p;
    for (p = 0; p < img_w * img_h; p++)
        gray_d[p] = (img_d[4 * p] + img_d[4 * p + 1] + img_d[4 * p + 2]) / 3;

    uint8_t *gauss_d;
    gauss_d = get_gaussian_blur(img_w, img_h, gray_d);

    int16_t *sobel_d;
    sobel_d = get_sobel(img_w, img_h, gauss_d);

    {
        FILE *file = fopen("sobel.ppm", "w");
        fprintf(file, "P6\n%d %d\n255\n", img_w, img_h);
        int p;
        for (p = 0; p < img_w * img_h; p++) {
            fputc((sobel_d[p * 2] >> 1) + 128, file);
            fputc(0, file);
            fputc((sobel_d[p * 2 + 1] >> 1) + 128, file);
        }
        fclose(file);
    }

    int *hough_d;
    hough_d = get_hough(sobel_d, img_w, img_h);

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

    int center_x = 0,
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
        center_x /= c;
        center_y /= c;
    }

    printf("center: (%d, %d)\n", center_x, center_y);

    int *hough_radius;
    hough_radius = get_hough_radius(sobel_d, img_w, img_h, center_x, center_y);

    {
        FILE *file = fopen("radius.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", MAX_RAD, 200);
        int x, y;
        for (y = 0; y < 200; y++)
            for (x = 0; x < MAX_RAD; x++)
                fputc((hough_radius[x] > (200 - y)) ? 255 : 0, file);
        fclose(file);
    }

    int inner_radius = 0,
        outer_radius = 0;

    {
        int rad;
        for (rad = MIN_RAD; rad < MAX_RAD; rad++)
            if (hough_radius[rad] > hough_radius[outer_radius])
                outer_radius = rad;

        for (rad = MIN_RAD; rad < outer_radius * 3 / 4; rad++)
            if (hough_radius[rad] > hough_radius[inner_radius])
                inner_radius = rad;
    }

    printf("inner radius: %d, outer radius: %d\n", inner_radius, outer_radius);

    return 0;
}
