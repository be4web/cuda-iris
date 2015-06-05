#include "gabor.h"

#include <math.h>
#include <string.h>

#define PI 3.141592654

static int img_w, img_h, img_p;
static uint8_t *img_d;

static void gabor(float r0, float theta0, float omega, float alpha, float beta, float rho, float phi, float *re, float *im)
{
    int x = (int)(phi * (float)( (img_w - 1) / (2.0 * PI) ) );
    int y = (int)(rho * (float)(img_h - 1));

    float rho_diff = r0 - rho;
    float phi_diff = theta0 - phi;

    float alpha2 = alpha * alpha;
    float beta2 = beta * beta;

    if (x < 0)
        x += img_w;

    if (x >= img_w)
        x -= img_w;

    if (y < 0)
        y = 0;

    if (y >= img_h)
        y = img_h - 1;

    float common = img_d[img_p * y + x] * exp(-(rho_diff * rho_diff) / alpha2) * exp(-(phi_diff * phi_diff) / beta2) * rho;

    float exp_im = -omega * phi_diff;

    *re = cos(exp_im) * common;
    *im = sin(exp_im) * common;
}

static int gabor_integrate(float r0, float theta0, float omega, float alpha, float beta, float ra, float rb, float pa, float pb, int n_rho, int n_phi, float *gabor_re, float *gabor_im)
{
    float step_rho, step_phi, sum_re, sum_im, re, im;
    int r, p;

    step_rho = (rb - ra) / (float)n_rho;
    step_phi = (pb - pa) / (float)n_phi;

    sum_re = 0.0;
    sum_im = 0.0;

    gabor(r0, theta0, omega, alpha, beta, ra, pa, &re, &im);
    sum_re += re;
    sum_im += im;

    gabor(r0, theta0, omega, alpha, beta, ra, pb, &re, &im);
    sum_re += re;
    sum_im += im;

    gabor(r0, theta0, omega, alpha, beta, rb, pa, &re, &im);
    sum_re += re;
    sum_im += im;

    gabor(r0, theta0, omega, alpha, beta, rb, pb, &re, &im);
    sum_re += re;
    sum_im += im;

    sum_re *= 0.5;
    sum_im *= 0.5;

    for (r = 1; r < n_rho; r++) {
        float rho = ra + step_rho * (float)r;

        gabor(r0, theta0, omega, alpha, beta, rho, pa, &re, &im);
        sum_re += re;
        sum_im += im;

        gabor(r0, theta0, omega, alpha, beta, rho, pb, &re, &im);
        sum_re += re;
        sum_im += im;
    }

    for (p = 1; p < n_rho; p++) {
        float phi = pa + step_phi * (float)p;

        gabor(r0, theta0, omega, alpha, beta, ra, phi, &re, &im);
        sum_re += re;
        sum_im += im;

        gabor(r0, theta0, omega, alpha, beta, rb, phi, &re, &im);
        sum_re += re;
        sum_im += im;
    }

    sum_re *= 0.5;
    sum_im *= 0.5;

    for (r = 1; r < n_rho; r++)
        for (p = 1; p < n_phi; p++) {
            float rho = ra + step_rho * (float)r;
            float phi = pa + step_phi * (float)p;

            gabor(r0, theta0, omega, alpha, beta, rho, phi, &re, &im);
            sum_re += re;
            sum_im += im;
        }

    //return ((sum_re < 0) ? 1 : 0) | ((sum_im < 0) ? 2 : 0);
    *gabor_re = sum_re; // * step_rho * step_phi;
    *gabor_im = sum_im; // * step_rho * step_phi;
}

#include <stdio.h>
#include <stdlib.h>

void generate_gabor_pattern(int iris_w, int iris_h, int iris_p, uint8_t *iris_d, uint8_t *pattern)
{
    img_w = iris_w;
    img_h = iris_h;
    img_p = iris_p;
    img_d = iris_d;

    int r, t, a, b, o;

    memset(pattern, 0, 256);

    for (o = 0; o < 16; o++) {
        float omega = 5.0 * (float)(o + 1) + 10.0;

        float alpha = 0.025;
        float beta = 0.4;

        float *gabor_re, *gabor_im;
        gabor_re = malloc(iris_w * iris_h * sizeof(float));
        gabor_im = malloc(iris_w * iris_h * sizeof(float));

        for (r = 0; r < iris_h; r++)
            for (t = 0; t < iris_w; t++) {
                float r0 = (float)r / (float)iris_h;
                float theta0 = (float)t * 2.0 * PI / (float)iris_w;

                gabor_integrate(r0, theta0, omega, alpha, beta, r0 - alpha * 2.0, r0 + alpha * 2.0, theta0 - beta * 2.0, theta0 - beta * 2.0, 20, 50,
                                gabor_re + r * iris_w + t, gabor_im + r * iris_w + t);
            }

        {
            char path[256];
            snprintf(path, sizeof(path), "gabor_%f.ppm", omega);
            FILE *file = fopen(path, "w");
            fprintf(file, "P6\n%d %d\n255\n", iris_w, iris_h);
            int p;
            for (p = 0; p < iris_w * iris_h; p++) {
                fputc((int)(gabor_re[p] / 16.0) + 128, file);
                fputc(0, file);
                fputc((int)(gabor_im[p] / 16.0) + 128, file);
            }
            fclose(file);
            printf("gabor wavelet for frequency %f written to `%s'\n", omega, path);
        }

        free(gabor_re);
        free(gabor_im);
    }

    /*
    for (r = 0; r < 4; r++) {
        float r0 = (float)r / 3.0;

        for (t = 0; t < 4; t++) {
            float theta0 = (float)t * 2.0 * PI / 3.0;

            for (a = 0; a < 4; a++) {
                float alpha = 0.15 + (1.2 - 0.15) * (float)a / 4.0;

                for (b = 0; b < 4; b++) {
                    float beta = 0.15 + (1.2 - 0.15) * (float)b / 4.0;

                    for (o = 0; o < 4; o++) {
                        float omega = 20.0 + (float)o * 40.0;

                        pattern[64 * r + 16 * t + 4 * a + b + (o >> 2)] |= gabor_integrate(r0, theta0, omega, alpha, beta, 0.0, 1.0, 0.0, 2.0 * PI, iris_h, iris_w) << (2 * (o & 3));
                    }
                }
            }
        }
    }
    */
}
