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

    float common = img_d[img_p * y + x] * exp(-(rho_diff * rho_diff) / alpha2) * exp(-(phi_diff * phi_diff) / beta2) * rho;

    float exp_im = -omega * phi_diff;

    *re = cos(exp_im) * common;
    *im = sin(exp_im) * common;
}

static int gabor_integrate(float r0, float theta0, float omega, float alpha, float beta, float ra, float rb, float pa, float pb, int n_rho, int n_phi)
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

    return ((sum_re < 0) ? 1 : 0) | ((sum_im < 0) ? 2 : 0);
}

void generate_gabor_pattern(int iris_w, int iris_h, int iris_p, uint8_t *iris_d, uint8_t *pattern)
{
    img_w = iris_w;
    img_h = iris_h;
    img_p = iris_p;
    img_d = iris_d;

    int a, b, o;

    memset(pattern, 0, 256);

    for (a = 0; a < 8; a++) {
        float alpha = (float)a / 8;

        for (b = 0; b < 8; b++) {
            float beta = (float)b / 8;

            for (o = 0; o < 16; o++) {
                float omega = (float)o * 2.0 * PI / 16.0;

                pattern[32 * a + 4 * b + (o >> 2)] |= gabor_integrate(0.0, 0.0, omega, alpha, beta, 0.0, 1.0, 0.0, 2.0 * PI, iris_h, iris_w) << (2 * (o & 3));
            }
        }
    }
}
