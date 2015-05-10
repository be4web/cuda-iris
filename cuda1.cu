#include <stdio.h>
#include <stdint.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <math_functions.h>

//eror handling stuff - copy from cuda by example book
#include "common/cuda_error.h"

#define MARGIN 20
// needs to be a power of 2 as well
#define STEP_SIZE 2

// diff should be a multiple of 2 to exploit warp size efficiently (> 32)
#define MIN_RAD 6
#define MAX_RAD 262
#define PI 3.141592654

#define THREAD_COUNT (MAX_RAD - MIN_RAD)

// TODO: change to polar coordinates
texture<float, 2> x_component;
texture<float, 2> y_component;

/**
 * CUDA Kernels.
 *
 **/
__global__ void gray_scale(uint8_t *d) {
	
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = (d[4 * p] + d[4 * p + 1] + d[4 * p + 2]) / 3;
	__syncthreads();
	d[p] = temp;
}

__global__ void hough_transform(int16_t *d, int *result) {
	
	__shared__ int radius_acc[THREAD_COUNT];
	
	int p_dx, p_dy;
	int radius_index = threadIdx.x;
	int rad = MIN_RAD + radius_index + STEP_SIZE;
	int x = MARGIN + blockIdx.x;
	int y = MARGIN + blockIdx.y;
	int width = MARGIN + gridDim.x;
	int height = MARGIN + gridDim.y;
	
	int local_acc = 0;
	radius_acc[radius_index] = 0;
	
	// TODO: doesn't unroll - why???!
#pragma unroll
	for (float phi = 0.0; phi < 2 * PI; phi += PI / 120) {
		p_dx = (float) rad * cosf(phi);
		p_dy = (float) rad * sinf(phi);
		

		// TODO: boundary check may not be necessary any more if we use texture memory
        if ((int) x + p_dx >= 0 && x + p_dx < width && (int) y + p_dy >= 0 && y + p_dy < height) {
        
		    int hori = d[((y + p_dy) * width +x + p_dx) * 2];
            int vert = d[((y + p_dy) * width + x + p_dx) * 2 + 1];
			
            if ( abs(hori) > 20 || abs(vert) > 20) {
				float grad = PI + atan2f(vert, hori) - phi;
				 
                if (-PI / 12.0 < grad && grad < PI / 12.0)
					local_acc++;
            }
        }
	}

	radius_acc[radius_index] = local_acc * local_acc * local_acc;
	
	// add all the stuff - TODO: naive approach
	__syncthreads();
	
	//relies on the fact that the # of threads is a power of 2
	int i = THREAD_COUNT/2;
	
	while (i != 0) {
		if (radius_index < i) {
			radius_acc[radius_index] += radius_acc[radius_index + i];
		}
		__syncthreads();
		i /= 2;
	}
	
	// this is the real shit...
	if (radius_index == 0) 
		result[x + y * width] = radius_acc[0];


}


/**
 * Gets the gradient
 *
 **/
int16_t *get_sobel(int width, int height, uint8_t *img)
{
    int16_t *sobel;
    int x, y;

    sobel = (int16_t *) malloc(width * height * 4 * sizeof(int16_t));

    for (x = 0; x < width; x++) {
        sobel[x * 2] = 0;
        sobel[x * 2 + 1] = 0;
        sobel[(width * (height - 1) + x) * 2] = 0;
        sobel[(width * (height - 1) + x) * 2 + 1] = 0;
    }

    for (y = 0; y < height; y++) {
        sobel[(width * y) * 2] = 0;
        sobel[(width * y) * 2 + 1] = 0;
        sobel[(width * y + height - 1) * 2] = 0;
        sobel[(width * y + height - 1) * 2 + 1] = 0;
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
/**
 * Blurs the image.
 *
 **/
uint8_t *get_gaussian_blur(int width, int height, uint8_t *img)
{
    uint8_t *gauss;
    int x, y;

    gauss = (uint8_t *) malloc(width * height);

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


/**
 * Main program.
 * Reference to image output format see http://en.wikipedia.org/wiki/Netpbm_format
 *
 **/
int main(int argc, char *argv[]) {
    GdkPixbuf *img;
    GError *err = NULL;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    if (argc < 2) {
        fprintf(stderr, "usage: %s <image file>\n", argv[0]);
        return 1;
    }

    if ((img = gdk_pixbuf_new_from_file(argv[1], &err)) == NULL) {
        fprintf(stderr, "error loading\n");
        return 2;
    }

	// get image dimensions
    int img_w = gdk_pixbuf_get_width(img),
        img_h = gdk_pixbuf_get_height(img);
        //img_s = gdk_pixbuf_get_rowstride(img),
        //img_c = gdk_pixbuf_get_n_channels(img),
        //img_b = gdk_pixbuf_get_bits_per_sample(img);

    uint8_t *img_d = gdk_pixbuf_get_pixels(img);
	
    
	
	uint8_t *cuda_mem;
	int16_t *sobel;
	
	// ------------ GPU ------------ //
    HANDLE_ERROR(cudaMalloc(&cuda_mem, img_w * img_h * 4));
    HANDLE_ERROR(cudaMemcpy(cuda_mem, img_d, img_w * img_h * 4, cudaMemcpyHostToDevice));

	//TODO: copying brings some seriouse performance bottleneck
    gray_scale<<<img_w, img_h>>>(cuda_mem);

    //synchronizing
	HANDLE_ERROR(cudaMemcpy(img_d, cuda_mem, img_w * img_h, cudaMemcpyDeviceToHost));
	cudaFree(cuda_mem);
	
	//------------ CPU ------------ //
	
	// gaussian blur
	uint8_t *blured = get_gaussian_blur(img_w, img_h, img_d);
	
	// sobel
	sobel = get_sobel(img_w, img_h, blured);
		
	// ------------ GPU ------------ //
	// copy to (texture) memory
	int16_t *sobel_cuda;
    HANDLE_ERROR(cudaMalloc(&sobel_cuda, img_w * img_h * 2 * sizeof(int16_t)));
    HANDLE_ERROR(cudaMemcpy(sobel_cuda, sobel, img_w * img_h * 2 * sizeof(int16_t), cudaMemcpyHostToDevice));
	
    {
        FILE *file = fopen("sobel.ppm", "w");
        fprintf(file, "P6\n%d %d\n255\n", img_w, img_h);
        int p;
        for (p = 0; p < img_w * img_h; p++) {
            fputc((sobel[p * 2] >> 1) + 128, file);
            fputc(0, file);
            fputc((sobel[p * 2 + 1] >> 1) + 128, file);
        }
        fclose(file);
    }
	
	
	// hough transform
	// performance metrics
	
	
	dim3 grid (img_w - MARGIN, img_h - MARGIN);
	dim3 threads (THREAD_COUNT/STEP_SIZE);
	
	cudaEventRecord(start);
	
	//cudaBindTexture2D(NULL, x_component, );
	
	int* hough_d = (int *) malloc(img_w * img_h * sizeof(int));
	int* hough_cuda;
	
	HANDLE_ERROR(cudaMalloc(&hough_cuda, img_w * img_h * sizeof(int)));
	HANDLE_ERROR(cudaMemset(hough_cuda, 0, img_w * img_h * sizeof(int)));
	
	hough_transform<<<grid, threads>>>(sobel_cuda, hough_cuda);
	
	HANDLE_ERROR(cudaMemcpy(hough_d, hough_cuda, img_w * img_h * sizeof(int), cudaMemcpyDeviceToHost));
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed_time = 0;
	
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Took %f ms to compute Hough-Transform\n", elapsed_time);
	
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
	
	cudaUnbindTexture(x_component);
	cudaUnbindTexture(y_component);
	
	// write gaussian to file
	/*
    {
        FILE *file = fopen("blur.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", img_w, img_h);
        int p;
        for (p = 0; p < img_w * img_h; p++)
            fputc(blured[p], file);
        fclose(file);
    }
	*/
	
	// free mem
	cudaFree(sobel_cuda);

    return 0;
}
