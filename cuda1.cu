#include <stdio.h>
#include <stdint.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <math.h>
#include <math_functions.h>

//eror handling stuff - copy from cuda by example book
#include "common/cuda_error.h"

// resides on the gpu
texture<float, cudaTextureType2D> phi_tex;

/**
 * CUDA Kernels.
 *
 **/
__global__ void color_to_gray_kernel(int color_p, uint32_t *color, int gray_p, uint8_t *gray, int coeff_r, int coeff_g, int coeff_b)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    int div = coeff_r + coeff_g + coeff_b;
    uint32_t clr = color[color_p * y + x];

    gray[gray_p * y + x] = ((clr & 0xff) * coeff_r + ((clr >> 8) & 0xff) * coeff_g + ((clr >> 16) & 0xff) * coeff_b) / div;
}

__global__ void image_resize_kernel(int src_p, uint8_t *src, int dst_p, uint8_t *dst, int img_w, int img_h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const float x_ratio = (float) img_w / (gridDim.x * blockDim.x);
	const float y_ratio = (float) img_h / (gridDim.y * blockDim.y);
	const int px = (int) floor((float) x * x_ratio), py = (int) floor((float) y * y_ratio);

	dst[x + y * dst_p] = src[src_p * py + px];	
}

void cu_color_to_gray(int img_w, int img_h, int color_p, void *gm_color, int gray_p, void *gm_gray)
{
    dim3 blocks(img_w / 8, img_h / 8);
    dim3 threads(8, 8);

    color_to_gray_kernel<<<blocks, threads>>>(color_p, (uint32_t *)gm_color, gray_p, (uint8_t *)gm_gray, 1, 1, 1);
}
/**
 * 
 *
 **/
void cu_resize(int img_w, int img_h, int src_p, void *gm_src, int dst_w, int dst_h, int dst_p, void *gm_dst) {
	dim3 blocks(dst_w / 8, dst_h / 8);
	dim3 threads(8, 8);
	image_resize_kernel<<<blocks, threads>>>(src_p, (uint8_t *) gm_src, dst_p, (uint8_t *) gm_dst, img_w, img_h);

}

/**
 * Main program.
 * Reference to image output format see http://en.wikipedia.org/wiki/Netpbm_format
 *
 **/
int main(int argc, char *argv[]) {
    GdkPixbuf *img;
    GError *err = NULL;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <image file>\n", argv[0]);
        return 1;
    }

    if ((img = gdk_pixbuf_new_from_file(argv[1], &err)) == NULL) {
        fprintf(stderr, "error loading image file: %s\n", err->message);
        return 2;
    }

    int img_w = gdk_pixbuf_get_width(img),
        img_h = gdk_pixbuf_get_height(img),
        img_s = gdk_pixbuf_get_rowstride(img),
        img_c = gdk_pixbuf_get_n_channels(img),
        img_b = gdk_pixbuf_get_bits_per_sample(img);

    printf("image prop: w: %d, h: %d, s: %d, c: %d, b %d\n", img_w, img_h, img_s, img_c, img_b);

    uint8_t *img_d = gdk_pixbuf_get_pixels(img);
	
    
    void *gm_color, *gm_gray, *gm_resized;
    int pitch32;

    {
        struct cudaDeviceProp prop;
        int tex_align;
        cudaGetDeviceProperties(&prop, 0);
        tex_align = prop.textureAlignment - 1;

        pitch32 = ((img_w * 4 + tex_align) & ~tex_align) >> 2;
        printf("texture alignment: %d, 32-bit width: %d => pitch: %d, per sample: %d\n", tex_align + 1, img_w * 4, pitch32 * 4, pitch32);
    }

    cudaMalloc(&gm_color, pitch32 * img_h * 4);
    cudaMalloc(&gm_gray, img_w * img_h);
	
	int target_w = 100, target_h = (target_w * img_h)/img_w;
	
	cudaMalloc(&gm_resized, target_h * target_w);	
	
    int h;
    for (h = 0; h < img_h; h++)
        cudaMemcpy2D((uint8_t *) gm_color + pitch32 * 4 * h, 4, img_d + img_s * h, img_c, img_c, img_w, cudaMemcpyHostToDevice);

    cu_color_to_gray(img_w, img_h, pitch32, gm_color, img_w, gm_gray);
	
	
	cu_resize(img_w, img_h, img_w, (uint8_t *) gm_gray, target_w, target_h, target_w, gm_resized);
	
    {
        uint8_t *resized_d = (uint8_t *) malloc(target_w * target_h);
        cudaMemcpy(resized_d, gm_resized, target_w * target_h, cudaMemcpyDeviceToHost);

        FILE *file = fopen("gray.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", target_w, target_h);
        int p;
        for (p = 0; p < target_w * target_h; p++)
            fputc(resized_d[p], file);
        fclose(file);

        free(resized_d);
    }
	
	cudaFree(gm_color);
	cudaFree(gm_gray);
	cudaFree(gm_resized);

	
	// ------------ GPU ------------ //
	// hough transform
	// performance metrics
	
	// copy to (texture) memory
	/*
	float *gradient_phi, *gradient_abs;
	size_t pitch;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	
    HANDLE_ERROR(cudaMallocPitch(&gradient_phi, &pitch, img_w * sizeof(float), img_h));
    HANDLE_ERROR(cudaMemcpy2D(gradient_phi, pitch, sobel.phi, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaBindTexture2D(NULL, phi_tex, gradient_phi, desc, img_w, img_h, pitch));
    
	
    HANDLE_ERROR(cudaMallocPitch(&gradient_abs, &pitch, img_w * sizeof(float), img_h));
    HANDLE_ERROR(cudaMemcpy2D(gradient_abs, pitch, sobel.abs, img_w * sizeof(float), img_w * sizeof(float), img_h, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaBindTexture2D(NULL, abs_tex, gradient_abs, desc, img_w, img_h, pitch));
	*/
	
	/* Clean up */
	//cudaUnbindTexture(phi_tex);
	
	
	
	

	
    return 0;
}
