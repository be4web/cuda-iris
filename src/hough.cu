extern "C" {
#include "hough.h"
#include <stdint.h>
}
//#include <cub/cub.cuh>

/****** PARAMETERS ********/
#define MARGIN 20
// needs to be a power of 2 as well
#define STEP_SIZE 2

// diff should be a multiple of 2 to exploit warp size efficiently (> 32)
#define MIN_RAD 6
#define MAX_RAD 262
#define PI 3.141592654

#define THREAD_COUNT (MAX_RAD - MIN_RAD)
#define WARP_SIZE 32

texture<float, cudaTextureType2D> phi_tex;
texture<float, cudaTextureType2D> abs_tex;

/*
 * Warp reduce sum: http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 **/

__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down(val,offset);
  return val;
}

__inline__ __device__
int blockReduceSum(int val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void hough_transform(int *result) {

    float dx, dy;
    int radius_index = threadIdx.x;
    float rad = MIN_RAD + radius_index + STEP_SIZE;
    int x = MARGIN + blockIdx.x;
    int y = MARGIN + blockIdx.y;
    int width = MARGIN + gridDim.x;


    int local_acc = int(0);

    int phi;
    for (phi = -120; phi < 120; phi++) {
        float phi_f = (float)phi * PI / 120.0;

        dx = rad * __cosf(phi_f);
        dy = rad * __sinf(phi_f);

        if (tex2D(abs_tex, x + dx, y + dy) > 20) {
            float grad = tex2D(phi_tex, x + dx, y + dy) - phi_f;
            if (-PI / 12.0 < grad && grad < PI / 12.0)
                local_acc++;
        }
    }


    local_acc = blockReduceSum(local_acc * local_acc * local_acc);

    if (threadIdx.x == 0)
        result[width * y + x] = local_acc;

}

#include <stdio.h>

extern "C" void cu_hough(int img_w, int img_h, int pitch, void *gm_abs, void *gm_phi, void *gm_hough)
{
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(NULL, abs_tex, gm_abs, desc, img_w, img_h, pitch);
    cudaBindTexture2D(NULL, phi_tex, gm_phi, desc, img_w, img_h, pitch);

    cudaMemset(gm_hough, 0, img_w * img_h * sizeof(int));

    cudaEventRecord(start);

    dim3 grid(img_w - MARGIN, img_h - MARGIN);
    dim3 threads(THREAD_COUNT/STEP_SIZE);

    hough_transform<<<grid, threads>>>((int *)gm_hough);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Took %f ms to compute Hough-Transform\n", elapsed_time);
}

__global__ void center_detection(int img_w, int img_h, int *accumulator) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = (threadIdx.x + threadIdx.y * blockDim.x) % warpSize;

    //check for uint type
    __shared__ float block_x, block_y, block_c;
    float center_x = 0,
          center_y = 0,
          c = 0;

    //intit shared variable
    if (threadIdx.x == 0 && threadIdx.y == 0)
        block_x = 0; block_y = 0, block_c = 0;

    __syncthreads();

    //warp reduction
    unsigned int cond = accumulator[y * img_w + x] >  (1 << 22);
    c = __popc(__ballot(cond));
    center_x = warpReduceSum(cond * x);
    center_y = warpReduceSum(cond * y);

    //block reduction
    if (lane == 0) {
        atomicAdd(&block_x, center_x);
        atomicAdd(&block_y, center_y);
        atomicAdd(&block_c, c);
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && c != 0) {
        accumulator[2*(blockIdx.x + gridDim.x * blockIdx.y)] = block_x/block_c;
        accumulator[2*(blockIdx.x + gridDim.x * blockIdx.y) + 1] = block_y/block_c;
    }

}
/**
 *
 *
 **/
__global__ void arithmetic_reduce(int *in, int2 *out) {
    int sumX = 0, sumY = 0;
    __shared__  int cX, cY;

    if (threadIdx.x == 0)
        cX = 0, cY = 0;

    sumX = blockReduceSum(in[2 * threadIdx.x]);
    sumY = blockReduceSum(in[2 * threadIdx.x + 1]);

    if (in[2 * threadIdx.x] > 0) {
        atomicAdd(&cX, 1);
    }

    if (in[2 * threadIdx.x + 1] > 0) {
        atomicAdd(&cY, 1);
    }

    __syncthreads();
    if(threadIdx.x == 0) {
        out->x = sumX/cX;
        out->y = sumY/cY;
    }
}

/**
 *
 * Thread size is limited to 1024. This limits the potential image dimensions if we keep
 * the block size of 8 * 8 = 64!
 *
 **/
extern "C" void cu_center_detection(int img_w, int img_h, void *gm_hough, int *center_x, int *center_y) {
    const int N = 8;
    dim3 blocks(img_w / N, img_h / N);
    dim3 threads(N, N);
    int2 center = make_int2(0,0);
    int2 *gm_center;

    center_detection<<<blocks, threads>>>(img_w, img_h, (int *) gm_hough);
    dim3 red_threads((img_w / N) * (img_h / N));

    cudaMalloc(&gm_center, sizeof(int2));

    arithmetic_reduce<<<1, red_threads>>>((int *)gm_hough, gm_center);

    cudaMemcpy(&center, gm_center, sizeof(int2), cudaMemcpyDeviceToHost);

    *center_x = center.x;
    *center_y = center.y;
}
