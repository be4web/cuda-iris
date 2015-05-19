extern "C" {
#include "convolve.h"
#include <stdint.h>
}

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

    //__shared__ int radius_acc[32];

    float dx, dy;
    int radius_index = threadIdx.x;
    float rad = MIN_RAD + radius_index + STEP_SIZE;
    int x = MARGIN + blockIdx.x;
    int y = MARGIN + blockIdx.y;
    int width = MARGIN + gridDim.x;


    int local_acc = int(0);

    /*
    int phi;
    for (phi = 0; phi < 240; phi++) {
        float phi_f = (float)phi * PI / 120.0;

        dx = rad * cos(phi_f);
        dy = rad * sin(phi_f);

        if (tex2D(abs_tex, x + dx, y + dy) > 20) {
            float grad = PI + tex2D(phi_tex, x + dx, y + dy) - phi_f;
            if (-PI / 12.0 < grad && grad < PI / 12.0)
                local_acc++;
        }
    }
    */

    dx = rad * 0.9996573250;
    dy = rad * 0.0261769483;


    {

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.02617993877991494;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }

    dx = rad * 0.9986295348;
    dy = rad * 0.0523359562;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.05235987755982988;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9969173337;
    dy = rad * 0.0784590957;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.07853981633974483;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9945218954;
    dy = rad * 0.1045284633;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.10471975511965977;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9914448614;
    dy = rad * 0.1305261922;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.1308996938995747;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9876883406;
    dy = rad * 0.1564344650;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.15707963267948966;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9832549076;
    dy = rad * 0.1822355255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.1832595714594046;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9781476007;
    dy = rad * 0.2079116908;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.20943951023931956;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9723699204;
    dy = rad * 0.2334453639;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.2356194490192345;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9659258263;
    dy = rad * 0.2588190451;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.26179938779914946;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9588197349;
    dy = rad * 0.2840153447;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.2879793265790644;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9510565163;
    dy = rad * 0.3090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.3141592653589793;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9426414911;
    dy = rad * 0.3338068592;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.34033920413889424;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9335804265;
    dy = rad * 0.3583679495;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.36651914291880916;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9238795325;
    dy = rad * 0.3826834324;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.3926990816987241;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9135454576;
    dy = rad * 0.4067366431;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.418879020478639;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9025852843;
    dy = rad * 0.4305110968;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.44505895925855393;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8910065242;
    dy = rad * 0.4539904997;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.47123889803846886;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8788171127;
    dy = rad * 0.4771587603;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.4974188368183838;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8660254038;
    dy = rad * 0.5000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.5235987755982987;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8526401644;
    dy = rad * 0.5224985647;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.5497787143782137;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8386705679;
    dy = rad * 0.5446390350;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.5759586531581287;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8241261886;
    dy = rad * 0.5664062369;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.6021385919380436;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8090169944;
    dy = rad * 0.5877852523;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.6283185307179586;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7933533403;
    dy = rad * 0.6087614290;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.6544984694978736;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7771459615;
    dy = rad * 0.6293203910;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.6806784082777886;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7604059656;
    dy = rad * 0.6494480483;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.7068583470577036;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7431448255;
    dy = rad * 0.6691306064;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.7330382858376185;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7253743710;
    dy = rad * 0.6883545757;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.7592182246175335;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7071067812;
    dy = rad * 0.7071067812;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.7853981633974485;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6883545757;
    dy = rad * 0.7253743710;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.8115781021773635;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6691306064;
    dy = rad * 0.7431448255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.8377580409572785;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6494480483;
    dy = rad * 0.7604059656;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.8639379797371934;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6293203910;
    dy = rad * 0.7771459615;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.8901179185171084;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6087614290;
    dy = rad * 0.7933533403;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.9162978572970234;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5877852523;
    dy = rad * 0.8090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.9424777960769384;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5664062369;
    dy = rad * 0.8241261886;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.9686577348568534;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5446390350;
    dy = rad * 0.8386705679;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 0.9948376736367683;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5224985647;
    dy = rad * 0.8526401644;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.0210176124166832;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5000000000;
    dy = rad * 0.8660254038;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.047197551196598;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4771587603;
    dy = rad * 0.8788171127;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.073377489976513;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4539904997;
    dy = rad * 0.8910065242;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.0995574287564278;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4305110968;
    dy = rad * 0.9025852843;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.1257373675363427;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4067366431;
    dy = rad * 0.9135454576;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.1519173063162575;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3826834324;
    dy = rad * 0.9238795325;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.1780972450961724;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3583679495;
    dy = rad * 0.9335804265;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.2042771838760873;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3338068592;
    dy = rad * 0.9426414911;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.2304571226560022;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3090169944;
    dy = rad * 0.9510565163;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.256637061435917;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2840153447;
    dy = rad * 0.9588197349;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.282817000215832;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2588190451;
    dy = rad * 0.9659258263;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.3089969389957468;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2334453639;
    dy = rad * 0.9723699204;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.3351768777756616;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2079116908;
    dy = rad * 0.9781476007;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.3613568165555765;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1822355255;
    dy = rad * 0.9832549076;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.3875367553354914;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1564344650;
    dy = rad * 0.9876883406;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.4137166941154062;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1305261922;
    dy = rad * 0.9914448614;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.439896632895321;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1045284633;
    dy = rad * 0.9945218954;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.466076571675236;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0784590957;
    dy = rad * 0.9969173337;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.4922565104551508;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0523359562;
    dy = rad * 0.9986295348;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.5184364492350657;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0261769483;
    dy = rad * 0.9996573250;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.5446163880149806;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0000000000;
    dy = rad * 1.0000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.5707963267948954;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.0261769483;
    dy = rad * 0.9996573250;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.5969762655748103;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.0523359562;
    dy = rad * 0.9986295348;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.6231562043547252;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.0784590957;
    dy = rad * 0.9969173337;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.64933614313464;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1045284633;
    dy = rad * 0.9945218954;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.675516081914555;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1305261922;
    dy = rad * 0.9914448614;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.7016960206944698;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1564344650;
    dy = rad * 0.9876883406;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.7278759594743847;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1822355255;
    dy = rad * 0.9832549076;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.7540558982542995;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2079116908;
    dy = rad * 0.9781476007;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.7802358370342144;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2334453639;
    dy = rad * 0.9723699204;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.8064157758141293;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2588190451;
    dy = rad * 0.9659258263;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.8325957145940441;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2840153447;
    dy = rad * 0.9588197349;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.858775653373959;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3090169944;
    dy = rad * 0.9510565163;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.8849555921538739;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3338068592;
    dy = rad * 0.9426414911;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.9111355309337887;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3583679495;
    dy = rad * 0.9335804265;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.9373154697137036;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3826834324;
    dy = rad * 0.9238795325;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.9634954084936185;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4067366431;
    dy = rad * 0.9135454576;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 1.9896753472735333;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4305110968;
    dy = rad * 0.9025852843;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.015855286053448;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4539904997;
    dy = rad * 0.8910065242;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.0420352248333633;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4771587603;
    dy = rad * 0.8788171127;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.0682151636132784;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5000000000;
    dy = rad * 0.8660254038;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.0943951023931935;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5224985647;
    dy = rad * 0.8526401644;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.1205750411731086;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5446390350;
    dy = rad * 0.8386705679;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.1467549799530237;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5664062369;
    dy = rad * 0.8241261886;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.1729349187329388;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5877852523;
    dy = rad * 0.8090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.199114857512854;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6087614290;
    dy = rad * 0.7933533403;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.225294796292769;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6293203910;
    dy = rad * 0.7771459615;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.251474735072684;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6494480483;
    dy = rad * 0.7604059656;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.277654673852599;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6691306064;
    dy = rad * 0.7431448255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.303834612632514;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6883545757;
    dy = rad * 0.7253743710;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.3300145514124293;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7071067812;
    dy = rad * 0.7071067812;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.3561944901923444;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7253743710;
    dy = rad * 0.6883545757;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.3823744289722595;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7431448255;
    dy = rad * 0.6691306064;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.4085543677521746;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7604059656;
    dy = rad * 0.6494480483;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.4347343065320897;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7771459615;
    dy = rad * 0.6293203910;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.4609142453120048;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7933533403;
    dy = rad * 0.6087614290;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.48709418409192;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8090169944;
    dy = rad * 0.5877852523;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.513274122871835;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8241261886;
    dy = rad * 0.5664062369;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.53945406165175;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8386705679;
    dy = rad * 0.5446390350;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.565634000431665;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8526401644;
    dy = rad * 0.5224985647;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.59181393921158;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8660254038;
    dy = rad * 0.5000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.6179938779914953;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8788171127;
    dy = rad * 0.4771587603;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.6441738167714104;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8910065242;
    dy = rad * 0.4539904997;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.6703537555513255;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9025852843;
    dy = rad * 0.4305110968;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.6965336943312406;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9135454576;
    dy = rad * 0.4067366431;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.7227136331111557;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9238795325;
    dy = rad * 0.3826834324;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.7488935718910708;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9335804265;
    dy = rad * 0.3583679495;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.775073510670986;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9426414911;
    dy = rad * 0.3338068592;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.801253449450901;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9510565163;
    dy = rad * 0.3090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.827433388230816;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9588197349;
    dy = rad * 0.2840153447;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.853613327010731;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9659258263;
    dy = rad * 0.2588190451;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.879793265790646;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9723699204;
    dy = rad * 0.2334453639;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.9059732045705613;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9781476007;
    dy = rad * 0.2079116908;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.9321531433504764;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9832549076;
    dy = rad * 0.1822355255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.9583330821303915;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9876883406;
    dy = rad * 0.1564344650;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 2.9845130209103066;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9914448614;
    dy = rad * 0.1305261922;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.0106929596902217;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9945218954;
    dy = rad * 0.1045284633;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.0368728984701368;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9969173337;
    dy = rad * 0.0784590957;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.063052837250052;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9986295348;
    dy = rad * 0.0523359562;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.089232776029967;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9996573250;
    dy = rad * 0.0261769483;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.115412714809882;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -1.0000000000;
    dy = rad * -0.0000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.141592653589797;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9996573250;
    dy = rad * -0.0261769483;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.167772592369712;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9986295348;
    dy = rad * -0.0523359562;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.1939525311496273;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9969173337;
    dy = rad * -0.0784590957;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.2201324699295424;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9945218954;
    dy = rad * -0.1045284633;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.2463124087094575;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9914448614;
    dy = rad * -0.1305261922;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.2724923474893726;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9876883406;
    dy = rad * -0.1564344650;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.2986722862692877;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9832549076;
    dy = rad * -0.1822355255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.3248522250492027;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9781476007;
    dy = rad * -0.2079116908;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.351032163829118;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9723699204;
    dy = rad * -0.2334453639;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.377212102609033;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9659258263;
    dy = rad * -0.2588190451;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.403392041388948;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9588197349;
    dy = rad * -0.2840153447;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.429571980168863;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9510565163;
    dy = rad * -0.3090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.455751918948778;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9426414911;
    dy = rad * -0.3338068592;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.4819318577286933;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9335804265;
    dy = rad * -0.3583679495;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.5081117965086084;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9238795325;
    dy = rad * -0.3826834324;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.5342917352885235;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9135454576;
    dy = rad * -0.4067366431;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.5604716740684386;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.9025852843;
    dy = rad * -0.4305110968;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.5866516128483537;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8910065242;
    dy = rad * -0.4539904997;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.6128315516282687;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8788171127;
    dy = rad * -0.4771587603;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.639011490408184;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8660254038;
    dy = rad * -0.5000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.665191429188099;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8526401644;
    dy = rad * -0.5224985647;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.691371367968014;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8386705679;
    dy = rad * -0.5446390350;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.717551306747929;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8241261886;
    dy = rad * -0.5664062369;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.743731245527844;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.8090169944;
    dy = rad * -0.5877852523;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.7699111843077593;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7933533403;
    dy = rad * -0.6087614290;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.7960911230876744;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7771459615;
    dy = rad * -0.6293203910;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.8222710618675895;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7604059656;
    dy = rad * -0.6494480483;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.8484510006475046;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7431448255;
    dy = rad * -0.6691306064;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.8746309394274197;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7253743710;
    dy = rad * -0.6883545757;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.9008108782073347;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.7071067812;
    dy = rad * -0.7071067812;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.92699081698725;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6883545757;
    dy = rad * -0.7253743710;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.953170755767165;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6691306064;
    dy = rad * -0.7431448255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 3.97935069454708;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6494480483;
    dy = rad * -0.7604059656;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.005530633326995;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6293203910;
    dy = rad * -0.7771459615;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.03171057210691;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.6087614290;
    dy = rad * -0.7933533403;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.057890510886825;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5877852523;
    dy = rad * -0.8090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.08407044966674;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5664062369;
    dy = rad * -0.8241261886;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.110250388446655;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5446390350;
    dy = rad * -0.8386705679;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.13643032722657;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5224985647;
    dy = rad * -0.8526401644;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.162610266006485;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.5000000000;
    dy = rad * -0.8660254038;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.1887902047864;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4771587603;
    dy = rad * -0.8788171127;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.214970143566315;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4539904997;
    dy = rad * -0.8910065242;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.2411500823462305;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4305110968;
    dy = rad * -0.9025852843;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.267330021126146;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.4067366431;
    dy = rad * -0.9135454576;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.293509959906061;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3826834324;
    dy = rad * -0.9238795325;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.319689898685976;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3583679495;
    dy = rad * -0.9335804265;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.345869837465891;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3338068592;
    dy = rad * -0.9426414911;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.372049776245806;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.3090169944;
    dy = rad * -0.9510565163;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.398229715025721;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2840153447;
    dy = rad * -0.9588197349;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.424409653805636;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2588190451;
    dy = rad * -0.9659258263;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.450589592585551;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2334453639;
    dy = rad * -0.9723699204;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.476769531365466;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.2079116908;
    dy = rad * -0.9781476007;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.502949470145381;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1822355255;
    dy = rad * -0.9832549076;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.5291294089252965;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1564344650;
    dy = rad * -0.9876883406;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.555309347705212;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1305261922;
    dy = rad * -0.9914448614;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.581489286485127;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.1045284633;
    dy = rad * -0.9945218954;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.607669225265042;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.0784590957;
    dy = rad * -0.9969173337;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.633849164044957;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.0523359562;
    dy = rad * -0.9986295348;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.660029102824872;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * -0.0261769483;
    dy = rad * -0.9996573250;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.686209041604787;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0000000000;
    dy = rad * -1.0000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.712388980384702;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0261769483;
    dy = rad * -0.9996573250;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.738568919164617;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0523359562;
    dy = rad * -0.9986295348;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.764748857944532;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.0784590957;
    dy = rad * -0.9969173337;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.790928796724447;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1045284633;
    dy = rad * -0.9945218954;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.8171087355043625;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1305261922;
    dy = rad * -0.9914448614;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.843288674284278;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1564344650;
    dy = rad * -0.9876883406;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.869468613064193;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.1822355255;
    dy = rad * -0.9832549076;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.895648551844108;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2079116908;
    dy = rad * -0.9781476007;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.921828490624023;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2334453639;
    dy = rad * -0.9723699204;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.948008429403938;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2588190451;
    dy = rad * -0.9659258263;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 4.974188368183853;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.2840153447;
    dy = rad * -0.9588197349;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.000368306963768;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3090169944;
    dy = rad * -0.9510565163;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.026548245743683;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3338068592;
    dy = rad * -0.9426414911;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.052728184523598;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3583679495;
    dy = rad * -0.9335804265;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.078908123303513;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.3826834324;
    dy = rad * -0.9238795325;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.1050880620834285;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4067366431;
    dy = rad * -0.9135454576;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.131268000863344;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4305110968;
    dy = rad * -0.9025852843;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.157447939643259;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4539904997;
    dy = rad * -0.8910065242;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.183627878423174;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.4771587603;
    dy = rad * -0.8788171127;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.209807817203089;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5000000000;
    dy = rad * -0.8660254038;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.235987755983004;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5224985647;
    dy = rad * -0.8526401644;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.262167694762919;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5446390350;
    dy = rad * -0.8386705679;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.288347633542834;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5664062369;
    dy = rad * -0.8241261886;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.314527572322749;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.5877852523;
    dy = rad * -0.8090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.340707511102664;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6087614290;
    dy = rad * -0.7933533403;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.366887449882579;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6293203910;
    dy = rad * -0.7771459615;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.3930673886624945;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6494480483;
    dy = rad * -0.7604059656;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.41924732744241;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6691306064;
    dy = rad * -0.7431448255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.445427266222325;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.6883545757;
    dy = rad * -0.7253743710;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.47160720500224;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7071067812;
    dy = rad * -0.7071067812;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.497787143782155;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7253743710;
    dy = rad * -0.6883545757;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.52396708256207;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7431448255;
    dy = rad * -0.6691306064;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.550147021341985;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7604059656;
    dy = rad * -0.6494480483;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.5763269601219;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7771459615;
    dy = rad * -0.6293203910;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.602506898901815;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.7933533403;
    dy = rad * -0.6087614290;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.62868683768173;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8090169944;
    dy = rad * -0.5877852523;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.654866776461645;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8241261886;
    dy = rad * -0.5664062369;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.6810467152415605;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8386705679;
    dy = rad * -0.5446390350;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.7072266540214756;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8526401644;
    dy = rad * -0.5224985647;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.733406592801391;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8660254038;
    dy = rad * -0.5000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.759586531581306;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8788171127;
    dy = rad * -0.4771587603;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.785766470361221;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.8910065242;
    dy = rad * -0.4539904997;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.811946409141136;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9025852843;
    dy = rad * -0.4305110968;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.838126347921051;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9135454576;
    dy = rad * -0.4067366431;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.864306286700966;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9238795325;
    dy = rad * -0.3826834324;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.890486225480881;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9335804265;
    dy = rad * -0.3583679495;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.916666164260796;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9426414911;
    dy = rad * -0.3338068592;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.942846103040711;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9510565163;
    dy = rad * -0.3090169944;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.9690260418206265;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9588197349;
    dy = rad * -0.2840153447;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 5.9952059806005416;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9659258263;
    dy = rad * -0.2588190451;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.021385919380457;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9723699204;
    dy = rad * -0.2334453639;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.047565858160372;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9781476007;
    dy = rad * -0.2079116908;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.073745796940287;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9832549076;
    dy = rad * -0.1822355255;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.099925735720202;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9876883406;
    dy = rad * -0.1564344650;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.126105674500117;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9914448614;
    dy = rad * -0.1305261922;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.152285613280032;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9945218954;
    dy = rad * -0.1045284633;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.178465552059947;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9969173337;
    dy = rad * -0.0784590957;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.204645490839862;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9986295348;
    dy = rad * -0.0523359562;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.230825429619777;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 0.9996573250;
    dy = rad * -0.0261769483;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.2570053683996925;
        if (-PI / 12.0 < grad && grad < PI / 12.0)
            local_acc++;
    }



    dx = rad * 1.0000000000;
    dy = rad * 0.0000000000;

    if (tex2D(abs_tex, x + dx, y + dy)  > 20) {
        float grad = PI + tex2D(phi_tex, x + dx, y + dy) - 6.2831853071796075;
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
    printf("cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Took %f ms to compute Hough-Transform\n", elapsed_time);
}
