# Iris

PROG = iris
TESTFILE = test3.png
OBJECTS = main.o gabor.o convolve.co pixel.co histogram.co hough.co unroll.co

# CUDA Toolkit location
ifeq ($(shell uname -s),Linux)
	CUDA_INCL_PATH := /opt/cuda/include/
	CUDA_LIB_PATH := /opt/cuda/lib64/
	NVCC := nvcc
else
	CUDA_PATH := /Developer/NVIDIA/CUDA-7.0
	CUDA_INCL_PATH := $(CUDA_PATH)/include/
	CUDA_LIB_PATH := $(CUDA_PATH)//lib/
	NVCC := $(CUDA_PATH)/bin/nvcc
endif

all: $(PROG)

test: $(PROG)
	./$(PROG) $(TESTFILE)

$(PROG): $(OBJECTS)
	gcc -o $(PROG) $(OBJECTS) `pkg-config --libs gdk-pixbuf-2.0` -lm -L$(CUDA_LIB_PATH) -lcudart -lstdc++

%.o: %.c
	gcc -c -g -o $@ $< `pkg-config --cflags gdk-pixbuf-2.0` -I$(CUDA_INCL_PATH)

%.co: %.cu
	$(NVCC) -c -arch=sm_30 -o $@ $<
#	nvcc -c -G -arch=sm_30 -o $@ $<

clean:
	rm -f $(PROG) *.o *.co *.ppm *.pgm
