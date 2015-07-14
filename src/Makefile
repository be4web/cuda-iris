# Iris

PROG = iris
OBJECTS = main.o convolve.co pixel.co histogram.co hough.co unroll.co

DISTANCE = distance

REFFILE = 009_2_1.bmp
TESTFILE1 = 009_2_4.bmp
TESTFILE2 = 013_2_3.bmp

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

all: $(PROG) $(DISTANCE)

test: $(PROG) $(DISTANCE)
	./$(PROG) $(REFFILE) > iris_ref.dat
	./$(PROG) $(TESTFILE1) > iris_test.dat
	./$(DISTANCE) iris_ref.dat iris_test.dat
	./$(PROG) $(TESTFILE2) > iris_test.dat
	./$(DISTANCE) iris_ref.dat iris_test.dat

$(DISTANCE): distance.c
	gcc -o $@ $< -lm

$(PROG): $(OBJECTS)
	gcc -o $(PROG) $(OBJECTS) `pkg-config --libs gdk-pixbuf-2.0` -lm -L$(CUDA_LIB_PATH) -lcudart -lstdc++

%.o: %.c
	gcc -c -g -o $@ $< `pkg-config --cflags gdk-pixbuf-2.0` -I$(CUDA_INCL_PATH)

%.co: %.cu
	$(NVCC) -c -G -arch=sm_30 -o $@ $<

clean:
	rm -f $(PROG) $(DISTANCE) *.o *.co *.ppm *.pgm *.dat
