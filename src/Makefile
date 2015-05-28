# Convolve

PROG = convolve
TESTFILE = test3.png
OBJECTS = main.o convolve.co pixel.co histogram.co hough.co unroll.co
# Location of the CUDA Toolkit
CUDA_PATH       ?= /Developer/NVIDIA/CUDA-7.0
NVCC := $(CUDA_PATH)/bin/nvcc

all: $(PROG)

test: $(PROG)
	./$(PROG) $(TESTFILE)

$(PROG): $(OBJECTS)
	gcc -o $(PROG) $(OBJECTS) `pkg-config --libs gdk-pixbuf-2.0` -lm -L$(CUDA_PATH)/lib/ -lcudart

%.o: %.c
	gcc -c -g -o $@ $< `pkg-config --cflags gdk-pixbuf-2.0` -I$(CUDA_PATH)/include/

%.co: %.cu
	$(NVCC) -c -arch=sm_30 -o $@ $<
#	nvcc -c -G -arch=sm_30 -o $@ $<

clean:
	rm -f $(PROG) *.o *.co *.ppm *.pgm
