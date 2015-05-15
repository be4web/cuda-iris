# Cuda1

PROG = cuda1
TESTFILE = test.gif
OBJECTS = cuda1.o

CFLAGS = -D_REENTRANT -I/usr/local/Cellar/gdk-pixbuf/2.30.8/include/gdk-pixbuf-2.0 -I/usr/local/Cellar/libpng/1.6.17/include/libpng16 -I/usr/local/Cellar/glib/2.44.0/include/glib-2.0 -I/usr/local/Cellar/glib/2.44.0/lib/glib-2.0/include -I/usr/local/opt/gettext/include 



# Location of the CUDA Toolkit
CUDA_PATH       ?= /Developer/NVIDIA/CUDA-6.5
NVCC := $(CUDA_PATH)/bin/nvcc

all: $(PROG)

test: $(PROG)
	./$(PROG) $(TESTFILE)

$(PROG): $(OBJECTS)
	$(NVCC) `pkg-config --libs gdk-pixbuf-2.0` -lm -o $(PROG) $(OBJECTS)


%.o: %.cu #device arch needed for __warp reduce
	$(NVCC) $(CFLAGS)  -c -o $@ $<   -arch=sm_30 #--keep

clean:
	rm -f $(PROG) *.o *.ppm *.pgm *.gpu *.ptx *.i *.ii *.stub.c *.fatbin* *.hash *.cudafe* *.cubin *.module_id

########################################################################
#							#                                          #
# sm_20						#	Basic features                         #
#							#                                          #
#							#	+ Fermi support                        # 
#							#                                          #
########################################################################	
#							#	+ Kepler support					   #
# sm_30 and sm_32			#										   #
#							#	+ Unified memory programming		   #
########################################################################
# sm_35						#	+ Dynamic parallelism support		   #
# sm_50, sm_52, and sm_53	#	+ Maxwell support					   #
########################################################################

# Read more at: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ixzz3ZmqYiS1l 
# Follow us: @GPUComputing on Twitter | NVIDIA on Facebook