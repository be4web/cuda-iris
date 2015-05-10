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

%.o: %.cu
	$(NVCC) $(CFLAGS)  -c -o $@ $< #--keep

clean:
	rm -f $(PROG) *.o *.ppm *.pgm *.gpu *.ptx *.i *.ii *.stub.c *.fatbin* *.hash *.cudafe* *.cubin *.module_id
