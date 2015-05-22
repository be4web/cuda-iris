# Convolve

PROG = convolve
TESTFILE = test.gif
OBJECTS = main.o convolve.co pixel.co reduction.co hough.co

all: $(PROG)

test: $(PROG)
	./$(PROG) $(TESTFILE)

$(PROG): $(OBJECTS)
	gcc -o $(PROG) $(OBJECTS) `pkg-config --libs gdk-pixbuf-2.0` -lm -L/opt/cuda/lib64/ -lcudart

%.o: %.c
	gcc -c -g -o $@ $< `pkg-config --cflags gdk-pixbuf-2.0` -I/opt/cuda/include/

%.co: %.cu
	nvcc -c -arch=sm_30 -o $@ $<
#	nvcc -c -G -arch=sm_30 -o $@ $<

clean:
	rm -f $(PROG) *.o *.co *.ppm *.pgm
