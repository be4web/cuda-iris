# Convolve

PROG = convolve
TESTFILE = test.gif
OBJECTS = main.o cu_convolve.co cu_pixel.co cu_reduction.co

all: $(PROG)

test: $(PROG)
	./$(PROG) $(TESTFILE)

$(PROG): $(OBJECTS)
	gcc -o $(PROG) $(OBJECTS) `pkg-config --libs gdk-pixbuf-2.0` -L/opt/cuda/lib64/ -lcudart

%.o: %.c
	gcc -c -g -o $@ $< `pkg-config --cflags gdk-pixbuf-2.0` -I/opt/cuda/include/

%.co: %.cu
	nvcc -c -G -o $@ $<

clean:
	rm -f $(PROG) *.o *.co *.ppm *.pgm
