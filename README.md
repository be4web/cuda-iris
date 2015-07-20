# cuda-iris
Real-time iris recognition with CUDA

Work in progress.

To see current status:
`$ cd src/ && make test`

To get into the code:
take a look at [src/iris.c](https://github.com/be4web/cuda-iris/blob/master/src/iris.c),
uncomment the definition of `DEBUG` on [line 23](https://github.com/be4web/cuda-iris/blob/master/src/iris.c#L23)
to get debug output and a ton of pictures which visualize the process.

Dependencies:
* CUDA
* gdk-pixbuf-2.0
* libmongoc-1.0 (optional)
