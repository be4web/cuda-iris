all:
	gcc -o database.o -c database.c `pkg-config --cflags --libs libmongoc-1.0`
	gcc mongodb.c `pkg-config --cflags --libs libmongoc-1.0` -o mongodbinterface database.o -lm
clean:
	rm mongodbinterface database.o -f
