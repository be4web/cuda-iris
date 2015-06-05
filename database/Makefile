all:
	rm database.o -f
	gcc -o database.o -c database.c `pkg-config --cflags --libs libmongoc-1.0`
	gcc mongodb.c `pkg-config --cflags --libs libmongoc-1.0` -o mongodbinterface database.o -lm
test:
	./mongodbinterface	
clean:
	rm mongodbinterface database.o -f
