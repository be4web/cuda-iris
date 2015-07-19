#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <mongoc.h>

#include "iris.h"
#include "database.h"

#define MONGO_CLIENT "mongodb://localhost:27017/"
#define MONGO_COLLECTION "Iris"

int main(int argc, char *argv[])
{
    float feature_vect[32];
    mongoc_client_t *client;
    mongoc_collection_t *collection;
    int op;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <operation> <file>\n", argv[0]);
        fprintf(stderr, "operation: insert | search\n");
        fprintf(stderr, "file: path to iris image file\n");
        return 1;
    }

    if (!strcmp(argv[1], "insert"))
        op = 1;
    else if (!strcmp(argv[1], "search"))
        op = 2;
    else {
        fprintf(stderr, "error: unknown operation `%s'\n", argv[1]);
        return 1;
    }

    mongoc_init();

    if ((client = mongoc_client_new(MONGO_CLIENT)) == NULL) {
        fprintf(stderr, "error connecting to client `%s'\n", MONGO_CLIENT);
        return 2;
    }

    if ((collection = mongoc_client_get_collection(client, MONGO_COLLECTION, MONGO_COLLECTION)) == NULL) {
        fprintf(stderr, "error getting collection `%s'\n", MONGO_COLLECTION);
        return 2;
    }

    if (get_iris_features(argv[2], feature_vect) < 0) {
        fprintf(stderr, "error getting iris features\n");
        return 3;
    }

    if (op == 1)
        insert_data_database(collection, feature_vect, MONGO_COLLECTION);

    else if (op == 2)
        printf("Result: %s\n", search_vector_database(collection, feature_vect));

    mongoc_collection_destroy (collection);
    mongoc_client_destroy (client);
    return 0;
}
