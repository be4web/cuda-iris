/*!
 * \file test.c
 * Test application
 *
 * Extracts the feature vector of an iris image and prints it to stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "iris.h"

int main(int argc, char *argv[])
{
    float feature_vect[32];
    int f;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <image file>\n", argv[0]);
        return 1;
    }

    if (get_iris_features(argv[1], feature_vect) < 0) {
        fprintf(stderr, "error getting iris features\n");
        return 2;
    }

    for (f = 0; f < 16; f++)
        printf("%f %f\n", f, feature_vect[f * 2], feature_vect[f * 2 + 1]);

    return 0;
}
