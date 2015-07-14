#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

int main(int argc, char *argv[])
{
    char buf1[1024], buf2[1024], *endp1, *endp2;
    FILE *f1, *f2;
    float dist, diff, v1, v2;
    int i;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <file1> <file2>\n", argv[0]);
        return 1;
    }

    if ((f1 = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "error opening `%s': %s\n", argv[1], strerror(errno));
        return 2;
    }

    if ((f2 = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "error opening `%s': %s\n", argv[2], strerror(errno));
        return 2;
    }

    while (1) {
        if (fgets(buf1, sizeof(buf1), f1) == NULL)
            break;

        if (fgets(buf2, sizeof(buf2), f2) == NULL)
            break;

        endp1 = buf1;
        endp2 = buf2;

        for (i = 0; i < 4; i++) {
            v1 = strtof(endp1, &endp1);
            v2 = strtof(endp2, &endp2);

            diff = v1 - v2;
            dist += diff * diff;
        }
    }

    printf("%f\n", sqrt(dist));
    return 0;
}
