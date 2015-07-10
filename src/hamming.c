#include <stdio.h>
#include <string.h>
#include <errno.h>

int main(int argc, char *argv[])
{
    char buf1[256], buf2[256];
    FILE *f1, *f2;
    int hd, i, j;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <file1> <file2>\n", argv[0]);
        return 1;
    }

    if ((f1 = fopen(argv[1], "rb")) == NULL) {
        fprintf(stderr, "error opening `%s': %s\n", argv[1], strerror(errno));
        return 2;
    }

    if ((f2 = fopen(argv[2], "rb")) == NULL) {
        fprintf(stderr, "error opening `%s': %s\n", argv[2], strerror(errno));
        return 2;
    }

    hd = 0;

    while (1) {
        if (fread(buf1, 1, sizeof(buf1), f1) != sizeof(buf1))
            break;

        if (fread(buf2, 1, sizeof(buf2), f2) != sizeof(buf2))
            break;

        for (i = 0; i < sizeof(buf1); i++)
            for (j = 0; j < 8; j++)
                if ((buf1[i] & (1 << j)) != (buf2[i] & (1 << j)))
                    hd++;
    }

    printf("%d\n", hd);
    return 0;
}
