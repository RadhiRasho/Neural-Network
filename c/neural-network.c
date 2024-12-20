#include <stdio.h>
#include <stdlib.h>

void read_mnist_images(const char *filename, unsigned char **images, int *nImages)
{
    FILE *file = fopen(filename, "rb");

    if (!file)
        exit(1);

    int temp, rows, cols;

    fread(&temp, sizeof(int), 1, file);
    fread(&nImages, sizeof(int), 1, file);
    *nImages = __builtin_bswap32(*nImages);

    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
}