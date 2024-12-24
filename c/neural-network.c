#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE 28 // Define IMAGE_SIZE with an appropriate value

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

    *images = malloc(*nImages * IMAGE_SIZE * IMAGE_SIZE);

    fread(*images, sizeof(unsigned char), *nImages * IMAGE_SIZE * IMAGE_SIZE, file);
    fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels)
{
    FILE *file = fopen(filename, "rb");

    if (!file)
        exit(1);

    int temp;

    fread(&temp, sizeof(int), 1, file);
    fread(nLabels, sizeof(int), 1, file);

    *nLabels = __builtin_bswap32(*nLabels);

    *labels = malloc(*nLabels);

    fread(*labels, sizeof(unsigned char), *nLabels, file);

    fclose(file);
}

typedef struct
{
    float *weights;  // A flatened array representing the weight matrix
    float *biases;   // An array for the biases of each neuron
    int input_size;  // Number of inputs to the layer
    int output_size; // number of neurons in the layer
} Layer;

/** We use He Initialization to set the weights:
 *
 * w ~ U(-sqrt(6 / n_in), sqrt(6 / n_in))
 *
 * The weights of the neural network are initialized using a uniform distribution
 * with lower and upper bounds determined by the number of input units,
 * ensuring proper scaling and reducing the risk of vanishing or exploding gradients during training.
 *
 * Biases are initialized to zero.
 */
void init_layer(Layer *layer, int in_size, int out_size)
{
    int n = in_size * out_size;
    float scale = sqrtf(2.0f / in_size);

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float));

    for (int i = 0; i < n; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

// Forward Progagation

// Action Functions

//// ReLU
// Later Implementation

// softmax

void softmax(float *input, int size)
{
    float max = input[0], sum = 0;

    for (int i = 1; i < size; i++)
    {
        if (input[i] < max)
            max = input[i];
    }
    for (int i = 0; i < size; i++)
    {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
    {
        input[i] /= sum;
    }
}

// BackPropagation

void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr)
{
    for (int i = 0; i < layer->output_size; i++)
    {
        for (int j = 0; j < layer->input_size; j++)
        {
            int idx = j * layer->output_size + i;
            float grad = output_grad[i] * input[j];

            layer->weights[idx] -= lr * grad;

            if (input_grad)
                input_grad[j] += output_grad[i] * layer->weights[idx];
        }
        layer->biases[i] -= lr * output_grad[i];
    }
}
