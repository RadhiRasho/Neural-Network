# From Theory to Code: Implementing a Neural Network in 200 Lines of C

- [Blog Post](https://konrad.gg/blog/posts/001.html)

## Introduction - Simplified

- Neural Networks are inspired by the human brain - dohhoy
- Have been simplified via libraries such as TensorFLow, PyTorch, or Kera
- Implementation from scratch helps build a mental (pun intended) model

## Structure of the Neural Network

Our neural network has three layers: input, hidden, and output. For the MNIST dataset, which has 28x28 pixel images, the input layer has 784 neurons (one for each pixel).

The output layer has 10 neurons, one for each digit (0-9).

- Input Layer - Accepts the `28 X 28` pixel image (flattened into 784-dimensional vectors a.k.a arrays of numbers)
- Hidden Layer: Contains 256 neurons (number is by design, powers of 2 usually preferred)
- Output Layer: Contains 10 neurons (one for each digit class, `0` to `9`)


## Processing Input Data

MNIST Dataset contains `60,000` training images and `10,000` testing images of handwritten digits from `0` to `9`. Each image is `28 X 28` pixels grayscale.

Two Methods are used to preprocess the images into a shape that aligns with the neural networks architecture.

1. `read_mnist_images` - reads images from the IDX file formats
2. `read_mnist_labels` - reads labels from the IDX file formats

### Basic Idea
- Open IDX file in binary mode
- Reac and convert header information (number of images, rows, columns)
- Allocate necessary memory and read the image pixel data and labels.

## Implementing the Neural Network Structure

### Layer Structure
- Weights: A Flattened array representing the weight matrix
- biases: An array for the biases of each neuron
- input_size: Number of inputs to the layer
- output_size: Number of neurons in the layer

All weights and biases need to be initialized at a default value, the network will later use backpropagation to learn and adjust these as necessary.

Initializtiion happens in the `init_layer` method

The Initialization to set the weights is as follows:
$$
\Large
W \sim u \left( -\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}} \right)
$$
