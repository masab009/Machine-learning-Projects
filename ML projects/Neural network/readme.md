# Neural Network Implementation in C++

This project implements a **feedforward neural network** in C++ for tasks like stock price prediction. It features backpropagation, gradient descent, and efficient matrix operations, making it a foundation for exploring machine learning concepts.

## Features

- **Matrix Operations**:
  - Transposition, dot product, and Z-score normalization.
- **Activation Functions**:
  - ReLU, Sigmoid, and Linear.
- **Data Handling**:
  - Reads data from CSV files.
  - Splits data into training and testing sets.
- **Gradient Descent**:
  - Backpropagation for weight and bias updates.
- **Customizable Architecture**:
  - Flexible layer configuration with specified activation functions.
- **Multithreading**:
  - Parallel computation via OpenMP.

## Getting Started

### Prerequisites
- **OpenMP** for multithreading.
- Standard C++ libraries (`<vector>`, `<fstream>`, `<cmath>`, etc.).

### Setup and Compilation
1. Clone or download the repository.
2. Modify `main.cpp` to set:
   - Dataset file path.
   - Neural network architecture.
3. Compile the program:
   ```bash
   g++ -o neural_net main.cpp -fopenmp
