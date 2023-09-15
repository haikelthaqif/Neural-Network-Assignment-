# Neural-Network-Assignment-Project


This GitHub repository contains the code for a simple Backpropagation neural network written in Python. The project's goal is to investigate the changes in learning when noise is present in inputs and/or target data. The neural network has only one unit with a sigmoid activation function.

## Project Description

For this project, we implement a 2-input single unit Backpropagation neural network with a sigmoid activation function. The system reads data from a file named "training.txt" and reads parameters from another file named "params.txt." The parameters include initial weights and learning rate. The project uses a batch size of 1, meaning weights are updated for each example, and an epoch represents one complete run through the training set.

The error measure used is the sum-of-squares-of-errors (SSE) or mean-squared-error (MSE) calculated over the entire training set. The expectation is that this error will decrease with training. During training, weights and errors are printed out after 2, 4, 8, 16, 32, etc., epochs and saved in a file named "SRNresults1.txt," where SRN should be replaced with your student registration number.

Additionally, the project includes a testing phase where the network's ability to learn predefined weights is evaluated using different sets of initial weights and training data.

## Python Code

The Python code in this repository is responsible for implementing and training the neural network. It includes functions for the sigmoid activation, its derivative, mean-squared-error loss, feedforward pass, backward pass, and the training process.

```python
# Python code here
```

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine using the following command:

   ```bash
   git clone <repository-url>
   ```

2. Make sure you have Python and required dependencies installed on your system.

3. Prepare your data files:
   - Create a "training.txt" file with your training data.
   - Create a "params.txt" file with initial weights and learning rate.

4. Run the Python code to train the neural network.

## Usage

To use this project for your own experiments:

1. Prepare your training and validation data in the required format.

2. Adjust the initial weights and learning rate in the "params.txt" file.

3. Modify any other parameters in the Python code as needed for your specific use case.

4. Run the Python script to train and evaluate the neural network.

## Results

The results of the training and validation runs, including weights and errors, will be saved in "SRNresults1.txt" and plotted on a log/log scale.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project is part of a coding assignment. Credits to the assignment creator and the educational institution.

Feel free to explore and modify this project for your own learning and research purposes!
