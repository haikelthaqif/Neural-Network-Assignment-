# Name: Haikel Thaqif  

import numpy as np
import time

start = time.time()



# Activation function (sigmoid function)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the activation function (sigmoid function)
def sigmoid_derivative(x):
    return x * (1 - x)


# Mean-squared-error loss function
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# Feedforward pass function
def feedforward(inputs, weights):
    g = np.dot(inputs, weights)  #inputs*weights
    return sigmoid(g)


# Backward pass function
def backward_pass(inputs, weights, targets, actualOutput, learning_rate):
    error = targets - actualOutput
    d_actualOutput = error * sigmoid_derivative(actualOutput)
    d_weights = np.dot(inputs.T, d_actualOutput)
    weights += learning_rate * d_weights
    return weights


# Define the training process
def train(inputs, weights, targets, learning_rate, epochs):
    errors = []

    with open("200553917results1.txt", "a") as file:

        file.write("Inputs: {} || Weights: {} || Targets: {}\n".format( inputs[0],weights, targets[tan]))
        file.write("**********************************************************************************************************\n")

        for epoch in range(epochs):
            actualOutput = feedforward(inputs, weights)  # feedfoward results
            weights = backward_pass(inputs, weights, targets, actualOutput, learning_rate)
            
            
            # print the weights and error every 2^x epochs
            if (epoch+1) in [2**x for x in range(15)]: 
                error = mean_squared_error(actualOutput, targets)
                errors.append(error)
                
                
                print("Epoch:", epoch + 1, "|| Weights:", weights, "|| Error:", error)

                file.write("Epoch: {} || Weights: {} || Error: {}\n".format(epoch + 1, weights, error))
                
        file.write("\n")
        file.write("\n")
        return weights, errors


# Set the learning rate
learning_rate = 0.1

# Set the number of epochs
epochs = (2**15)

# Set Noise levels
inputNoise = 1.9
targetNoise = 0.9



tan=0

#********************************************************** Start  Training ********************************************************************#
# open training.txt file
with open("training.txt", "r") as f1:
    
    print("Start Training")
    with open("200553917results1.txt", "a") as file:  #writing to 200553917results1.txt
        file.write("\n")
        file.write("////////////////////////////////////////// Training Data \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ \n")
        file.write("\n")

 # loop through each line in the file
    for line in f1:
            # split the line into x, y, and target numbers
            x, y, *targets = map(float, line.strip().split(","))

            # add noise levels into inputs and targets
            x += inputNoise
            y += inputNoise
            for i in range(len(targets)):
                targets[i] += targetNoise

            inputs = np.array([[1, x, y], [1, x, y], [1, x, y], [1, x, y], [1, x, y]])
            
            # open params.txt file
            with open("params.txt", "r") as f2:
                # loop through each line in the file
                for line in f2:
                    
                    # split the line into weights
                    weights = list(map(float, line.strip().split(",")))
                    
                    print("Inputs:", "[", x,",",y,"]", " Targets:[", targets[tan],"]", "Weights:",weights)
                    
                    tan+=1
                    if tan == 5:
                        tan=0
                    print("******************************************************************************************")
                    
                    # Train the model
                    weights, errors = train(inputs, weights, targets, learning_rate, epochs)
                    

                    
                    print("\n")
    print(" Training Run Ended")
    print("\n")



#************************************************* Start Validation Training **************************************************************#
# open validation.txt file
with open("validation.txt", "r") as f1:
    
    print("Start Validation Training")  
    with open("200553917results1.txt", "a") as file:  #writing to 200553917results1.txt
        file.write("\n")
        file.write("\n")
        file.write("////////////////////////////////////////// Validation Data \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ \n")
        file.write("\n")


    # loop through each line in the file
    for line in f1:
        # split the line into x, y, and target numbers
        x, y, *targets = map(float, line.strip().split(","))

        # add noise levels into inputs and targets
        x += inputNoise
        y += inputNoise
        for i in range(len(targets)):
            targets[i] += targetNoise


        inputs = np.array([[1, x, y], [1, x, y], [1, x, y], [1, x, y], [1, x, y]])
        
        # open params.txt file
        with open("params.txt", "r") as f2:
            # loop through each line in the file
            for line in f2:
                
                # split the line into weights
                weights = list(map(float, line.strip().split(",")))
                
                print("Inputs:", "[", x,",",y,"]", " Targets:[", targets[tan],"]", "Weights:",weights)
                
                tan+=1
                if tan == 5:
                    tan=0
                print("******************************************************************************************")
                
                # Train the model
                weights, errors = train(inputs, weights, targets, learning_rate, epochs)
                

                
                print("\n")
    print("\n")
    print(" Validation Run Ended")
    end = time.time()
    print(end - start)
    print("\n")
                