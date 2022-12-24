import numpy as np
import nnfs
from nnfs.datasets import spiral_data

inputs = np.array([[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]])

class Layer_Dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1 , n_neurons))
    def forward(self , inputs):
        self.outputs = np.dot(inputs , self.weights) + self.biases
    
class Activation_ReLU:
    def forward(self , inputs):
        self.output = np.maximum(0 , inputs)

class Activation_Softmax:
    def forward(self , inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis = 1 , keepdims = True))
        probabilities = exp_values / np.sum(exp_values , axis = 1 , keepdims = True)
        self.output = probabilities

class Loss:
    def calculate(self , output , y):
        sample_loss = self.forward(output , y)
        data_loss = np.mean(sample_loss)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self , y_pred , y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred , 1e-7 , 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples) , y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true , axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Accuracy():
    def calculate(self , output , y_true):
        predictions = np.argmax(output , axis = 1)
        accuracy = np.mean(predictions == y_true)
        return accuracy

# Consider a Dataset with 100 features and 3 classes 'Cat' , 'Dog' , 'Human'
X , y = spiral_data(samples= 100 , classes = 3)

dense1 = Layer_Dense(2 , 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3 , 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.outputs)
dense2.forward(activation1.output)
activation2.forward(dense2.outputs)

# print("\nLayer 1 : " , layer1.outputs)
# print("\nLayer 2 : " , layer1.outputs)
# print("\ReLU 1 : ",activation1.output)
print("\nSoftmax : ",activation2.output)

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output , y)
print("\nLoss : " , loss)

# accuracy_function = Accuracy()
# accuracy = accuracy_function.calculate(activation2.output , y)
# print('\nAccuracy : ' , accuracy)

lowest_loss = loss
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.output)
    activation2.forward(dense2.outputs)

    loss = loss_function.calculate(activation2.output , y)

    predictions = np.argmax(activation2.output , axis = 1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print("New set of weights found , iteration : " , iteration , ' Loss : ' , loss , ' Accuracy : ' , accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

print("Iterations Complete \n Loss : " , lowest_loss , ' Accuracy : ' , accuracy)