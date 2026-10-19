import nnfs
from nnfs.datasets import spiral_data
import denselayer
from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
import numpy as np
from loss import Loss_CategoricalCrossEntropy
X, y = spiral_data(samples=100, classes = 3)

dense1 = denselayer.Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = denselayer.Layer_Dense(3,3)
activation2 = Activation_Softmax()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss_function = Loss_CategoricalCrossEntropy()
print(activation2.output[:5])
loss = loss_function.calculate(activation2.output, y)
print('loss:', loss)
predictions = np.argmax(activation2.output, axis = 1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print("acc:", accuracy)