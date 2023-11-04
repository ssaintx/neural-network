import random
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset

images, labels = load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))

bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 3
learning_rate = 0.01

for epoch in range(epochs):
    e_loss = 0
    e_correct = 0
    print("Epoch:", epoch + 1)
    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # forward propagation (to hidden layer)
        hidden_raw = bias_input_to_hidden + np.dot(weights_input_to_hidden, image)
        hidden = 1 / (1 + np.exp(-hidden_raw))

        # forward propagation (to output layer)
        output_raw = bias_hidden_to_output + np.dot(weights_hidden_to_output, hidden)
        output = 1 / (1 + np.exp(-output_raw))

        # loss / error calculation
        e_loss += np.sum((output - label) ** 2)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # backpropagation (output layer)
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * np.dot(delta_output, np.transpose(hidden))
        bias_hidden_to_output += -learning_rate * delta_output

        # backpropagation (hidden layer)
        delta_hidden = np.dot(np.transpose(weights_hidden_to_output), delta_output) * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * np.dot(delta_hidden, np.transpose(image))
        bias_input_to_hidden += -learning_rate * delta_hidden

    # compute loss and accuracy at each epoch
    e_loss /= len(labels)
    e_correct /= len(labels)

    # print some debug info between epochs
    print("Loss:", round(e_loss * 100, 3), "%")
    print("Accuracy:", round(e_correct * 100, 3), "%")
    print()

# CHECK
test_image = random.choice(images)
image = np.reshape(test_image, (-1, 1))

# forward propagation (to hidden layer)
hidden_raw = bias_input_to_hidden + np.dot(weights_input_to_hidden, image)
hidden = 1 / (1 + np.exp(-hidden_raw))

# forward propagation (to output layer)
output_raw = bias_hidden_to_output + np.dot(weights_hidden_to_output, hidden)
output = 1 / (1 + np.exp(-output_raw))

# show the image, and test it
plt.imshow(np.reshape(test_image, (28, 28)), cmap="gray")
plt.title("NN suggests the number is: " + str(np.argmax(output)))
plt.show()