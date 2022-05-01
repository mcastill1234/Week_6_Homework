import numpy as np
from matplotlib import pyplot as plt

def softmax(x):
    """Compute softmax values for x"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Consider a NN with input x = [x1, x2].T with ReLU activation functions in f1 on all hidden neurons and softmax
# activation (f2) in the output with the following settings:
w1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
w0_1 = np.array([[-1, -1, -1, -1]]).T
w2 = np.array([[1, -1], [1, -1], [1, -1], [1, -1]])
w0_2 = np.array([[0, 2]]).T

# Problem 3.1: Output: Consider x = [3, 14].T
x = np.array([[3, 14]]).T

# Problem 3.1.A:
z1 = w1.T @ x + w0_1
a1 = np.maximum(z1, 0)
print("3.1.A: The outputs of the hidden units f1(z1) are:", a1[:, 0].tolist())

# Problem 3.1.B:
z2 = w2.T @ a1 + w0_2
a2 = softmax(z2)
print("3.1.B: The final output (a2) of the network is:", np.round(a2[:, 0], decimals=2).tolist())


# Problem 3.2.C:
xs = np.array([[0.5, 0, -3], [0.5, 2, 0.5]])
z3 = w1.T @ xs + w0_1
a3 = np.maximum(z3,0)
print("3.1.C: The output of f1(z1) for the given inputs is:", a3.tolist())

# Problem 3.2: Unit decision boundaries
# "Hint: You should draw a diagram of the decision boundaries for each unit in the xx-space..."
figure1 = plt.figure(1)
plt.xlabel('x1 feature')
plt.ylabel('x2 feature')
x_values1 = [1, 1, 1, 1]
y_values1 = [-1, -0.5 , 0.5, 1]
plt.plot(x_values1, y_values1, 'r--')

x_values2 = [-1, -0.5 , 0.5, 1]
y_values2 = [1, 1, 1, 1]
plt.plot(x_values2, y_values2, 'r--')

x_values3 = [-1, -1, -1, -1]
y_values3 = [-1, -0.5 , 0.5, 1]
plt.plot(x_values3, y_values3, 'r--')

x_values4 = [-1, -0.5 , 0.5, 1]
y_values4 = [-1, -1, -1, -1]
plt.plot(x_values4, y_values4, 'r--')

x1_pos = [1, 2, 2, 1, -1, -2, -2, -1]
x2_pos = [2, 1, -1, -2, -2, -1, 1, 2]
x1_neg = [0.5, 0.5, -0.5, -0.5]
x2_neg = [0.5, -0.5, -0.5, 0.5]
plt.scatter(x1_pos, x2_pos, marker='+', c='green', s=200, linewidths=2)
plt.scatter(x1_neg, x2_neg, marker='x', c='red', s=200, linewidths=2)

# Problem 3.2.A:
print("3.2.A: The shape of the decision boundary for a single unit is a line")

# Problem 3.2.B:
print("3.2.B: 4 points laying in the decision boundary of the first unit are: ", [x_values1, y_values1])

# Problem 3.2.C:
x_vals = np.array([[0.5, 0, -3], [0.5, 2, 0.5]])
z4 = w1.T @ x_vals + w0_1
f1z4 = np.maximum(z4, 0)
print("3.2.C: The outputs of the hidden units f1(z1) for the given inputs are:  ", f1z4.tolist())

# Problem 3.3 Network outputs: Answered directly in online course.
plt.grid()
plt.show()