import numpy as np


# Problem 2.C:
def softmax(x):
    """Compute softmax values for x"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1, 1]]).T
y = np.array([[0, 1, 0]]).T
z = w.T @ x
a = softmax(z)
grad_NLL = x * (a - y).T
print("2.C: The gradient matrix of NLL is:", np.round(grad_NLL, decimals=3).tolist())

# Problem 2.D:
print("2.D: The predicted probability that x is in class 1 is:", np.round(a[1, 0], decimals=3))

# Problem 2.E:
w_next = w - 0.5 * grad_NLL
print("2.E: The values of W after one step of GD is:", np.round(w_next, decimals=3).tolist())

# Problem 2.F:
z_next = w_next.T @ x
a_next = softmax(z_next)
print("2.F: The predicted probability that x is in class 1 after one step of GD is:", np.round(a_next[1, 0], decimals=3))

