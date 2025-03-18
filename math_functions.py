import numpy as np
# numpy is used for log. dont see the value in approximating and having slow network

def factorial(x: int):
    if x < 0:
        raise ValueError("X cannot be negative")
    fact = 1
    for num in range(2, x + 1):
        fact *= num
    return fact

def taylor_e(terms):
    result = 0.0
    for n in range(terms):
        result += 1 / factorial(n)
    return result

E_APPROX = taylor_e(20)

def relu(z: float) -> float:
    return max(0, z)

def sigmoid(z: float) -> float:
    return (1) / (1+(E_APPROX**(-z)))

# Derivative of sigmoid used in backprop
def sigmoid_derivative(z: float):
    return (sigmoid(z)) * (1 - sigmoid(z))

def bin_cross_entropy_loss(y_true: list, y_pred: list, epsilon=1e-15):
    loss = 0
    m = len(y_true)
    for i in range(m):
        pred = max(epsilon, min(1 - epsilon, y_pred[i]))
        loss += (y_true[i] * np.log(pred)) + ((1 - y_true[i]) * np.log(1 - pred))
    
    return - (1 / m) * loss