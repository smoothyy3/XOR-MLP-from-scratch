import math_functions as mf

def forward_prop(X, weights):
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]

    Z1 = mf.matrix_mult(X, W1) # matrix mult
    for i in range(len(Z1)): # loop over all rows -> samples in batch
        for j in range(len(b1)): # loop over all columns -> neurons in hidden layer
            Z1[i][j] += b1[j] # add bias to neuron output

    A1 = [[0] * len(Z1[0]) for _ in range(len(Z1))]
    for i in range(len(Z1)):
        for j in range(len(Z1[0])):
            A1[i][j] = mf.relu(Z1[i][j])

    Z2 = mf.matrix_mult(A1, W2)
    for i in range(len(Z2)):
        for j in range(len(b2)):
            Z2[i][j] += b2[j]

    A2 = [[0] * len(Z2[0]) for _ in range(len(Z2))]
    for i in range(len(Z2)):
        for j in range(len(Z2[0])):
            A2[i][j] = mf.sigmoid(Z2[i][j])

    return Z1, A1, Z2, A2

def back_prop(X, Y, Z1, A1, Z2, A2, weights, learning_rate):
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]

    dA2 = []
    for i in range(len(A2)): # over training samples
        row = []
        for j in range(len(A2[0])): # over output neurons
            row.append(A2[i][j] - Y[i][j])
        dA2.append(row)

    dZ2 = dA2

    dW2 = mf.matrix_mult(mf.transpose(A1), dZ2)

    db2 = [0] * len(dZ2[0])
    for i in range(len(dZ2)):
        for j in range(len(dZ2[0])):
            db2[j] += dZ2[i][j]

    dA1 = mf.matrix_mult(dZ2, mf.transpose(W2))

    dZ1 = []
    for i in range(len(Z1)):
        row = []
        for j in range(len(Z1[0])):
            row.append(dA1[i][j] * (1 if Z1[i][j] > 0 else 0))
        dZ1.append(row)

    dW1 = mf.matrix_mult(mf.transpose(X), dZ1)

    db1 = [0] * len(dZ1[0])
    for i in range(len(dZ1)):
        for j in range(len(dZ1[0])):
            db1[j] += dZ1[i][j]

    # update weights with grad descent
    for i in range(len(W1)):
        for j in range(len(W1[0])):
            W1[i][j] -= learning_rate * dW1[i][j]

    for i in range(len(W2)):
        for j in range(len(W2[0])):
            W2[i][j] -= learning_rate * dW2[i][j]

    for j in range(len(b1)):
        b1[j] -= learning_rate * db1[j]

    for j in range(len(b2)):
        b2[j] -= learning_rate * db2[j]

    # return updatet weights
    return weights