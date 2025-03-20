import utils
import model
import math_functions as mf

def train_model(X, Y, layers, epochs, learn_rate):
    weights = utils.init_weights(layers)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = model.forward_prop(X, weights)

        loss = mf.bin_cross_entropy_loss(Y, A2)

        weights = model.back_prop(X, Y, Z1, A1, Z2, A2, weights, learn_rate)

        if epoch % 2000 == 0 and epoch > 0:
            learn_rate *= 0.9 

        #if epoch % 1000 == 0:
            #print(f"Epoch {epoch}, Loss: {loss}")
    
    return weights