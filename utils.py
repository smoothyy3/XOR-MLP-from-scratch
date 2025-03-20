import random

def init_weights(layers: list) -> dict:
    weights = {}

    for l in range(1, len(layers)):
        prev_layer = layers[l - 1]
        current_layer = layers[l]

        bound = 1 / (prev_layer ** 0.5)

        W = [[random.uniform(-bound, bound) for _ in range(current_layer)] for _ in range(prev_layer)]

        b = [0.1] * current_layer

        weights[f"W{l}"] = W
        weights[f"b{l}"] = b
    
    return weights