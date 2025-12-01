import activations as a
import layers as l
import lossfunctions as lf
import regularizers as r
import numpy as np
import gradients as grad

input = l.Linear(1, 6)

output = l.Linear(6, 1)


def sample(n=10, m=1):
    X = np.zeros((n, 1))
    y = np.zeros((n, 1))
    for i in range(1, n):
        X[i][0] = i
        y[i][0] = i * m
    return X, y


data, labels = sample(n=100, m=2)
norm_data = data / data.max()
epochs = 20

shuffled = np.random.choice(len(data), size=len(data), replace=False)
for i in range(epochs):
    print(f"This is epoch {i}", "\n", "-" * 80)
    for ind in shuffled:

        # Forward pass
        z1 = input.forward(norm_data[ind])
        a1 = a.relu(z1)
        z2 = output.forward(a1)
        a2 = z2  # Linear activation for output

        # calculating loss
        loss = lf.mse(labels[ind], a2)
        print("loss: ", loss)

        # Backward pass
        # Output layer error (delta2)
        delta2 = lf.mse_prime(labels[ind], a2)

        # Hidden layer error (delta1)
        # Backpropagate error through weights and activation derivative
        delta1 = np.dot(delta2, output.weights) * a.relu_deriv(z1)

        # Calculate gradients
        layer2_update = grad.calc_gradient(a1, delta2, lr=0.0001)
        layer1_update = grad.calc_gradient(norm_data[ind], delta1, lr=0.0001)

        # Update weights
        output.weights -= layer2_update
        input.weights -= layer1_update

        # Update biases
        if hasattr(output, 'bias'):
            output.bias -= 0.0001 * delta2
        if hasattr(input, 'bias'):
            input.bias -= 0.0001 * delta1


# test
def printn(string):
    print("-" * 40)
    print(string)


val = 15
printn(f"input is {val}")
# Forward pass for test
z1 = input.forward(np.array([val / data.max()]))
a1 = a.relu(z1)
pred = output.forward(a1)
print(f"this is the pred  {pred}")
