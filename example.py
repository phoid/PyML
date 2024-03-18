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


data, labels = sample(m=2)

epochs = 50
for i in range(epochs):
    print(f"This is epoch {i}", "\n", "-" * 80)
    for i, X in enumerate(data):

        pass1 = input.forward(X)
        pass2 = output.forward(a.relu(pass1))

        # calculating loss
        print(f"pred = {pass2 * - 1}")
        loss = lf.mse(labels[i], pass2 * -1)
        # print("A1: ", np.array([pass1]).T)
        print("loss: ", loss)

        # back propogation
        layer2grad = grad.stochastic(pass1, loss, lr=0.001)
        layer1grad = grad.stochastic(X, loss, output.weights, lr=0.001)

        print("l2grad: ", layer2grad, " l1grad: ", layer1grad)
        print("al2weights: ", output.weights, " al1weights: ", input.weights)

        output.weights = output.weights - layer2grad
        input.weights = input.weights - layer1grad
        print("bl2weights: ", output.weights, " bl1weights: ", input.weights)


# test
def printn(val):
    print("-" * 40)
    print(val)


val = 15
printn(f"input is {val}")
printn(
    f"this is the pred  {output.forward(a.relu(input.forward(np.array([val])))) * -1}"
)
printn(input.weights)
printn(output.weights)
