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

        pass1 = input.forward(data[ind])
        pass2 = output.forward(a.relu(pass1))

        # calculating loss
        # print(f"pred = {pass2 * -1}")
        loss = lf.mse(labels[ind], pass2)
        # print("A1: ", np.array([pass1]).T)
        print("loss: ", loss)

        # back propogation
        layer2grad = grad.stochastic(pass1, loss, lr=0.0001)
        layer1grad = grad.stochastic(data[ind], loss, output.weights, lr=0.0001)

        print("l2grad: ", layer2grad, " l1grad: ", layer1grad)
        print("al2weights: ", output.weights, " al1weights: ", input.weights)

        output.weights = output.weights - layer2grad
        input.weights = input.weights - layer1grad
        # print("bl2weights: ", output.weights, " bl1weights: ", input.weights)


# test
def printn(string):
    print("-" * 40)
    print(string)


val = 15
printn(f"input is {val}")
print(f"this is the pred  {output.forward(a.relu(input.forward(np.array([val]))))}")
