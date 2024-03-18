# PyML (pie - M L) 

**Disclaimer** This project is a way for me to learn more about implementing Deep Learning programatically, not intended (for now) to be a distribution of any kind

PyML is intended to be a very low level implementation of Deep Learning, using only numpy for math. It will allow the user to create pieces of typical ANNs and put them together themselves

ex: This is an implementation of a 2 layer linear regression model for 1 input:
```
input = layers.Linear(1, 6)
output = layers.Linear(6, 1)

data, labels = sample()

for i in range(epochs):
    
    for i, X in enumerate(data):
        pass1 = input.forward(X)
        pass2 = output.forward(activations.relu(pass1))

        loss = lossfunctions.mse(labels[i], pass2 * -1)

        # back propogation
        layer2grad = gradients.stochastic(pass1, loss, lr=0.001)
        layer1grad = gradients.stochastic(X, loss, output.weights, lr=0.001)

        output.weights = output.weights - layer2grad
        input.weights = input.weights - layer1grad
```
Model achieves > 99% accuracy on y = mx + b
It's clear that this can become very messy very quickly with more complex models.

While tedious, my hope is that PyML would ultimately be useful for learning and getting started in deeplearning by abstracting enough to be useable, but not so much as to mystify the inner workings on an ANN.
Being lightweight and simple, it forces a the user to be more familiar with the properties of ANNs and the math in order to actually get good results. Knowledge that helped me leverage larger frameworks 
like pytorch to their fullest extent.

Inspiration for this project comes from my own experience learning Deeplearning using frameworks like PyTorch and Tensorflow, which, while incredibly powerful, abstracted much of the process and left me wondering what was going on.
Although, admittedly, this is mostly my fault in most cases. 
