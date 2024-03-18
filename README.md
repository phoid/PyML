# PyML (pie - M L) 

**Disclaimer** This project is a way for me to learn more about implementing Deep Learning programatically, not intended (for now) to be a distribution of any kind

PyML is intended to be a very low level implementation of Deep Learning, using only numpy for math. It will allow the user to create pieces of typical ANNs and put them together themselves

ex: This is an implementation of a 2 layer linear regression model:
'''
input = l.Linear(1, 6)

output = l.Linear(6, 1)
'''


While tedious, my hope is that PyML would ultimately be useful for learning and getting started in deeplearning by abstracting enough to be useable, but not so much as to mystify the inner workings on an ANN.
Additionally, It's meant to be lightweight and simple in hopes to not distract with the seemingly endless different options for layers, optimizers, regularizers, etc. 

Inspiration for this project comes from my own experience learning Deeplearning using frameworks like PyTorch and Tensorflow, which, while incredibly powerful, abstracted much of the process and left me wondering what was going on.
Although, admittedly, this is mostly my fault. 
