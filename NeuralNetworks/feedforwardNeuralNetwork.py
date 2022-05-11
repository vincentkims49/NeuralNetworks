import numpy as num
from sklearn import datasets

num.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

inputlayer_dimensionality = 4 
outputlayer_dimensionality = 3 
hiddenlayer_dimensionality = 6 

a1 = num.random.randn(inputlayer_dimensionality, hiddenlayer_dimensionality)
c1 = num.zeros((1, hiddenlayer_dimensionality))

a2= num.random.randn(hiddenlayer_dimensionality, hiddenlayer_dimensionality)
c2 = num.zeros((1, hiddenlayer_dimensionality))

a3= num.random.randn(hiddenlayer_dimensionality, outputlayer_dimensionality)
c3 = num.zeros((1, outputlayer_dimensionality))

d1 = X.dot(a1) + c1
q1 = num.tanh(z1)

d2 = q1.dot(a2) + c2
q2 = num.tanh(z2)

d3 = q2.dot(a3) + c3

probs = num.exp(d3) / num.sum(num.exp(d3), axis=1, keepdims=True)
