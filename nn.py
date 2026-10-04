import random

from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(nin)
        ]

        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = self.b

        for wi, xi in zip(self.w, x):
            act = act + wi * xi

        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [
            Neuron(nin, **kwargs)
            for _ in range(nout)
        ]

    def __call__(self, x):
        outs = [
            neuron(x)
            for neuron in self.neurons
        ]

        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params

class MLP(Module):

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts

        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                nonlin=(i != len(nouts) - 1)
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params