from engine import Value
from nn import Neuron, Layer, MLP

# ============
# n = Neuron(3)
#
# x = [
#     Value(1.0),
#     Value(2.0),
#     Value(3.0)
# ]
#
# y = n(x)
#
# print("output =", y)
#
# print("\nparameters:")
# for p in n.parameters():
#     print(p)
#
# print("\nparameter count =", len(n.parameters()))
#
# y.backward()
#
# print("\ngradients:")
#
# for p in n.parameters():
#     print(p.grad)

# ============
# layer = Layer(3, 2)
#
# x = [
#     Value(1.0),
#     Value(2.0),
#     Value(3.0)
# ]
#
# out = layer(x)
#
# print("output:")
# print(out)
#
# print("\nparameter count:")
# print(len(layer.parameters()))
#
# loss = out[0] + out[1]
#
# loss.backward()
#
# for p in layer.parameters():
#     print(p.grad)

# ============
# model = MLP(3, [4, 4, 1])
#
# x = [
#     Value(1.0),
#     Value(2.0),
#     Value(3.0)
# ]
#
# y = model(x)
#
# print("output =", y)
# print("parameter count =", len(model.parameters()))
#
# y.backward()
#
# print("\nfirst few gradients:")
#
# for p in model.parameters()[:5]:
#     print(p.grad)

# ============
# x = [1.0, 2.0]
#
# for _ in range(10):
#     model = MLP(2, [4, 1])
#     y = model(x)
#     print(y.data)

# ============
model = MLP(2, [4, 1])

x = [1.0, 2.0]

y = model(x)
y.backward()

print("before zero_grad:")

for p in model.parameters()[:5]:
    print(p.grad)

model.zero_grad()
print("\nafter zero_grad:")

for p in model.parameters()[:5]:
    print(p.grad)