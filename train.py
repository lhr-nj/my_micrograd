from nn import MLP

model = MLP(2, [1])

xs = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]

ys = [
    0.0,
    1.0,
    1.0,
    2.0,
]

learning_rate = 0.1

for step in range(100):

    # 1. forward
    predictions = [
        model(x)
        for x in xs
    ]

    # 2. loss
    losses = [
        (pred - target) ** 2
        for pred, target in zip(predictions, ys)
    ]

    loss = sum(losses) / len(losses)

    # 3. clear gradients
    model.zero_grad()

    # 4. backward
    loss.backward()

    # 5. update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    print(
        "step =", step,
        "loss =", loss.data
    )

print("\nFinal predictions:")

for x, target in zip(xs, ys):
    pred = model(x)

    print(
        "x =", x,
        "prediction =", pred.data,
        "target =", target
    )

print("\nLearned parameters:")

for p in model.parameters():
    print(p.data)