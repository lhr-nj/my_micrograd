from nn import MLP

model = MLP(2, [4, 1])

xs = [
    [2.0, 3.0],
    [3.0, 1.0],
    [2.0, 1.0],

    [-2.0, -3.0],
    [-3.0, -1.0],
    [-2.0, -1.0]
]

ys = [
    1.0,
    1.0,
    1.0,

    -1.0,
    -1.0,
    -1.0
]

for step in range(100):

    # 1. forward
    scores = [
        model(x)
        for x in xs
    ]

    # 2. max-margin loss
    losses = [
        (1 - y * score).relu()
        for y, score in zip(ys, scores)
    ]

    data_loss = sum(losses) / len(losses)

    # 3. L2 regularization
    alpha = 1e-4


    reg_loss = alpha * sum(
        p * p
        for p in model.parameters()
    )

    loss = data_loss + reg_loss

    # 4. zero grad
    model.zero_grad()

    # 5. backward
    loss.backward()

    # 6. learning rate
    learning_rate = 0.1

    # 7. update
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # 8. accuracy
    correct = [
        (score.data > 0) == (y > 0)
        for score, y in zip(scores, ys)
    ]

    accuracy = sum(correct) / len(correct)

    print(
        "step =", step,
        "loss =", loss.data,
        "accuracy =", accuracy,
    )

print("\nFinal predictions:")

for x, y in zip(xs, ys):
    score = model(x)

    prediction = 1 if score.data > 0 else -1

    print(
        "x =", x,
        "score =", score.data,
        "prediction =", prediction,
        "target =", int(y)
    )