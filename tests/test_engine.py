from engine import Value

def test_add_same_value():
    a = Value(2.0)

    b = a + a

    b.backward()

    assert b.data == 4.0
    assert a.grad == 2.0

def test_shared_node():
    a = Value(2.0)

    b = a * 3
    c = a * 4

    d = b + c

    d.backward()

    assert d.data == 14.0
    assert a.grad == 7.0

def test_chain_and_branch():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b
    d = c + a
    e = d * 2

    e.backward()

    assert e.data == 16.0

    assert e.grad == 1.0
    assert d.grad == 2.0
    assert c.grad == 2.0

    assert a.grad == 8.0
    assert b.grad == 4.0

def test_relu_positive():
    a = Value(2.0)

    b = a.relu()

    b.backward()

    assert b.data == 2.0
    assert a.grad == 1.0

def test_relu_negative():
    a = Value(-2.0)

    b = a.relu()

    b.backward()

    assert b.data == 0
    assert a.grad == 0.0

def test_power():
    a = Value(3.0)

    b = a ** 2

    b.backward()

    assert b.data == 9.0
    assert a.grad == 6.0

def test_division():
    a = Value(6.0)
    b = Value(2.0)

    c = a / b

    c.backward()

    assert abs(c.data - 3.0) < 1e-6
    assert abs(a.grad - 0.5) < 1e-6
    assert abs(b.grad - (-1.5)) < 1e-6
