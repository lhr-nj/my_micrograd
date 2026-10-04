from engine import Value

# ============
# a = Value(2.0)
# b = Value(3.0)
#
# print(a)
# print(b)
#
# c = a + b
#
# print("a =", a)
# print("b =", b)
# print("c =", c)
#
# print("c._prev =", c._prev)
# print("c._op =", c._op)
#
# d = a + 10
#
# print("d =", d)
#
# print("d._prev =", d._prev)
# print("d._op =", d._op)

# ============
# a = Value(2.0)
# b = Value(3.0)
#
# c = a + b
#
# print("before:")
# print("a.grad=", a.grad)
# print("b.grad=", b.grad)
# print("c.grad=", c.grad)
#
# c.grad = 1.0
# c._backward()
#
# print("\nafter:")
# print("a.grad=", a.grad)
# print("b.grad=", b.grad)
# print("c.grad=", c.grad)

# ============
# a = Value('2.0')
#
# b = a + a
#
# b.grad = 1.0
# b._backward()
#
# print("a.grad =", a.grad)

# ============
# a = Value(2.0)
# b = Value(3.0)
#
# c = a * b
#
# c.grad = 1.0
# c._backward()
#
# print("c.data=", c.data)
# print("a.grad=", a.grad)
# print("b.grad=", b.grad)
# print("c.grad=", c.grad)

# ============
# a = Value(2.0)
#
# c = a * 4
#
# c.grad = 1.0
# c._backward()
#
# print("c.data=", c.data)
# print("a.grad=", a.grad)
# print("c.grad=", c.grad)

# ============
# a = Value(2.0)
# b = Value(3.0)
#
# c = a * b
# d = c * 4
#
# d.grad = 1.0
# d._backward()
# c._backward()
#
# print("d.grad =", d.grad)
# print("c.grad =", c.grad)
# print("a.grad =", a.grad)
# print("b.grad =", b.grad)

# ============
# a = Value(2.0)
# b = Value(3.0)
#
# c = a * b
# d = c + a
# e = d * 2
#
# e.backward()
#
# print("a.grad =", a.grad)
# print("b.grad =", b.grad)
# print("c.grad =", c.grad)
# print("d.grad =", d.grad)
# print("e.grad =", e.grad)

# ============
# a = Value(2.0)
# b = a.relu()
#
# b.backward()
#
# print("a.data =", a.data)
# print("b.data =", b.data)
#
# print("a.grad =", a.grad)
# print("b.grad =", b.grad)

# ============
# a = Value(2.0)
# b = Value(-3.0)
#
# c = a * b
# d = c + 10
# e = d.relu()
#
# e.backward()
#
# print("e.data =", e.data)
# print("a.grad =", a.grad)
# print("b.grad =", b.grad)
# print("c.grad =", c.grad)
# print("d.grad =", d.grad)

# ============
# a = Value(3.0)
# b = a ** 2
#
# b.backward()
#
# print(b.data)
# print(a.grad)

# ============
# a = Value(3.0)
# b = -a
#
# b.backward()
#
# print(b.data)
# print(a.grad)

# ============
# a = Value(5.0)
# b = Value(2.0)
#
# c = a - b
#
# c.backward()
#
# print(c.data)
# print(a.grad)
# print(b.grad)

# ============
# a = Value(6.0)
# b = Value(2.0)
#
# c = a / b
#
# c.backward()
#
# print(c.data)
# print(a.grad)
# print(b.grad)

# ============
from engine import Value

a = Value(2.0)
b = Value(4.0)

c = 2 * a + b / 2 - 1

c.backward()

print("c =", c)
print("a.grad =", a.grad)
print("b.grad =", b.grad)