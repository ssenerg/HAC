from HAC.heaps.max_fibonacci import MaxFibonacciHeap
a = MaxFibonacciHeap()
for i in range(0, 100000):
    a.push(i, i + 1)
print(a.find_max.key)
for i in range(0, 100000):
    a.pop()