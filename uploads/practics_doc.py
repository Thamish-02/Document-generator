# Fibonacci Series in Python

# Function to generate Fibonacci series up to n terms
def fibonacci(n):
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence

# Take user input
n = 10

# Generate and display Fibonacci series
print("Fibonacci Series:")
print(fibonacci(n))
