def hello_world():
    """Return a greeting message."""
    return "Hello, World!"

def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def multiply(self, x, y):
        """Multiply two numbers and store in history."""
        result = x * y
        self.history.append(f'{x} * {y} = {result}')
        return result
    
    def get_history(self):
        """Return calculation history."""
        return self.history