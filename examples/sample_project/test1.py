"""
Sample Python test file for docgen example project.

This is a simple Python script that demonstrates basic functionality
for documentation generation purposes.
"""

def hello_world():
    """
    A simple function that returns a greeting message.
    
    Returns:
        str: A greeting message
    """
    return "Hello, World!"


def add_numbers(a, b):
    """
    Add two numbers together.
    
    Args:
        a (int): First number
        b (int): Second number
    
    Returns:
        int: Sum of the two numbers
    """
    return a + b


class Calculator:
    """
    A simple calculator class for basic arithmetic operations.
    """
    
    def __init__(self):
        """Initialize the calculator."""
        self.history = []
    
    def multiply(self, x, y):
        """
        Multiply two numbers.
        
        Args:
            x (float): First number
            y (float): Second number
        
        Returns:
            float: Product of the two numbers
        """
        result = x * y
        self.history.append(f"{x} * {y} = {result}")
        return result
    
    def get_history(self):
        """
        Get the calculation history.
        
        Returns:
            list: List of calculation strings
        """
        return self.history


if __name__ == "__main__":
    print(hello_world())
    print(f"2 + 3 = {add_numbers(2, 3)}")
    
    calc = Calculator()
    print(f"4 * 5 = {calc.multiply(4, 5)}")
    print("History:", calc.get_history())