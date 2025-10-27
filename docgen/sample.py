# sample.py

class User:
    """Represents a system user."""

    def __init__(self, username: str, email: str):
        """Initialize user with username and email."""
        self.username = username
        self.email = email

    def greet(self) -> str:
        """Return a personalized greeting."""
        return f"Hello, {self.username}!"


def add_numbers(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
