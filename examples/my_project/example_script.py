"""
A simple example script to demonstrate the AI documentation generator.
"""

class DataProcessor:
    """A class for processing numerical data."""
    
    def __init__(self, data):
        """Initialize with a list of numbers.
        
        Args:
            data (list): A list of numerical values
        """
        self.data = data
        self.processed = False
    
    def calculate_mean(self):
        """Calculate the mean of the data.
        
        Returns:
            float: The mean value of the data
        """
        if not self.data:
            return 0
        return sum(self.data) / len(self.data)
    
    def find_max(self):
        """Find the maximum value in the data.
        
        Returns:
            float: The maximum value
        """
        return max(self.data) if self.data else 0

def process_file(filename):
    """Process a file containing numerical data.
    
    Args:
        filename (str): Path to the file
        
    Returns:
        dict: A dictionary with processing results
    """
    try:
        with open(filename, 'r') as f:
            numbers = [float(line.strip()) for line in f if line.strip()]
        
        processor = DataProcessor(numbers)
        return {
            'mean': processor.calculate_mean(),
            'max': processor.find_max(),
            'count': len(numbers)
        }
    except FileNotFoundError:
        print(f"File {filename} not found")
        return {}
    except ValueError:
        print(f"Invalid data in file {filename}")
        return {}

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 3, 4, 5, 10, 15, 20]
    processor = DataProcessor(sample_data)
    
    print(f"Mean: {processor.calculate_mean()}")
    print(f"Max: {processor.find_max()}")