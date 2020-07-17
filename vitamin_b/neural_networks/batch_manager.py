import numpy as np



class SequentialIndexer(object):
    """
    goes through data in order accoriding to batch size. can wrap around
    """
    def __init__(self, batch_size, total_points):
        self.counter = 0
        self.batch_size = batch_size
        self.total_points = total_points

    def next_indices(self):
        """
        Written so that if total_points
        changes this will still work
        """
        first_index = self.counter
        last_index = self.counter + self.batch_size
        self.counter = last_index % self.total_points
        return np.arange(first_index, last_index) % self.total_points

