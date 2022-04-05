import torch
import numpy as np
class HEBatchGenerator(object):
    def __init__(self, hyperedges, labels, batch_size, device, test_generator=False):
        """Creates an instance of HyperedgeGroupBatchGenerator.
        
        Args:
            hyperedges: List(frozenset). List of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size. int. Batch size of each batch.
            test_generator: bool. Whether batch generator is test generator.
        """
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self._cursor = 0
        self.device = device
        self.test_generator = test_generator
        self.shuffle()
    
    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]
  
    def __iter__(self):
        self._cursor = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.test_generator:
            return self.next_test_batch()
        else:
            return self.next_train_batch()

    def next_train_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:] + self.hyperedges[
                :ncursor - len(self.hyperedges)]

            labels = self.labels[self._cursor:] + self.labels[
                :ncursor - len(self.labels)]
          
            self._cursor = ncursor - len(self.hyperedges)
            hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            self.shuffle()
            return hyperedges, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]
        
        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)
       
        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False

    def next_test_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:]
            labels = self.labels[self._cursor:]
            self._cursor = 0
            hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            
            return hyperedges, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]

        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)
       
        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False