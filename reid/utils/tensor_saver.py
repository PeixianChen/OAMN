import os
import pickle
from collections import defaultdict
import torch


class Saver(object):
    """
    Usage
    >>> saver = Saver('./tensors')
    >>> saver(outputs, labels, 'xxx')
    >>> saver(outputs, labels, 'yyy')
    >>> saver.save()
    """

    def __init__(self, path, n):
        self.data = defaultdict(lambda: defaultdict(list)) # {tag1: {label1: [tensor], label2: [tensor]}, tag2: {...}}
        self.path, self.n = path, n
        if not os.path.exists(path):
            os.mkdir(path)

    def __call__(self, tensor_batch, label_batch, tag):
        tensor_batch = tensor_batch.detach().cpu().numpy()
        label_batch = label_batch.detach().cpu().numpy()
        data = self.data[tag]

        for tensor, label in zip(tensor_batch, label_batch):
            if len(data[int(label)]) <= self.n:
                data[int(label)].append(tensor)

        summary = '  '.join('[%s] %d (%d)' % (tag, len(data), sum(map(len, data.values()))) for tag, data in self.data.items())
        print('Tensor Saver:', summary)

    def save(self):
        for tag, data in self.data.items():
            filename = os.path.join(self.path, '%s.pkl' % tag)
            with open(filename, 'wb') as f:
                pickle.dump(data, f)                                                                                                                                                                            
                print('Dump `%s` with %d tensors.' % (tag, len(data)))
        self.data = defaultdict(lambda: defaultdict(list))

SAVER = Saver('./pig-tensors', 20)
