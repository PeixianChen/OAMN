import torch

class CudaUsage(object):

    NUM_DEVICES = torch.cuda.device_count()
    ENABLED = True

    def __init__(self):
        self.usage = [0] * self.NUM_DEVICES

    def __call__(self, prompt=None):
        if not self.ENABLED:
            return

        if prompt is not None:
            print(prompt, end=' ')

        current = list(map(torch.cuda.memory_allocated, range(self.NUM_DEVICES)))
        for i, new in enumerate(current):
            self.usage[i], update = new, new - self.usage[i]
            print(f'[{i}] {new >> 20:6d} ({update >> 20:+d}) MB', end='\t')
        print()


USAGE = CudaUsage()
