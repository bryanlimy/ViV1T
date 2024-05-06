from typing import Dict, Iterable

from torch.utils.data import DataLoader


class CycleDataloaders:
    """
    Cycles through dataloaders until the loader with the largest size is
    exhausted.

    Code reference: https://github.com/sinzlab/neuralpredictors/blob/9b85300ab854be1108b4bf64b0e4fa2e960760e0/neuralpredictors/training/cyclers.py#L68
    """

    def __init__(self, ds: Dict[str, DataLoader]):
        self.ds = ds
        self.max_iterations = max([len(ds) for ds in self.ds.values()])

    @staticmethod
    def cycle(iterable: Iterable):
        # see https://github.com/pytorch/pytorch/issues/23900
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def __iter__(self):
        cycles = [self.cycle(loader) for loader in self.ds.values()]
        for mouse_id, mouse_ds, _ in zip(
            self.cycle(self.ds.keys()),
            (self.cycle(cycles)),
            range(len(self.ds) * self.max_iterations),
        ):
            yield mouse_id, next(mouse_ds)

    def __len__(self):
        return len(self.ds) * self.max_iterations
