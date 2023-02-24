from typing import Callable, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .datasets.selfsup_fractaldata import SelfSupMultiFractalDataset


class SelfSupMultiFractalDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = 'data/',
            data_file: str = None,
            batch_size: int = 64,
            num_workers: int = 4,
            pin_memory: bool = True,
            size: int = 256,
            num_systems: int = 1000,
            num_class: int = 1000,
            per_class: int = 1000,
            generator: Optional[Callable] = None,
            period: int = 2,
            num_augs: int = 2,
            transform=None, **kwargs):

        super().__init__()

        assert(generator is not None)
        self.data_dir  = data_dir
        self.data_file = data_file

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.period = period
        self.num_augs = num_augs

        self.num_systems = num_systems
        self.num_class = num_class
        self.per_class = per_class
        self.generator = generator
        self.transform = transform
        self.dims = (3, size, size)

        self.data_train = SelfSupMultiFractalDataset(
                param_file=self.data_dir+self.data_file,
                num_systems=self.num_systems,
                num_class=self.num_class,
                per_class=self.per_class,
                generator=self.generator,
                period=self.period, num_augs=self.num_augs,
                transform=self.transform)
        return

    def train_dataloader(self, train_sampler=None):
        return DataLoader(
                dataset=self.data_train, batch_size=self.batch_size,
                shuffle=(train_sampler is None),
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                sampler=train_sampler, drop_last=True)
