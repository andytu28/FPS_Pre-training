def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

import os 
from fractal_datamodule.selfsup_fractal_datamodule import SelfSupMultiFractalDataModule
from fractal_datamodule.datasets.selfsup_fractaldata import SelfSupGenerator
from torchvision import transforms
from torchvision.utils import save_image


if __name__ == '__main__':
    image_level_augs = transforms.Compose([
    		transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])])

    image_size = 224
    num_systems = 100000
    num_class = 100000
    per_class = int(1000000 // num_class)  # total_images // num_class
    max_num_objs = 5
    color_mode = 'random'
    gen_num_augs = 2
    background = False

    batch_size = 256
    workers = 8

    generator = SelfSupGenerator(
            size=image_size,
            cache_size_per_class=4,
            max_num_cached_class=2048,
            niter=100000,
            num_class=num_class,
            num_objects=(2, max_num_objs),
            color_mode=color_mode,
            num_augs=gen_num_augs,
            background=background)

    datamodule = SelfSupMultiFractalDataModule(
            data_dir='fractal_code/',
            data_file='ifs-1mil.pkl',
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            size=image_size,
            num_systems=num_systems,
            num_class=num_class,
            per_class=per_class,
            generator=generator,
            period=2, num_augs=2,
            transform=image_level_augs)

    print(f'Dataset Length: {len(datamodule.data_train)}')

    os.makedirs('FPS_SAMPLES/', exist_ok=True)
    dataloader = datamodule.train_dataloader()
    for index, (x1, x2, y) in enumerate(dataloader):
        print(index+1, x1.shape, x2.shape, y.shape)
        num_class = sum(y[0]).item()
        save_image(x1[0], f'FPS_SAMPLES/fpsidx{index+1}_ncls{num_class}_aug1.png')
        save_image(x2[0], f'FPS_SAMPLES/fpsidx{index+1}_ncls{num_class}_aug2.png')

        if index >= 49:
            break
