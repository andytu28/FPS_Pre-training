from functools import partial
import random
import pickle
from typing import Callable, Optional, Tuple, Union
import warnings

from cv2 import GaussianBlur, resize, INTER_LINEAR
import numpy as np
import torch
import torchvision

from .fractals import diamondsquare, ifs
from .generator import _GeneratorBase



class SelfSupMultiFractalDataset(object):
    def __init__(
            self, param_file: str,
            num_systems: int = 1000,
            num_class: int = 1000,
            per_class: int = 1000,
            generator: Optional[Callable] = None,
            period: int = 2,
            num_augs: int = 2,
            transform=None):

        assert(num_systems == num_class)  # one code per class
        assert(num_augs == 1 or num_augs == 2)
        assert(generator is not None)
        self.num_systems = num_systems
        self.num_class = num_class
        self.per_class = per_class
        self.systems_per_class = int(self.num_systems // self.num_class)
        assert(self.systems_per_class == 1)
        self.num_augs = num_augs
        self.params = pickle.load(
                open(param_file, 'rb'))['params'][:num_systems]

        self.generator = generator
        assert(self.generator.num_augs <= self.num_augs)

        self.image_level_transform = transform
        if self.generator.num_augs < self.num_augs:
            assert(self.image_level_transform is not None)
            assert(self.generator.num_augs == 1)

        # Add the first random sample into the generator's cache
        k = np.random.default_rng().integers(0, num_class)
        self.generator.add_sample(
                self.params[k]['system'], label=k)
        self.steps = 0
        self.period = period
        return

    def get_label(self, idx):
        """
        [0, 1, 2, 3, 4, 5, ..., num_class * per_class]
        ==>
        [0, 1, 2, .., num_class] * per_class
        """
        return int(idx % self.num_class)

    def get_system(self, idx):
        """
        [0, 1, 2, 3, 4, 5, ..., num_class * per_class]
        ==>
        [0-0, 1-0, 2-0, .., num_class-0]
        [0-1, 1-1, 2-1, .., num_class-1] ...
        """
        label = self.get_label(idx)
        sys_idx = label  # assume only one code per class
        return sys_idx

    def __len__(self):
        return self.num_class * self.per_class

    def __getitem__(self, idx):

        # generate a new sample periodically
        new_sample = (self.steps == 0)
        self.steps = (self.steps + 1) % self.period

        # get a system and get an image pair and its labels
        # we combine the sampled label with other labels to form an image
        sys_idx = self.get_system(idx)
        label = self.get_label(idx)
        imgs, labels = self.generator(
                sys=self.params[sys_idx]['system'],
                label=label, new_sample=new_sample,
                all_params=self.params)

        transformed_imgs = []  # self.num_augs is 1 or 2
        if len(imgs) == 1 and self.num_augs == 2:
            img = torch.from_numpy(imgs[0]).float().mul_(1/255.).permute(2,0,1)
            transformed_imgs.append(self.image_level_transform(img))
            transformed_imgs.append(self.image_level_transform(img))
        else:
            for img in imgs:
                img = torch.from_numpy(img).float().mul_(1/255.).permute(2,0,1)
                if self.image_level_transform is not None:
                    img = self.image_level_transform(img)
                transformed_imgs.append(img)

        labels = torch.zeros((self.num_class,), dtype=torch.long).scatter_(
                0, torch.LongTensor(labels), 1)

        return transformed_imgs[0], transformed_imgs[1], labels


class SelfSupGenerator(_GeneratorBase):
    def __init__(
            self, size: int = 224,
            cache_size_per_class: int = 32,
            max_num_cached_class: int = 512,
            size_range: Tuple[float, float] = (0.45, 0.75),
            jitter_params: Union[bool, str] = True,
            flips: bool = True,
            sigma: Optional[Tuple[float, float]] = (0.5, 2.0),
            blur_p: Optional[float] = 0.5,
            niter = 100000,
            patch = True,
            num_class = 1000,
            num_objects: Tuple[int, int] = (2, 5),
            color_mode: str = 'random',
            num_augs: int = 2,
            background: bool = False):

        assert(color_mode in ['jitter', 'random'])
        assert(num_augs in [1, 2])
        assert(num_objects[1] >= num_objects[0]
               and num_objects[0] > 1)
        self.size = size
        self.cache_size_per_class = cache_size_per_class
        self.max_num_cached_class = max_num_cached_class
        self.size_range = size_range
        self.jitter_params = jitter_params
        self.flips = flips
        self.sigma = sigma
        self.blur_p = blur_p
        self.niter = niter
        self.patch = patch
        self.num_class = num_class
        self.num_objects = num_objects
        self.color_mode = color_mode
        self.num_augs = num_augs
        self.background = background

        self.nobj_p = np.ones(
                num_objects[1]-num_objects[0]+1)
        self.nobj_p /= self.nobj_p.sum()

        self.rng = np.random.default_rng()
        self.have_cache = set()
        self.cache = {
                c: [] for c in range(num_class)}
        if self.background:
            self.cache_size_for_bg = 512
            self.cache['bg'] = []
            self.bg_replace_idx = 0
        self.replace_idx = [0 for _ in range(num_class)]
        self._set_jitter()
        return

    def _update_cache(self, fg, label, bg=None):
        if len(self.have_cache) >= self.max_num_cached_class:
            rm_label = random.sample(self.have_cache, 1)[0]
            del self.cache[rm_label]
            self.cache[rm_label] = []
            self.have_cache.remove(rm_label)

        if len(self.cache[label]) < self.cache_size_per_class:
            self.cache[label].append(fg)
        else:
            idx = self.replace_idx[label]
            self.cache[label][idx] = fg
            self.replace_idx[label] = (
                    (self.replace_idx[label]+1)%self.cache_size_per_class)
        self.have_cache.add(label)

        if self.background:
            if len(self.cache['bg']) < self.cache_size_for_bg:
                self.cache['bg'].append(bg)
            else:
                self.cache['bg'][self.bg_replace_idx] = bg
                self.bg_replace_idx = (
                        (self.bg_replace_idx+1)%self.cache_size_for_bg)
        return

    def render(self, sys):
        rng = self.rng
        coords, region = self._iterate(sys)

        # Render fractals at half resolution for large sizes
        render_img_size = (
                self.size//2 if self.size > 100 else self.size)
        img = ifs.render(
                coords, render_img_size, binary=False,
                region=region, patch=self.patch)
        return img

    def add_sample(self, sysidx, label):
        """
        Add a new foreground and a new background (if needed)
        into the cache
        """
        sysc = self.jitter(sysidx)
        frac = self.render(sysc)
        bg = self.render_background() if self.background else None
        self._update_cache(frac, label, bg=bg)
        return frac

    def generate(self, sys, label, new_sample=True, all_params=[]):
        rng = self.rng

        if new_sample or (label not in self.have_cache):
            self.add_sample(sys, label)

        # Check if we have at least one other labels to choose
        other_labels = self.have_cache - set([label])
        if len(other_labels) == 0:
            k = np.random.choice(
                    list(range(0, label))+list(range(label+1, self.num_class)), 1)[0]
            self.add_sample(all_params[k]['system'], label=k)
            other_labels.add(k)

        # Pick the number of objects we want to put in an image
        n = rng.choice(range(self.num_objects[0],
                             self.num_objects[1]+1), p=self.nobj_p)
        n = min(n-1, len(other_labels))

        # Randomly sample labels and generate fractal images
        labels = [label] + random.sample(other_labels, n)
        imgs = []
        label_hue_shifts = [None for _ in range(len(labels))]
        for aug_idx in range(self.num_augs):

            # Prepare the background
            if self.background:
                idx = rng.integers(0, len(self.cache['bg']))
                img = self.cache['bg'][idx].copy()
            else:
                img = np.zeros((self.size, self.size, 3),
                               dtype=np.uint8)

            # Generate and paste fractals on the background
            for i, l in enumerate(labels):
                idx = rng.integers(0, len(self.cache[l]))
                fg = self.cache[l][idx].copy()

                # random flips
                if self.flips:
                    fg = self.random_flips(fg)

                # random color or color jitter (saturation and brightness)
                if self.color_mode == 'random':
                    fg, _ = self.to_color(fg)
                else:
                    hue_shift = label_hue_shifts[i]
                    fg, hue_shift = self.to_color(fg, hue_shift)
                    label_hue_shifts[i] = hue_shift

                # random size
                f = rng.uniform(*self.size_range)
                s = int(f * self.size)
                fg = resize(
                        fg, (s, s), interpolation=INTER_LINEAR)

                # random location
                x, y = rng.integers(
                        -(s//4), self.size-(s-s//4), 2)
                x1 = 0 if x >= 0 else -x
                x2 = s if x < self.size - s else self.size - x
                y1 = 0 if y >= 0 else -y
                y2 = s if y < self.size - s else self.size - y
                fg = fg[y1:y2, x1:x2]

                # add object to image
                y = max(y, 0)
                x = max(x, 0)
                self.composite(fg, img[y:y+fg.shape[0],
                                       x:x+fg.shape[1]])

            # randomly apply gaussian blur
            if self.blur_p and rng.random() > 0.5:
                img = self.random_blur(img)

            imgs.append(img)

        return imgs, labels
