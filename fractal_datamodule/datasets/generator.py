"""
Similar to https://github.com/catalys1/fractal-pretraining/blob/main/fractal_learning/training/datamodule/datasets/generator.py
"""
from functools import partial
from typing import Callable, Optional, Tuple, Union

from cv2 import GaussianBlur, resize, INTER_LINEAR
import numpy as np

from .fractals import diamondsquare, ifs


class _GeneratorBase(object):
    def __init__(
        self,
        size: int = 224,
        jitter_params: Union[bool, str] = True,
        flips: bool = True,
        sigma: Optional[Tuple[float, float]] = (0.5, 1.0),
        blur_p: Optional[float] = 0.5,
        niter = 100000,
        patch = True,
    ):
        self.size = size
        self.jitter_params = jitter_params
        self.flips = flips
        self.sigma = sigma
        self.blur_p = blur_p
        self.niter = niter
        self.patch = patch

        self.rng = np.random.default_rng()
        self.cache = {'fg': [], 'bg': []}
        self._set_jitter()

    def _set_jitter(self):
        if isinstance(self.jitter_params, str):
            if self.jitter_params.startswith('fractaldb'):
                k = int(self.jitter_params.split('-')[1]) / 10
                choices = np.linspace(1-2*k, 1+2*k, 5, endpoint=True)
                self.jitter_fnc = partial(self._fractaldb_jitter, choices=choices)
            elif self.jitter_params.startswith('svd'):
                self.jitter_fnc = self._svd_jitter
            elif self.jitter_params.startswith('sval'):
                self.jitter_fnc = self._sval_jitter
        elif self.jitter_params:
            self.jitter_fnc = self._basic_jitter
        else:
            self.jitter_fnc = lambda x: x

    def _fractaldb_jitter(self, sys, choices=(.8,.9,1,1.1,1.2)):
        n = len(sys)
        y, x = np.divmod(self.rng.integers(0, 6, (n,)), 3)
        sys[range(n), y, x] *= self.rng.choice(choices)
        return sys

    def _basic_jitter(self, sys, prange=(0.8, 1.1)):
        # tweak system parameters--randomly choose one transform and scale it
        # this actually amounts to scaling the singular values by a random factor
        n = len(sys)
        sys[self.rng.integers(0, n)] *= self.rng.uniform(*prange)
        return sys

    def _svd_jitter(self, sys):
        '''Jitter the parameters of one of the systems functions, in SVD space.'''
        k = self.rng.integers(0, len(sys) * 3)
        sidx, pidx = divmod(k, 3)
        if pidx < 2:
            q = self.rng.uniform(-0.5, 0.5)
            u, s, v = np.linalg.svd(sys[sidx, :, :2])
            cq, sq = np.cos(q), np.sin(q)
            r = np.array([[cq, -sq], [sq, cq]])
            if pidx == 0:
                u = r @ u
            else:
                v = r @ v
            sys[sidx, :, :2] = (u * s[None,:]) @ v
        else:
            x, y = self.rng.uniform(-0.5, 0.5, (2,))
            sys[sidx, :, 2] += [x, y]
        return sys

    def _sval_jitter(self, sys):
        k = self.rng.integers(0, sys.shape[0])
        svs = np.linalg.svd(sys[...,:2], compute_uv=False)
        fac = (svs * [1, 2]).sum()
        minf = 0.5 * (5 + sys.shape[0])
        maxf = minf + 0.5
        ss = svs[k, 0] + 2 * svs[k, 1]
        smin = (minf - fac + ss) / ss
        smax = (maxf - fac + ss) / ss
        m = self.rng.uniform(smin, smax)
        u, s, v = np.linalg.svd(sys[k, :, :2])
        s = s * m
        sys[k, :, :2] = (u * s[None]) @ v
        return sys

    def jitter(self, sys):
        attempts = 4 if self.jitter_params else 0
        for i in range(attempts):
            # jitter system parameters
            sysc = sys.copy()
            sysc = self.jitter_fnc(sysc)
            # occasionally the modified parameters cause the system to explode
            svd = np.linalg.svd(sysc[:,:,:2], compute_uv=False)
            if svd.max() > 1: continue
            break
        else:
            # fall back on not jittering the parameters
            sysc = sys
        return sysc

    def _iterate(self, sys):
        rng = self.rng

        coords = ifs.iterate(sys, self.niter)
        region = np.concatenate(ifs.minmax(coords))

        return coords, region

    def render(self, sys):
        raise NotImplementedError()

    def random_flips(self, img):
        # random flips/rotations
        if self.rng.random() > 0.5:
            img = img.transpose(1, 0)
        if self.rng.random() > 0.5:
            img = img[::-1]
        if self.rng.random() > 0.5:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        return img

    def to_color(self, img, hue_shift=None):
        return ifs.colorize(img, hue_shift=hue_shift)

    def to_gray(self, img):
        return (img * 127).astype(np.uint8)[..., None].repeat(3, axis=2)

    def render_background(self):
        bg = diamondsquare.colorized_ds(self.size)
        return bg

    def composite(self, foreground, base, idx=None):
        return ifs.composite(foreground, base)

    def random_blur(self, img):
        sigma = self.rng.uniform(*self.sigma)
        img = GaussianBlur(img, (3, 3), sigma, dst=img)
        return img

    def generate(self, sys):
        raise NotImplementedError()

    def __call__(self, sys, *args, **kwargs):
        return self.generate(sys, *args, **kwargs)
