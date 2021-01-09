import os
import numpy as np
import pandas as pd
import torch

DEFAULT_PRESSURE_RANGE = (5.0, 35.0)
DEFAULT_KEYPOINTS = [1e-8, 1.0, 1.5, 3.0]
DEFAULT_BPM = 20


class BreathWaveform:
    """Waveform generator with shape |‾\_"""

    def __init__(self, range=None, keypoints=None, bpm=DEFAULT_BPM, kernel=None, dt=0.01):
        self.lo, self.hi = range or DEFAULT_PRESSURE_RANGE
        self.fp = [self.lo, self.hi, self.hi, self.lo, self.lo]

        self.xp = np.zeros(len(self.fp))
        self.xp[1:] = np.array(keypoints or DEFAULT_KEYPOINTS)
        self.xp[-1] = 60 / bpm

        self._keypoints = self.xp

        pad = 0
        num = int(1 / dt)
        if kernel is not None:
            pad = 60 / bpm / (num - 1)
            num += len(kernel) // 2 * 2

        tt = np.linspace(-pad, 60 / bpm + pad, num)
        self.fp = self.at(tt)
        self.xp = tt

        if kernel is not None:
            self.fp = np.convolve(self.fp, kernel, mode="valid")
            self.xp = np.linspace(0, 60 / bpm, int(1 / dt))

    @property
    def keypoints(self):
        if hasattr(self, "_keypoints"):
            return self._keypoints
        else:
            return self.xp

    @property
    def period(self):
        return self.xp[-1]

    def at(self, t):
        return np.interp(t, self.xp, self.fp, period=self.period)

    def elapsed(self, t):
        return t % self.period

    def decay(self, t):
        elapsed = self.elapsed(t)
        if elapsed < self.keypoints[2]:
            return None
        elif elapsed < self.keypoints[3]:
            return 0.0
        else:
            return 5 * (1 - np.exp(5 * (self.keypoints[3] - elapsed)))

    def phase(self, t):
        return np.searchsorted(self.keypoints, t % self.period, side="right")


class ValveCurve:
    def __init__(self, filename=None):
        filename = None or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../data/valve_response_bidir.csv"
        )
        df = pd.read_csv(filename, header=None)
        self.data = df.to_numpy()

    def at(self, x):
        return np.interp(np.clip(x, 0, 100) / 100, self.data[:, 0], self.data[:, 1])


class WeightClipper(object):
    def __init__(self, min=0.0, max=100.0):
        self.min, self.max = min, max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w.clamp_(min=self.min, max=self.max)
