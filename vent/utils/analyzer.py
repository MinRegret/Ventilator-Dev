import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import dill as pickle
import matplotlib.pyplot as plt

from vent.utils import BreathWaveform


class Analyzer:
    def __init__(self, path):
        self.data = path if isinstance(path, dict) else pickle.load(open(path, "rb"))
        self.controller = self.data["controller"]
        waveforms = self.data["timeseries"]
        self.tt = waveforms["timestamp"]
        self.u_in = waveforms["u_in"]
        self.u_out = waveforms["u_out"]
        self.pressure = waveforms["pressure"]
        self.target = waveforms["target"]
        self.flow = waveforms["flow"]

    ###########################################################################
    # Plotting methods
    ###########################################################################

    def plot(self, axes=None, figsize=None, xlim=None, ylim=[0, 60], **kwargs):
        matplotlib.pyplot.figure(figsize=(figsize or (12, 6)))
        # trash
        if axes is None:
            axes = matplotlib.pyplot.axes()

        if xlim is not None:
            axes.set_xlim(xlim)

        if "color" not in kwargs:
            kwargs["color"] = "b"

        axes.plot(self.tt, self.pressure, **kwargs)
        axes.plot(self.tt, self.target, color="orange")
        axes.set_ylabel("pressure")
        axes.set_ylim(ylim)
        axes.fill_between(
            self.tt,
            axes.get_ylim()[0],
            axes.get_ylim()[1],
            where=self.u_out.astype(bool),
            color="lightgray",
            alpha=0.3,
        )

        twin_ax = axes.twinx()
        twin_ax.set_ylim([-2, 102])
        twin_ax.plot(self.tt, np.clip(self.u_in, 0, 100), c="gray")
        twin_ax.set_ylabel("control")

    def plot_inspiratory_clips(self, **kwargs):
        inspiratory_clips = self.infer_inspiratory_phases()

        plt.subplot(121)
        plt.title("u_in")
        for start, end in inspiratory_clips:
            u_in = self.u_in[start:end]
            plt.plot(self.tt[start:end] - self.tt[start], u_in, "k", alpha=0.1)

        plt.subplot(122)
        plt.title("pressure")
        for start, end in inspiratory_clips:
            pressure = self.pressure[start:end]
            plt.plot(self.tt[start:end] - self.tt[start], pressure, "b", alpha=0.1)

    ###########################################################################
    # Utility methods
    ###########################################################################

    def infer_inspiratory_phases(self, use_cached=True):
        # finds inspiratory phase intervals from expiratory valve controls
        # returns list of endpoints so that u_out[lo:hi] == 1

        if not use_cached or not hasattr(self, "cached_inspiratory_phases"):
            d_u_out = np.diff(self.u_out, prepend=1)

            starts = np.where(d_u_out == -1)[0]
            ends = np.where(d_u_out == 1)[0]

            self.cached_inspiratory_phases = list(zip(starts, ends))

        return self.cached_inspiratory_phases

    ###########################################################################
    # Metric methods
    ###########################################################################

    def losses_per_breath(self, target, loss_fn=None):
        # computes trapezoidally integrated loss per inferred breath

        loss_fn = loss_fn or np.square

        # handle polymorphic targets
        if type(target) is int:
            target_fn = lambda _: target
        elif type(target) is BreathWaveform:
            target_fn = lambda t: target.at(t)
        else:
            raise ValueError("unrecognized type for target")

        breaths = self.infer_inspiratory_phases()
        losses = []

        # integrate loss for each detected inspiratory phase
        for start, end in breaths:
            errs = loss_fn(target_fn(self.tt[start:end]) - self.pressure[start:end])
            loss = np.trapz(errs, self.tt[start:end])
            losses.append(loss)

        return np.array(losses)

    def default_metric(self, target=None):
        # I suggest keeping a separate function for default settings
        # so nobody has to change code if we change the benchmark
        # as it stands: average loss across breaths, discounting first breath
        target = target or int(max(self.target))

        return self.losses_per_breath(target, np.abs)[1:].mean()

    def lazy_default_metric(self):
        # a way to guess PIP, for super quick triage...

        candidate_targets = [15, 20, 25, 30, 35]
        losses = [self.default_metric(target) for target in candidate_targets]
        best_idx = np.argmin(losses)

        return candidate_targets[best_idx], losses[best_idx]
