import torch
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt

from vent.utils.analyzer import Analyzer
from vent.core import VentObj


class Munger(VentObj):
    def __init__(self, paths, **kwargs):
        self.paths = paths
        self.analyzers = []
        self.data = []
        self.splits = {}

        # Add data
        self.add_data(paths, **kwargs)

        # Shuffle and split data
        self.split_data(**kwargs)

        # Compute mean, std for u_in and pressure
        self.u_scaler, self.p_scaler = self.fit_scalers(**kwargs)

    # Note: the following methods are meant to be run in order
    def add_data(self, paths, **kwargs):
        for path in paths:
            analyzer = Analyzer(path)
            self.analyzers.append(analyzer)
            inspiratory_clips = analyzer.infer_inspiratory_phases()

            for start, end in inspiratory_clips[2:-1]:  # skip first 2 & last breaths
                u_in = analyzer.u_in[start:end]
                pressure = analyzer.pressure[start:end]
                self.data.append((u_in, pressure))

        print(f"Added {len(self.data)} breaths from {len(self.paths)} paths.")

    def split_data(self, seed=0, keys=["train", "test"], splits=[0.9, 0.1], **kwargs):
        rng = np.random.default_rng(seed)

        # Determine split boundaries
        splits = (np.array(splits) / np.sum(splits) * len(self.data)).astype("int")[:-1]

        # Everyday I'm shuffling
        rng.shuffle(self.data)

        # Splitting
        self.splits = {key: val for key, val in zip(keys, np.split(self.data, splits))}

    def fit_scalers(self, key="train"):
        u_scaler = sklearn.preprocessing.StandardScaler()
        u_scaler.fit(np.concatenate([u for u, p in self.splits[key]]).reshape(-1, 1))
        print(f"u_in: mean={u_scaler.mean_}, std={u_scaler.scale_}")

        p_scaler = sklearn.preprocessing.StandardScaler()
        p_scaler.fit(np.concatenate([p for u, p in self.splits[key]]).reshape(-1, 1))
        print(f"pressure: mean={p_scaler.mean_}, std={p_scaler.scale_}")

        return u_scaler, p_scaler

    def scale_and_window(self, key, u_window=5, p_window=3):
        X, y = [], []
        for u, p in self.splits[key]:
            T = len(u)
            u_scaled = self.u_scaler.transform(u.reshape(-1, 1))
            p_scaled = self.p_scaler.transform(p.reshape(-1, 1))

            for t in range(max(u_window, p_window) + 1, T):
                features = np.concatenate(
                    [u_scaled[t - u_window : t, 0], p_scaled[t - p_window : t, 0]]
                )
                target = p_scaled[t, 0]
                X.append(features)
                y.append(target)

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def scale_and_window_boundary(self, key, boundary_index, odd_indexing=True):
        X, y = [], []

        even_indexing = 1 - odd_indexing

        if boundary_index == 0 and even_indexing:  # special case: no features, predict p[0]
            for u, p in self.splits[key]:
                p_scaled = self.p_scaler.transform([[p[0]]])
                target = p_scaled[0][0]
                y.append(target)
            return None, torch.tensor(y, dtype=torch.float)

        # otherwise, collate [first B inputs, first B pressures] -> (next pressure) pairs
        for u, p in self.splits[key]:
            T = len(u)
            if T < boundary_index + even_indexing:  # if trajectory is too short, abort
                continue

            u_scaled = self.u_scaler.transform(u[:boundary_index].reshape(-1, 1)).flat
            p_scaled = self.p_scaler.transform(
                p[: boundary_index + even_indexing].reshape(-1, 1)
            ).flat

            features = np.concatenate([u_scaled, p_scaled[:-1]])
            target = p_scaled[-1]

            X.append(features)
            y.append(target)

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def get_data_loader(self, key, u_window=5, p_window=3, batch_size=512, shuffle=True):
        X, y = self.scale_and_window(key, u_window, p_window)
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_boundary_data_loader(
        self, key, boundary_index, odd_indexing=True, batch_size=512, shuffle=True
    ):
        X, y = self.scale_and_window_boundary(key, boundary_index, odd_indexing)
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def unscale_pressures(self, p):
        if type(p) is float:
            p = np.array([p])
        return self.p_scaler.inverse_transform(p.reshape(-1, 1))[:, 0]

    ###########################################################################
    # Plotting methods
    ###########################################################################

    def plot_boundary_pressures(self):
        plt.rc("figure", figsize=(16, 4))

        for tau in range(1, 6):
            plt.subplot(150 + tau)

            u_init = []
            p_init = []

            for u, p in self.splits["train"]:
                u_init.append(u[:tau].mean())
                p_init.append(p[tau])

            plt.xlim([0, 105])
            plt.ylim([2, 33])
            plt.xlabel(f"u_in[0:{tau}].mean")
            plt.title(f"pressure[{tau}]")

            plt.scatter(u_init, p_init, s=1)
