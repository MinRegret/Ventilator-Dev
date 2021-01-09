import numpy as np
import tqdm

from lung.controllers import PID
from lung.environments import PhysicalLung
from lung.utils import Analyzer, BreathWaveform

DEFAULT_VALUES = np.concatenate((
    np.array([0., 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
    np.linspace(1, 10, 10)
))

def run_grid(values=DEFAULT_VALUES, R=50, C=10, PEEP=5, abort=60, T=300, directory=None, lung=PhysicalLung(), **kwargs):
    analyzers = []

    for PIP in tqdm.tqdm([10, 15, 20, 25, 30, 35]):
        for P in tqdm.tqdm(values, leave=False):
            for I in tqdm.tqdm(values, leave=False):
                lung.reset()

                waveform = BreathWaveform((PEEP, PIP))
                pid = PID(K=[P, I, 0.], waveform=waveform, decay=True)
                analyzer = lung.run(pid, R=R, C=C, PEEP=PEEP, abort=abort, directory=directory, T=T, use_tqdm=False, **kwargs)
                analyzers.append(analyzer)

