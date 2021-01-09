import numpy as np
import tqdm

from vent.controllers import PID
from vent.environments import PhysicalLung
from vent.utils import Analyzer, BreathWaveform

DEFAULT_VALUES = np.concatenate((
    np.array([0., 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
    np.linspace(1, 10, 10)
))

def run_grid(values=DEFAULT_VALUES, R=50, C=10, PEEP=5, abort=60, T=300, directory=None, vent=PhysicalLung(), **kwargs):
    analyzers = []

    for PIP in tqdm.tqdm([10, 15, 20, 25, 30, 35]):
        for P in tqdm.tqdm(values, leave=False):
            for I in tqdm.tqdm(values, leave=False):
                vent.reset()

                waveform = BreathWaveform((PEEP, PIP))
                pid = PID(K=[P, I, 0.], waveform=waveform, decay=True)
                analyzer = vent.run(pid, R=R, C=C, PEEP=PEEP, abort=abort, directory=directory, T=T, use_tqdm=False, **kwargs)
                analyzers.append(analyzer)

