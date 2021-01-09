import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from lung.controllers import ResidualExplorer, PID
from lung.environments import PhysicalLung
from lung.utils.analyzer import Analyzer
from lung.utils import BreathWaveform
import torch

plt.rc('figure', figsize=(10,3))

def collect_runs(controller, n_runs=1, append_to=None, lung=None, **kwargs):
    if append_to is None:
        results = []
    else:
        results = append_to
    
    for i in range(n_runs):
        lung.reset()
        result = lung.run(controller, **kwargs)
        results.append(result)
    return results

def save_runs(results, path_template, save_controller=True):
    for i,result in enumerate(results):
        if not save_controller:
            del result['controller']
        filename = path_template % i
        pickle.dump(result, open(filename, 'wb'))

def populate_grid_entry(R, C, PEEP, PIP, T, n_runs, abort, lung): # 5 runs at a specific PIP
    waveform = BreathWaveform((PEEP, PIP))
    pid = PID([0,1,0])
    explorer = ResidualExplorer(pid, waveform=waveform)
    explorer.delta_range = (-20, 10)  # hand-tuned magic numbers for R=50 C=10

    results = collect_runs(explorer, T=T, dt=0.03, R=R, C=C, PEEP=PEEP, abort=abort, n_runs=n_runs, lung=lung)
    return results

def explore(R=50, C=10, PEEP=5, T=10000, directory=None, n_runs=5, PIPs=[10, 15, 20, 25, 30, 35], abort=70, lung=PhysicalLung()):
    all_results = {}
    for PIP in PIPs:
        results = populate_grid_entry(R, C, PEEP, PIP, T, n_runs=n_runs, abort=abort, lung=lung)
        if directory is not None:
            save_runs(results, os.path.join(directory, f"R{R}C{C}PEEP{PEEP}_PIP{PIP}_pid_triangle_residual_%i.pkl"))
        all_results[PIP] = results
    return all_results
