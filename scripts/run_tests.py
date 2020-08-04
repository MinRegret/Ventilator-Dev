#!/usr/bin/env bash
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"
import os
import tqdm
import pathos

PREP_TIME = 5
EXPERIMENT_TIME = 5
SLEEP_TIME = 5

# Run a single trial
def runner(args):
    import time
    from vent.gui.jupyter import JupyterGUI

    os.system("sudo killall pigpiod")
    time.sleep(PREP_TIME / 2)
    os.system("sudo pigpiod")
    time.sleep(PREP_TIME / 2)

    gui = JupyterGUI(**args)

    gui.start()
    time.sleep(EXPERIMENT_TIME)

    gui.stop()
    time.sleep(SLEEP_TIME)


    return args


# Generate the grid
def gen_grid():
    for a in [1, 2, 3]:
        for b in [2, 3, 4]:
            yield {"a": a, "b": b}


def grid_size(generator):
    return sum(1 for blah in generator)


if "__main__" == __name__:
    run = lambda: pathos.pools.ProcessPool(nodes=1).imap(runner, gen_grid())
    total = grid_size(gen_grid())
    results = list(tqdm.tqdm(run(), total=total))
