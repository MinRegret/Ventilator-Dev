#!/usr/bin/env bash
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"
import os
import tqdm
import click
import pathos
import datetime

from lung.controllers import AdaPI
from lung.controllers import PredictiveBiasPI
from lung.controllers import PredictivePID

def get_controllers(log_directory):
    controllers = []
    for p in [0.01, 0.1, 0.3, 0.5, 1.0]:
        for i in [0.1, 0.5, 1.0, 1.5, 2.0]:
            for rc in [0.06, 0.3]:
                for lookahead_steps in [0, 15]:
                    controllers.append(AdaPI(p=p, i=i, RC=rc, lookahead_steps=lookahead_steps, log_directory=log_directory))
                    controllers.append(PredictiveBiasPI(p=p, i=i, RC=rc, lookahead_steps=lookahead_steps, log_directory=log_directory))

    controllers.append(PredictivePID(hallucination_length=0, log_directory=log_directory))
    controllers.append(PredictivePID(hallucination_length=15, log_directory=log_directory))

    return controllers

def get_runner(prep_time, experiment_time, sleep_time):
    def get_runner_inner(args):
        import time
        from vent.gui.jupyter import JupyterGUI

        os.system("sudo killall pigpiod")
        time.sleep(prep_time / 2)
        os.system("sudo pigpiod")
        time.sleep(prep_time / 2)

        gui = JupyterGUI(**args)

        gui.start()
        time.sleep(experiment_time)

        gui.stop()
        time.sleep(sleep_time)

        return args
    return get_runner_inner


# Generate the grid
def gen_grid(**kwargs):
    directory = kwargs["directory"]
    controllers = get_controllers(directory)

    for pip in [15, 25, 35]:
        for peep in [5, 10]:
            for controller in controllers:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                kwargs["directory"] = f"{directory}/hazan/{timestamp}"
                kwargs.update({"pip": pip, "peep": peep, "controller": controller})
                yield kwargs


def grid_size(generator):
    return sum(1 for blah in generator)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--prep-time", default=10, type=int, help="Time to kill/restart pigpiod")
@click.option("--experiment-time", default=30, type=int, help="Time to run experiment")
@click.option("--sleep-time", default=20, type=int, help="Time to give lung a breather")
@click.option("-o", "--directory", default=os.path.join(os.path.expanduser("~"), "vent/logs"), type=str, help="Directory for this series of runs")
def main(prep_time, experiment_time, sleep_time, directory):
    runner = get_runner(prep_time, experiment_time, sleep_time)
    run = lambda: pathos.pools.ProcessPool(nodes=1).imap(runner, gen_grid(directory=directory))

    # Lol...we redo a lot of work here...
    total = grid_size(gen_grid(directory=directory))
    results = list(tqdm.tqdm(run(), total=total))


if "__main__" == __name__:
    main()
