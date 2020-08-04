#!/usr/bin/env bash
"exec" "`dirname $0`/../venv/bin/python" "$0" "$@"
import os
import tqdm
import click
import pathos

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
        time.sleep(sleep_tim)

        return args
    return get_runner_inner


# Generate the grid
def gen_grid():
    for a in [1, 2, 3]:
        for b in [2, 3, 4]:
            yield {"a": a, "b": b}


def grid_size(generator):
    return sum(1 for blah in generator)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--prep-time", default=10, type=int, help="Time to kill/restart pigpiod")
@click.option("--experiment-time", default=30, type=int, help="Time to run experiment")
@click.option("--sleep-time", default=20, type=int, help="Time to give lung a breather")
@click.option("-o", "--output-directory", default=os.path.join(os.path.expanduser("~"), "vent/logs"), type=str, help="Directory for this series of runs")
def main(prep_time, experiment_time, sleep_time, output_directory):
    runner = get_runner(prep_time, experiment_time, sleep_time)
    run = lambda: pathos.pools.ProcessPool(nodes=1).imap(runner, gen_grid())
    total = grid_size(gen_grid())
    results = list(tqdm.tqdm(run(), total=total))


if "__main__" == __name__:
    main()
