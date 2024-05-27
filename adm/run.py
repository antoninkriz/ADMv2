import os
import subprocess

import adm.run_build
import adm.run_train_test
import adm.settings


def run_all():
    adm.run_build.run()

    if adm.settings.NUM_RUNNERS == 1:
        adm.run_train_test.run(0, 1)
        return

    run_calc_path = os.path.abspath(adm.run_train_test.__file__)

    log_stdout_handles = [
        open(f'LOG_{i}_out.log', 'w')
        for i in range(adm.settings.NUM_RUNNERS)
    ]
    log_stderr_handles = [
        open(f'LOG_{i}_err.log', 'w')
        for i in range(adm.settings.NUM_RUNNERS)
    ]

    processes: list[None | subprocess.Popen] = [None] * adm.settings.NUM_RUNNERS
    for i in range(adm.settings.NUM_RUNNERS):
        processes[i] = subprocess.Popen(
            ['python', run_calc_path, str(i)],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=log_stdout_handles[i],
            stderr=log_stderr_handles[i],
        )

    for p in processes:
        p.wait()

    for i, p in enumerate(processes):
        if p.returncode != 0:
            print(f'Error in :: {i} :: run_calc.py')

    for h in log_stdout_handles:
        h.close()

    for h in log_stderr_handles:
        h.close()


if __name__ == '__main__':
    run_all()
