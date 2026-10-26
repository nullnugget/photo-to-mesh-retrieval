"""The start of a worker process (a data loader worker or a worker of a process pool).

In its own module, so that every step can use it: `src/data/datasets.py` imports the feature step, so the
feature step cannot import `datasets` (a circular import).
"""

import os

import torch


def loader_worker_init(_worker_id: int) -> None:
    """Prepare a data loader worker process: all CPU cores, and one PyTorch thread.

    1. All CPU cores. A worker is a copy (fork) of the main process, and it inherits the core
       binding of the main thread. The OpenMP runtime of the CPU math libraries can bind the main
       thread to one core (2 logical CPUs) when it starts; then all workers share these 2 CPUs.
       Measured: 111 images per second with 4 workers on 2 CPUs, 273 on all 16. Whether the binding
       exists depends on what the main process did before (the order of the imports, a notebook),
       so the worker always sets all cores again.
    2. One PyTorch thread. The image changes use small PyTorch operations; with the default (one
       thread per core in each worker) the workers slow each other down.

    Used by the training loader, the loaders of the feature and patch steps, and the worker pool of
    the query images (`src/preprocess/queries.py`).
    """
    if hasattr(os, "sched_setaffinity"):  # Linux; other systems do not bind this way
        os.sched_setaffinity(0, range(os.cpu_count()))
    torch.set_num_threads(1)
