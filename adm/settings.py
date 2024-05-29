import gc
import os
import random
import warnings

import numpy as np

import faiss
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 1337

TRAIN_TEST = True
TRAIN_RNG = 1337
TRAIN_RATIO = 0.9

BATCHES = [2]  # [1, 2, 3]

CPU_CORES = 8
os.environ['OMP_NUM_THREADS'] = str(CPU_CORES)
faiss.omp_set_num_threads(CPU_CORES)

TRAIN_TEST_FOLDER = "TRAIN_TEST_"

DEVICE = 'cuda'
DEVICE_TORCH: torch.device | None = torch.device('cuda') if DEVICE == 'cuda' else None
if DEVICE_TORCH is None:
    raise ValueError('CUDA is not available')
torch.set_default_device(DEVICE_TORCH)

DATA_TRAIN = 'data/TRAIN.csv'
DATA_B1 = 'data/B1.csv'
DATA_B2 = 'data/B2_round.csv'
DATA_BFinal = 'data/BFinal.csv'
MODELS_FOLDER = f'{TRAIN_TEST_FOLDER if TRAIN_TEST else ""}models_data'
MODELS_LIST = f'{TRAIN_TEST_FOLDER if TRAIN_TEST else ""}models_list.txt'
RESULTS_FOLDER = f'{TRAIN_TEST_FOLDER if TRAIN_TEST else ""}results'

NUM_RUNNERS = 1
RUNNER_ID = 0


def init_rng(
        seed: int,
) -> None:
    with torch.no_grad():
        torch.cuda.empty_cache()

    gc.collect()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
