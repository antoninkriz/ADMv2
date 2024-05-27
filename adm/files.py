from typing import Literal

import polars as pl

import adm.settings


def read_train(train_test: Literal['train', 'test'] | None) -> pl.DataFrame:
    df = pl.read_csv(adm.settings.DATA_TRAIN, columns=['User', 'Query', 'Item'], dtypes=[pl.Int32, pl.String, pl.Int32])
    if train_test is not None:
        df = df.sample(fraction=1, seed=adm.settings.TRAIN_RNG)
    match train_test:
        case 'train':
            df = df.head(int(df.shape[0] * adm.settings.TRAIN_RATIO))
        case 'test':
            df = df.tail(int(df.shape[0] * (1 - adm.settings.TRAIN_RATIO)))
        case None:
            pass
    return df


def read_b1() -> pl.DataFrame:
    return pl.read_csv(adm.settings.DATA_B1, columns=['User', 'Query'], dtypes=[pl.Int32, pl.String])


def read_b2() -> pl.DataFrame:
    return pl.read_csv(adm.settings.DATA_B2, columns=['User', 'Query'], dtypes=[pl.Int32, pl.String])


def read_bfinal() -> pl.DataFrame:
    return pl.read_csv(adm.settings.DATA_BFinal, columns=['User', 'Query'], dtypes=[pl.Int32, pl.String])
