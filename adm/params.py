from typing import TypedDict
import hashlib
import itertools

import adm.settings


class ElsaParams(TypedDict):
    n_dims: int
    batch_size: int
    epochs: int


class EncoderParams(TypedDict):
    model: str
    batch_size: int


class PredictParams(TypedDict):
    n_items: int
    multiplier: int
    cut_items: int


def params_to_name(name: str, params: dict) -> str:
    sha1hash = hashlib.sha1(str(tuple(params.items())).encode('utf-8')).hexdigest()
    return f'{"TRAIN_TEST_" if adm.settings.TRAIN_TEST else ""}{name}_{sha1hash}'


def get_params_elsa() -> list[ElsaParams]:
    return [
        ElsaParams(
            n_dims=elsa_n_dims,
            batch_size=elsa_batch_size,
            epochs=elsa_epochs,
        )
        for elsa_batch_size, elsa_n_dims, elsa_epochs in itertools.product(
            [64, 128, 192],
            [128, 192, 256],
            [3, 4, 5, 7],
        )
    ]


def get_params_encoder() -> list[EncoderParams]:
    return [
        EncoderParams(
            model=model_batch[0],
            batch_size=model_batch[1],
        )
        for model_batch in [
            # ('sentence-transformers/all-mpnet-base-v2', 512),
            ('mixedbread-ai/mxbai-embed-large-v1', 512),
            ('nomic-ai/nomic-embed-text-v1.5', 512),
        ]
    ]


def get_params_predict() -> list[PredictParams]:
    return [
        PredictParams(
            n_items=predict_n_items,
            multiplier=predict_multiplier,
            cut_items=predict_cut_items,
        )
        for predict_n_items, predict_multiplier, predict_cut_items in itertools.product(
            [165],
            [1, 12, 25, 40, 70, 95, 125],
            [150]
        )
    ]


def get_params_best() -> list[tuple[ElsaParams, EncoderParams, PredictParams]]:
    return [
        (
            ElsaParams(
                n_dims=192,
                batch_size=64,
                epochs=7,
            ),
            EncoderParams(
                model='nomic-ai/nomic-embed-text-v1.5',
                batch_size=512,
            ),
            PredictParams(
                n_items=165,
                multiplier=40,
                cut_items=150,
            ),
        ),
        # (
        #     ElsaParams(
        #         n_dims=128,
        #         batch_size=64,
        #         epochs=3,
        #     ),
        #     EncoderParams(
        #         model='nomic-ai/nomic-embed-text-v1.5',
        #         batch_size=512,
        #     ),
        #     PredictParams(
        #         n_items=165,
        #         multiplier=70,
        #         cut_items=150,
        #     ),
        # ),
        # (
        #     ElsaParams(
        #         n_dims=128,
        #         batch_size=64,
        #         epochs=5,
        #     ),
        #     EncoderParams(
        #         model='nomic-ai/nomic-embed-text-v1.5',
        #         batch_size=512,
        #     ),
        #     PredictParams(
        #         n_items=165,
        #         multiplier=70,
        #         cut_items=150,
        #     ),
        # ),
        # (
        #     ElsaParams(
        #         n_dims=128,
        #         batch_size=128,
        #         epochs=4,
        #     ),
        #     EncoderParams(
        #         model='nomic-ai/nomic-embed-text-v1.5',
        #         batch_size=512,
        #     ),
        #     PredictParams(
        #         n_items=165,
        #         multiplier=70,
        #         cut_items=150,
        #     ),
        # ),
        # (
        #     ElsaParams(
        #         n_dims=192,
        #         batch_size=128,
        #         epochs=5,
        #     ),
        #     EncoderParams(
        #         model='nomic-ai/nomic-embed-text-v1.5',
        #         batch_size=512,
        #     ),
        #     PredictParams(
        #         n_items=165,
        #         multiplier=70,
        #         cut_items=150,
        #     ),
        # )
    ]
