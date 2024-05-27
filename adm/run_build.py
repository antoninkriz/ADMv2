import os
import time

import torch

import adm.files
import adm.models
import adm.params
import adm.settings


def build_elsa_models(seed: int) -> None:
    for elsa_params in adm.params.get_params_elsa():
        adm.settings.init_rng(seed=seed)

        print(f'- Building ELSA model {elsa_params}')

        start_time = time.time()
        save_path = adm.models.train_elsa(params=elsa_params)
        print(f'- - {time.time() - start_time:.3f}s')

        with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
            f.write(f'{save_path} = {elsa_params}\n')


def build_encoder_models(seed: int) -> None:
    for encoder_params in adm.params.get_params_encoder():
        adm.settings.init_rng(seed=seed)

        print(f'- Building encoder model {encoder_params}')

        if not adm.settings.TRAIN_TEST:
            print('  - Encoding sentences train')
            save_path_train = adm.models.encode_sentences_train(params=encoder_params, train_test=None)
            with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
                f.write(f'{save_path_train} = {encoder_params}\n')

            print('  - Encoding sentences batch 1')
            save_path_b1 = adm.models.encode_sentences_batch(params=encoder_params, batch_id=1)
            with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
                f.write(f'{save_path_b1} = {encoder_params}\n')

            print('  - Encoding sentences batch 2')
            save_path_b2 = adm.models.encode_sentences_batch(params=encoder_params, batch_id=2)
            with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
                f.write(f'{save_path_b2} = {encoder_params}\n')

            print('  - Encoding sentences batch 3')
            save_path_b3 = adm.models.encode_sentences_batch(params=encoder_params, batch_id=3)
            with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
                f.write(f'{save_path_b3} = {encoder_params}\n')
        else:
            print(f'  - Encoding sentences train - part train')
            save_path_test = adm.models.encode_sentences_train(params=encoder_params, train_test='train')
            with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
                f.write(f'{save_path_test} = {encoder_params}\n')

            print(f'  - Encoding sentences train - part test')
            save_path_test = adm.models.encode_sentences_train(params=encoder_params, train_test='test')
            with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
                f.write(f'{save_path_test} = {encoder_params}\n')


def build_faiss_indexes(seed: int) -> None:
    for encoder_params in adm.params.get_params_encoder():
        save_path = f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("faiss", encoder_params)}.npy'

        adm.settings.init_rng(seed=seed)

        print(f'- Building faiss index {encoder_params}')
        df_embeddings = adm.models.load_encoded_sentences_train(
            params=encoder_params,
            train_test=('train' if adm.settings.TRAIN_TEST else None),
        )
        if os.path.exists(save_path):
            print('- Faiss index already exists')
        else:
            save_path = adm.models.build_faiss_index(df_embeddings=df_embeddings, encoder_params=encoder_params)

        with open(f'{adm.settings.MODELS_FOLDER}/{adm.settings.MODELS_LIST}', 'a') as f:
            f.write(f'{save_path} = {encoder_params}\n')


def run() -> None:
    print(f'STARTING BUILD')
    time.sleep(5)

    try:
        os.mkdir(adm.settings.RESULTS_FOLDER)
        os.mkdir(adm.settings.MODELS_FOLDER)
    except FileExistsError:
        pass

    seed = 1337

    print('BUILDING ELSA MODELS')
    build_elsa_models(seed)

    print('BUILDING ENCODER MODELS')
    with torch.no_grad():
        build_encoder_models(seed)

    print('BUILDING FAISS INDEXES')
    build_faiss_indexes(seed)


if __name__ == '__main__':
    run()
