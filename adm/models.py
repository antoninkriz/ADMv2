from typing import Literal
import os

import numpy as np
import scipy
import polars as pl

import torch
import elsa
import faiss
import sentence_transformers

import adm.files
import adm.params
import adm.settings


def make_sparse_interactions_matrix(*, train_test: Literal['train', 'test'] | None) -> scipy.sparse.csr_matrix:
    t = adm.files.read_train(train_test).drop('Query')

    max_user = t[['User']].max().item() + 1
    max_item = t[['Item']].max().item() + 1

    t = t.sort(['User', 'Item'])

    csr = scipy.sparse.csr_matrix(
        (
            np.ones(t.shape[0], dtype=np.float32),
            (t['User'], t['Item'])
        ),
        shape=(max_user, max_item),
        dtype=np.float32
    )

    return csr


def make_sparse_interactions_matrix_coo(*, train_test: Literal['train', 'test'] | None) -> scipy.sparse.coo_matrix:
    t = adm.files.read_train(train_test).drop('Query')

    max_user = t[['User']].max().item() + 1
    max_item = t[['Item']].max().item() + 1

    t = t.sort(['User', 'Item'])

    coo = scipy.sparse.coo_matrix(
        (
            np.ones(t.shape[0], dtype=np.float32),
            (t['User'], t['Item'])
        ),
        shape=(max_user, max_item),
        dtype=np.float32
    )

    return coo


def train_elsa(*, params: adm.params.ElsaParams) -> str:
    save_path = f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("elsa", params)}.pt'
    print(save_path)
    if os.path.exists(save_path):
        print('- ELSA model already exists')
        return save_path

    n_dims, batch_size, epochs = params['n_dims'], params['batch_size'], params['epochs']

    m = make_sparse_interactions_matrix(train_test='train' if adm.settings.TRAIN_TEST else None)
    _, items_cnt = m.shape

    model = elsa.ELSA(n_items=items_cnt, device=adm.settings.DEVICE_TORCH, n_dims=n_dims)
    model.compile()
    model.fit(m, batch_size=batch_size, epochs=epochs)
    torch.save(model.state_dict(), save_path)

    return save_path


def load_elsa(*, params: adm.params.ElsaParams, train_test: Literal['train', 'test'] | None) -> elsa.ELSA:
    _, items_cnt = make_sparse_interactions_matrix(train_test=train_test).shape

    model = elsa.ELSA(n_items=items_cnt, device=adm.settings.DEVICE_TORCH, n_dims=params['n_dims'])
    model.compile()
    model.load_state_dict(torch.load(f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("elsa", params)}.pt'))
    return model


def encode_sentences_train(*, params: adm.params.EncoderParams, train_test: Literal['train', 'test'] | None) -> str:
    save_path = f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("encode_sentences_train" + ((train_test.upper() + str(adm.settings.TRAIN_RATIO)) if type(train_test) == str else ""), params)}.npy'
    if os.path.exists(save_path):
        print('  - Encoded sentences train already exists')
        return save_path

    model, batch_size = params['model'], params['batch_size']

    q = adm.files.read_train(train_test)['Query'].unique(maintain_order=True).to_numpy()

    model = sentence_transformers.SentenceTransformer(model, device=adm.settings.DEVICE_TORCH, trust_remote_code=True)
    encoded = model.encode(q, device=adm.settings.DEVICE_TORCH, show_progress_bar=True, batch_size=batch_size)
    torch.save(encoded, save_path)

    return save_path


def load_encoded_sentences_train(
        *,
        params: adm.params.EncoderParams,
        train_test: Literal['train', 'test'] | None,
) -> tuple[pl.DataFrame, torch.Tensor]:
    df_train = adm.files.read_train(train_test)

    queries = df_train['Query'].unique(maintain_order=True)
    query_to_id = {q: i for i, q in enumerate(queries)}

    df = df_train.group_by(
        (
            ('Query', 'User') if train_test == 'test' else 'Query'
        ),
        maintain_order=True
    ).agg(
        pl.col('Item').first().alias('_item')
        if train_test == 'test' else
        pl.col('Item').value_counts(sort=True).struct[0].alias('_items')
    ).with_columns(
        pl.col('Query').replace(query_to_id, return_dtype=pl.Int32).alias('_q_index')
    ).drop('Query')

    encoded = torch.load(
        f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("encode_sentences_train" + ((train_test.upper() + str(adm.settings.TRAIN_RATIO)) if type(train_test) == str else ""), params)}.npy'
    )
    return df, encoded


def encode_sentences_batch(*, params: adm.params.EncoderParams, batch_id: int) -> str:
    save_path = f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("encode_sentences_batch_" + str(batch_id), params)}.npy'
    if os.path.exists(save_path):
        print(f'- Encoded sentences test {batch_id} already exists')
        return save_path

    model, batch_size = params['model'], params['batch_size']

    q: pl.DataFrame
    match batch_id:
        case 1:
            q = adm.files.read_b1()
        case 2:
            q = adm.files.read_b2()
        case 3:
            q = adm.files.read_bfinal()
        case _:
            raise ValueError(f'Got unexpected value batch_id: {batch_id}')

    model = sentence_transformers.SentenceTransformer(model, device=adm.settings.DEVICE_TORCH, trust_remote_code=True)
    encoded = model.encode(q['Query'].to_numpy(), device=adm.settings.DEVICE_TORCH, show_progress_bar=True, batch_size=batch_size)
    torch.save(encoded, save_path)

    return save_path


def load_encoded_sentences_batch(*, params: adm.params.EncoderParams, batch_id: int) -> tuple[pl.Series, torch.Tensor]:
    match batch_id:
        case 1:
            q = adm.files.read_b1()
        case 2:
            q = adm.files.read_b2()
        case 3:
            q = adm.files.read_bfinal()
        case _:
            raise ValueError(f'Got unexpected value batch_id: {batch_id}')

    encoded = torch.load(f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("encode_sentences_batch_" + str(batch_id), params)}.npy')
    return q['User'], encoded


def build_faiss_index(*, df_embeddings: tuple[pl.DataFrame, torch.Tensor], encoder_params: adm.params.EncoderParams) -> str:
    save_path = f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("faiss", encoder_params)}.npy'
    if os.path.exists(save_path):
        print('- Faiss index already exists')
        return save_path

    res = faiss.StandardGpuResources()

    _, embeddings = df_embeddings

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    faiss.normalize_L2(embeddings)
    gpu_index.add_with_ids(embeddings, np.arange(embeddings.shape[0]))
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), save_path)

    return save_path


def load_faiss_index(*, encoder_params: adm.params.EncoderParams) -> faiss.Index:
    res = faiss.StandardGpuResources()
    index = faiss.read_index(f'{adm.settings.MODELS_FOLDER}/{adm.params.params_to_name("faiss", encoder_params)}.npy')
    return faiss.index_cpu_to_gpu(res, 0, index)


def scipy_sparse_to_torch_sparse(scipy_csr_matrix: scipy.sparse.csr_matrix) -> torch.cuda.FloatTensor:
    data = scipy_csr_matrix.data
    indices = scipy_csr_matrix.indices
    indptr = scipy_csr_matrix.indptr

    torch_data = torch.tensor(data, dtype=torch.float32)
    torch_indices = torch.tensor(indices, dtype=torch.int32)
    torch_indptr = torch.tensor(indptr, dtype=torch.int32)

    torch_sparse_csr_matrix = torch.sparse_csr_tensor(
        torch_indptr,
        torch_indices,
        torch_data,
        size=scipy_csr_matrix.shape,
        device=adm.settings.DEVICE_TORCH,
    )

    return torch_sparse_csr_matrix
