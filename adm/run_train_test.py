import csv
import itertools
import json
import os
import sys
import time

import numba
import numpy as np
import polars as pl
import tqdm

import torch
import faiss

import adm.files
import adm.models
import adm.params
import adm.settings


@numba.njit(
    numba.float64(
        numba.types.Array(numba.types.int32, 1, 'C'),
        numba.types.Array(numba.types.int32, 2, 'C')
    ),
    cache=True,
)
def mrr(query_array: np.ndarray, target_array: np.ndarray) -> np.float32:
    total = np.float32(0.0)
    for j in range(query_array.shape[0]):
        for i in range(target_array.shape[1]):
            if target_array[j, i] == query_array[j]:
                total += 1.0 / (i + 1)
                break
    return total / len(query_array)

#
# @torch.jit.script
# def torch_sparse_interactions_recc(
#         interactions: torch.Tensor,
#         a: torch.Tensor,
#         user_indexes: torch.Tensor,
#         user_items: torch.Tensor,
#         user_weights: torch.Tensor,
#         device: torch.device,
# ):
#     a_tsp = a.T
#     results = torch.zeros([user_indexes.shape[0], user_items.shape[1]], dtype=torch.float32, device=device)
#     for i in range(user_indexes.shape[0]):
#         idx = int(user_indexes[i].item())
#         results[i] = (((interactions[idx] @ a) @ a_tsp) - interactions[idx])[user_items[i]]
#     user_weights += results


@torch.jit.script
def torch_sparse_interactions_recc(
        interactions: torch.Tensor,
        a: torch.Tensor,
        user_indexes: torch.Tensor,
        user_items: torch.Tensor,
        user_weights: torch.Tensor,
        arranged: torch.Tensor,
        ones: torch.Tensor,
        device: torch.device,
):
    # Number of users in the total interaction matrix
    total_users = interactions.shape[0]

    # Number of queries
    num_queries = user_indexes.shape[0]

    # Create indices for sparse matrix in CSR format
    crow_indices = arranged[:num_queries + 1]
    col_indices = user_indexes
    values = ones[:num_queries]

    # Create a sparse CSR tensor for the selection matrix
    selection_matrix = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(num_queries, total_users), device=device)

    # Use the sparse selection matrix to select the interactions for the users in users_for_query
    selected_interactions = selection_matrix @ interactions
    selected_interactions_dense = selected_interactions.to_dense()

    # Compute the dot products and the difference
    result = (((selected_interactions @ a) @ a.T) - selected_interactions_dense)

    # Select the items for the result using advanced indexing
    selected_result = result[arranged[:num_queries, None], user_items]

    # Update the weights tensor with the selected result
    user_weights += selected_result


def predict_batch(
        *,
        predict_params: adm.params.PredictParams,
        enc_batch: tuple[pl.Series, torch.Tensor],
        df_train: tuple[pl.DataFrame, torch.Tensor],
        idx: faiss.Index,
        a: torch.Tensor,
        interactions: torch.Tensor,
) -> np.ndarray:
    n_items, multiplier, cut_items = predict_params['n_items'], predict_params['multiplier'], predict_params[
        'cut_items']
    users_series, tensor_query_data = enc_batch

    results = torch.zeros((len(users_series), 100), dtype=torch.int32, device=adm.settings.DEVICE_TORCH)

    arranged = torch.arange(2 ** 14, dtype=torch.int32, device=adm.settings.DEVICE_TORCH)
    ones = torch.ones_like(arranged, dtype=torch.float32, device=adm.settings.DEVICE_TORCH)

    faiss_result: tuple[list[list[float]], list[list[int]]] = idx.search(
        tensor_query_data,
        n_items,
    )
    distances_all_queries_all, query_indexes_all_queries_all = faiss_result

    for idx_start, idx_end in ((i, i + 2048) for i in tqdm.tqdm(range(0, len(tensor_query_data), 2048), position=1)):
        distances_all_queries, query_indexes_all_queries = distances_all_queries_all[idx_start:idx_end], query_indexes_all_queries_all[idx_start:idx_end]
        n = len(distances_all_queries)
        user_indexes = torch.zeros(n, dtype=torch.int32, device=adm.settings.DEVICE_TORCH)
        user_items = torch.zeros((n, cut_items), dtype=torch.int32, device=adm.settings.DEVICE_TORCH)
        user_weights = torch.zeros((n, cut_items), dtype=torch.float32, device=adm.settings.DEVICE_TORCH)

        for i, (distances, found_query) in enumerate(zip(distances_all_queries, query_indexes_all_queries)):
            query_idx = idx_start + i

            items_tmp = []
            weights_tmp = []
            for dist, q_i in zip(distances, found_query):
                found = df_train[0][int(q_i), '_items']
                items_tmp.extend(found)
                weights_tmp.extend(np.repeat(float(dist) * multiplier, len(found)))

            assert len(items_tmp) >= 100

            order_indices = np.unique(items_tmp, return_index=True)[1]
            order_indices.sort()
            order_indices = order_indices[:cut_items]

            assert len(order_indices) >= 100, f'{len(order_indices)} < 100'

            stuff_indices = len(order_indices) if len(order_indices) < cut_items else cut_items
            user_indexes[i] = users_series[query_idx]
            user_items[i, :stuff_indices] = torch.tensor(items_tmp, dtype=torch.int32, device=adm.settings.DEVICE_TORCH)[order_indices]
            user_weights[i, :stuff_indices] = torch.tensor(weights_tmp, dtype=torch.float32, device=adm.settings.DEVICE_TORCH)[order_indices]

        torch_sparse_interactions_recc(
            interactions=interactions,
            a=a,
            user_indexes=user_indexes,
            user_items=user_items,
            user_weights=user_weights,
            arranged=arranged,
            ones=ones,
            device=adm.settings.DEVICE_TORCH,
        )

        result = user_items[torch.arange(len(user_items), device=adm.settings.DEVICE_TORCH)[:, None], torch.argsort(user_weights, dim=1, descending=True)[:, :100]]
        assert result.shape == (len(user_items), 100), f'{result.shape} != (n, 100)'

        results[idx_start:idx_end] = result

    return results.cpu().numpy()


@torch.jit.script
def torch_predict_train_test(
        users_for_query: torch.Tensor,
        interactions: torch.Tensor,
        a: torch.Tensor,
        items: torch.Tensor,
        weights: torch.Tensor,
        arranged: torch.Tensor,
        ones: torch.Tensor,
        device: torch.device,
):
    # Number of users in the total interaction matrix
    total_users = interactions.shape[0]

    # Number of queries
    num_queries = users_for_query.shape[0]

    # Create indices for sparse matrix in CSR format
    crow_indices = arranged[:num_queries + 1]
    col_indices = users_for_query
    values = ones[:num_queries]

    # Create a sparse CSR tensor for the selection matrix
    selection_matrix = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(num_queries, total_users), device=device)

    # Use the sparse selection matrix to select the interactions for the users in users_for_query
    selected_interactions = selection_matrix @ interactions
    selected_interactions_dense = selected_interactions.to_dense()

    # Compute the dot products and the difference
    result = (((selected_interactions @ a) @ a.T) - selected_interactions_dense)

    # Select the items for the result using advanced indexing
    selected_result = result[:, items]

    # Update the weights tensor with the selected result
    weights += selected_result


def predict_train_test(
        *,
        predict_params: adm.params.PredictParams,
        enc_train_test: tuple[pl.DataFrame, torch.Tensor] | None,
        # DataFrame contains not grouped columns ['User', '_q_index', '_items'] and it's guaranteed
        # that the [User, _q_index] is a unique combination
        df_train: tuple[pl.DataFrame, torch.Tensor],
        idx: faiss.Index,
        a: torch.Tensor,
        interactions: torch.Tensor,
) -> np.float32:
    n_items, multiplier, cut_items = predict_params['n_items'], predict_params['multiplier'], predict_params[
        'cut_items']

    # DataFrame contains not grouped columns ['User', '_q_index', '_item']
    # Tensor of unique queries
    user_query_df, tensor_query_data = enc_train_test

    query_idx_to_users_items_with_index: dict[int, tuple[np.ndarray, np.ndarray]] = {
        qid: (
            np.fromiter((x['User'] for x in users_items_with_index), dtype=np.int32),
            np.fromiter((x['_item'] for x in users_items_with_index), dtype=np.int32),
        )
        for qid, users_items_with_index in
        zip(*user_query_df.with_row_index().group_by('_q_index').agg(pl.struct('User', '_item')))
    }

    current_user_index = 0
    results = torch.zeros((len(user_query_df), 100), dtype=torch.int32, device=adm.settings.DEVICE_TORCH)
    results_correct = torch.zeros(len(user_query_df), dtype=torch.int32, device=adm.settings.DEVICE_TORCH)

    faiss_results: tuple[list[list[float]], list[list[int]]] = idx.search(
        tensor_query_data,
        n_items,
    )
    distances, found_queries = faiss_results
    assert len(distances) == len(tensor_query_data)
    assert len(found_queries) == len(tensor_query_data)
    assert len(distances[0]) == n_items
    assert len(found_queries[0]) == n_items
    assert len(tensor_query_data) == len(distances) == len(found_queries)

    arranged = torch.arange(2 ** 14, dtype=torch.int32, device=adm.settings.DEVICE_TORCH)
    ones = torch.ones_like(arranged, dtype=torch.float32, device=adm.settings.DEVICE_TORCH)

    idx_dst_fq_list = list(zip(range(len(tensor_query_data)), distances, found_queries))
    for query_idx, distance, found_query in tqdm.tqdm(idx_dst_fq_list, position=1):
        items_tmp = []
        weights_tmp = []
        for dist, q_i in zip(distance, found_query):
            found = df_train[0][int(q_i), '_items']
            items_tmp.extend(found)
            weights_tmp.extend(np.repeat(float(dist) * multiplier, len(found)))

        assert len(items_tmp) >= 100

        order_indices = np.unique(items_tmp, return_index=True)[1]
        order_indices.sort()
        order_indices = order_indices[:cut_items]

        assert len(order_indices) >= 100, f'{len(order_indices)} < 100'

        users_for_query_cpu, correct_items_for_query_cpu = query_idx_to_users_items_with_index[query_idx]
        users_for_query = torch.from_numpy(users_for_query_cpu).to(device=adm.settings.DEVICE_TORCH, non_blocking=True)
        correct_items_for_query = torch.from_numpy(correct_items_for_query_cpu).to(device=adm.settings.DEVICE_TORCH, non_blocking=True)

        items = torch.tensor(items_tmp, dtype=torch.int32)[order_indices].to(device=adm.settings.DEVICE_TORCH, non_blocking=True)
        weights = torch.tile(
            torch.tensor(weights_tmp, dtype=torch.float32)[order_indices].to(device=adm.settings.DEVICE_TORCH, non_blocking=True),
            (users_for_query.shape[0], 1),
        )

        torch_predict_train_test(
            users_for_query,
            interactions,
            a,
            items,
            weights,
            arranged,
            ones,
            device=adm.settings.DEVICE_TORCH,
        )

        results[current_user_index:current_user_index + users_for_query.shape[0], :] = items[
            torch.argsort(weights, dim=1, descending=True)[:, :100]]
        results_correct[current_user_index:current_user_index + users_for_query.shape[0]] = correct_items_for_query

        current_user_index += len(users_for_query)

    return mrr(results_correct.cpu().numpy(), results.cpu().numpy())


def single_run_from_trained_models(
        elsa_params: adm.params.ElsaParams,
        encoder_params: adm.params.EncoderParams,
        predict_params: adm.params.PredictParams,
) -> None:
    els = adm.models.load_elsa(params=elsa_params, train_test='train' if adm.settings.TRAIN_TEST else None)
    df_train = adm.models.load_encoded_sentences_train(params=encoder_params,
                                                       train_test='train' if adm.settings.TRAIN_TEST else None)

    idx = adm.models.load_faiss_index(encoder_params=encoder_params)

    a = torch.nn.functional.normalize(els.get_items_embeddings(), dim=-1)
    interactions = adm.models.scipy_sparse_to_torch_sparse(
        adm.models.make_sparse_interactions_matrix(train_test='train' if adm.settings.TRAIN_TEST else None)
    )

    combined_params = {
        'elsa_params': elsa_params,
        'encoder_params': encoder_params,
        'predict_params': predict_params,
    }
    name = adm.params.params_to_name('params', combined_params)

    try:
        os.mkdir(f'{adm.settings.RESULTS_FOLDER}/{name}')
    except FileExistsError:
        pass
    with open(f'{adm.settings.RESULTS_FOLDER}/{name}/params.json', 'w') as f:
        f.write(json.dumps(combined_params, indent=4))

    for batch_id in (adm.settings.BATCHES if not adm.settings.TRAIN_TEST else [None]):
        print(f'- Predicting batch {batch_id if batch_id is not None else "test"}')

        if batch_id is not None:
            result_path = f'{adm.settings.RESULTS_FOLDER}/{name}/submission_batch{batch_id}.csv'
            if os.path.exists(result_path):
                print('- Submission already exists')
                continue
            enc_batch = adm.models.load_encoded_sentences_batch(params=encoder_params, batch_id=batch_id)
            res = predict_batch(
                predict_params=predict_params,
                enc_batch=enc_batch,
                df_train=df_train,
                idx=idx,
                a=a,
                interactions=interactions,
            )
            with open(result_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(res)
        else:
            result_path = f'{adm.settings.RESULTS_FOLDER}/{name}/mrr.txt'
            if os.path.exists(result_path):
                print('- MRR already exists')
                continue
            enc_train_test = adm.models.load_encoded_sentences_train(params=encoder_params, train_test='test')
            res = predict_train_test(
                predict_params=predict_params,
                enc_train_test=enc_train_test,
                df_train=df_train,
                idx=idx,
                a=a,
                interactions=interactions,
            )
            print(f'- MRR: {res:.10f}')

            with open(result_path, 'w') as f:
                f.write(f'{res:.10f}\n')


def run(runner_id: int, num_runners: int) -> None:
    print(f'STARTING RUNNER {runner_id + 1} / {num_runners}')
    time.sleep(5)

    try:
        os.mkdir(adm.settings.RESULTS_FOLDER)
    except FileExistsError:
        pass

    if adm.settings.TRAIN_TEST:
        params = list(itertools.product(
                adm.params.get_params_elsa(),
                adm.params.get_params_encoder(),
                adm.params.get_params_predict(),
        ))
        params_use = [
            p
            for i, p in enumerate(params)
            if i % adm.settings.NUM_RUNNERS == adm.settings.RUNNER_ID
        ]
    else:
        params_use = adm.params.get_params_best()
    
    print(f'WILL RUN {len(params_use)} PARAMS')

    for elsa_params, encoder_params, predict_params in tqdm.tqdm(params_use, position=0):
        adm.settings.init_rng(adm.settings.SEED)
        print(f'RUNNING {elsa_params} {encoder_params} {predict_params}')

        single_run_from_trained_models(
            elsa_params=elsa_params,
            encoder_params=encoder_params,
            predict_params=predict_params,
        )


if __name__ == '__main__':
    arg_runner_id = 0
    arg_num_runners = 1

    if len(sys.argv) == 2:
        arg_runner_id = int(sys.argv[1])
        arg_num_runners = adm.settings.NUM_RUNNERS
        assert 0 <= arg_runner_id < arg_num_runners

    if len(sys.argv) == 3:
        arg_runner_id = int(sys.argv[1])
        arg_num_runners = int(sys.argv[2])
        assert 0 <= arg_runner_id < arg_num_runners

    run(arg_runner_id, arg_num_runners)
