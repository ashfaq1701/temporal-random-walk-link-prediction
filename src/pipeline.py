from logger import logger
from _temporal_random_walk import TemporalRandomWalk
from tgb.linkproppred.dataset import LinkPropPredDataset

from src.model import LinkPredictionModel


def start_link_prediction_experiment(
        dataset_name,
        is_directed,
        batch_time_duration,
        streaming_window_duration,
        walk_length,
        num_walks_per_node,
        edge_picker,
        embedding_dim,
        edge_op,
        negative_edges_per_positive,
        n_epochs,
        embedding_use_gpu,
        link_prediction_use_gpu,
        word2vec_n_workers,
        output_path,
        n_runs
):
    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
    dataset.load_val_ns()
    dataset.load_test_ns()

    for run in range(n_runs):
        logger.info(f"Running run # {run}")

        current_results = launch_run(
            dataset=dataset,
            is_directed=is_directed,
            batch_time_duration=batch_time_duration,
            streaming_window_duration=streaming_window_duration,
            walk_length=walk_length,
            num_walks_per_node=num_walks_per_node,
            edge_picker=edge_picker,
            embedding_dim=embedding_dim,
            edge_op=edge_op,
            negative_edges_per_positive=negative_edges_per_positive,
            n_epochs=n_epochs,
            embedding_use_gpu=embedding_use_gpu,
            link_prediction_use_gpu=link_prediction_use_gpu,
            word2vec_n_workers=word2vec_n_workers
        )


def launch_run(
        dataset: LinkPropPredDataset,
        is_directed,
        batch_time_duration,
        streaming_window_duration,
        walk_length,
        num_walks_per_node,
        edge_picker,
        embedding_dim,
        edge_op,
        negative_edges_per_positive,
        n_epochs,
        embedding_use_gpu,
        link_prediction_use_gpu,
        word2vec_n_workers,
):
    temporal_random_walk = TemporalRandomWalk(
        is_directed=is_directed,
        use_gpu=embedding_use_gpu,
        max_time_capacity=streaming_window_duration,
        enable_weight_computation=True
    )

    embedding_store = {}


def run_training_epoch(
        dataset: LinkPropPredDataset,
        is_directed,
        batch_time_duration,
        streaming_window_duration,
        walk_length,
        num_walks_per_node,
        edge_picker,
        embedding_dim,
        edge_op,
        negative_edges_per_positive,
        embedding_use_gpu,
        link_prediction_use_gpu,
        word2vec_n_workers
):
    temporal_random_walk = TemporalRandomWalk(
        is_directed=is_directed,
        use_gpu=embedding_use_gpu,
        max_time_capacity=streaming_window_duration,
        enable_weight_computation=True
    )

    embedding_store = {}

    model = LinkPredictionModel(input_dim=embedding_dim)


