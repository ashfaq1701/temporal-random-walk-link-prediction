import argparse

from src.pipeline import start_link_prediction_experiment

if __name__ == 'main':
    parser = argparse.ArgumentParser(description="Temporal Link Prediction")

    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='TGB dataset name')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed (true/false)')

    parser.add_argument('--batch_time_duration', type=int, required=True, help='Batch time duration in number of time steps')
    parser.add_argument('--streaming_window_duration', type=int, required=True, help='Streaming window duration in number of time steps')

    # Model parameters
    parser.add_argument('--walk_length', type=int, default=80,
                        help='Maximum length of random walks')
    parser.add_argument('--num_walks_per_node', type=int, default=10,
                        help='Number of walks to generate per node')
    parser.add_argument('--edge_picker', type=str, default='ExponentialIndex',
                        help='Edge picker. Can be ExponentialIndex, ExponentialWeight, Linear, Uniform')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimensionality of node embeddings')
    parser.add_argument('--edge_op', type=str, default='hadamard',
                        choices=['average', 'hadamard', 'weighted-l1', 'weighted-l2'],
                        help='Edge operation for combining node embeddings')

    parser.add_argument('--negative_edges_per_positive', type=int, default=1,
                        help='Number of negative edges per positive edge')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs for neural network training')

    # GPU settings
    parser.add_argument('--embedding_use_gpu', action='store_true',
                        help='Enable GPU acceleration for embedding approach')
    parser.add_argument('--link_prediction_use_gpu', action='store_true',
                        help='Enable GPU acceleration for link prediction neural network')

    # Other settings
    parser.add_argument('--word2vec_n_workers', type=int, default=8,
                        help='Number of workers for Word2Vec training')
    parser.add_argument('--output_path', type=str, default=None,
                        help='File path to save results (optional)')

    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of experimental runs for averaging results')

    args = parser.parse_args()

    start_link_prediction_experiment(
        dataset_name=args.dataset_name,
        is_directed=args.is_directed,
        batch_time_duration=args.batch_time_duration,
        streaming_window_duration=args.streaming_window_duration,
        walk_length=args.walk_length,
        num_walks_per_node=args.num_walks_per_node,
        edge_picker=args.edge_picker,
        embedding_dim=args.embedding_dim,
        edge_op=args.edge_op,
        negative_edges_per_positive=args.negative_edges_per_positive,
        n_epochs=args.n_epochs,
        embedding_use_gpu=args.embedding_use_gpu,
        link_prediction_use_gpu=args.link_prediction_use_gpu,
        word2vec_n_workers=args.word2vec_n_workers,
        output_path=args.output_path,
        n_runs=args.n_runs
    )
