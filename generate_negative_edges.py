import argparse

from src.utils import sample_negative_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Temporal negative edge generator")

    parser.add_argument('--dataset_name', type=str, required=True,
                        help='TGB dataset name')
    parser.add_argument('--is_directed', type=lambda x: x.lower() == 'true', required=True,
                        help='Whether the graph is directed (true/false)')

    parser.add_argument('--output_path', type=str, required=True,
                        help='File path to save the results ')

    parser.add_argument('--negative_edges_per_positive', type=int, required=True,
                        help='Number of negative edges per positive edge')

    parser.add_argument('--historical_negative_percentage', type=float, default=0.5,
                        help='Percentage of historical negative edges')

    args = parser.parse_args()

    dataset, (positive_edges, negative_edges) = sample_negative_dataset(
        args.dataset_name,
        args.is_directed,
        args.save_path,
        num_negatives_per_positive=args.negative_edges_per_positive,
        historical_negative_percentage=args.historical_negative_percentage
    )

    print(f'Generated {len(positive_edges)} negative edges.')
