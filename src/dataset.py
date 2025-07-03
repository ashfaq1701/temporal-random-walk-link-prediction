from tgb.linkproppred.dataset import LinkPropPredDataset
import numpy as np


def get_train_batches(dataset: LinkPropPredDataset, batch_time_duration: int):
    full_data = dataset.full_data

    train_sources = full_data['sources'][dataset.train_mask]
    train_targets = full_data['destinations'][dataset.train_mask]
    train_timestamps = full_data['timestamps'][dataset.train_mask]

    sorted_indices = np.argsort(train_timestamps)
    train_sources = train_sources[sorted_indices]
    train_targets = train_targets[sorted_indices]
    train_timestamps = train_timestamps[sorted_indices]

    min_ts = train_timestamps[0]
    max_ts = train_timestamps[-1]

    current_start = min_ts
    current_end = min_ts + batch_time_duration

    while current_start <= max_ts:
        mask = (train_timestamps >= current_start) & (train_timestamps < current_end)
        if np.any(mask):
            yield {
                'sources': train_sources[mask],
                'targets': train_targets[mask],
                'timestamps': train_timestamps[mask]
            }

        current_start = current_end
        current_end += batch_time_duration
