from typing import Set, Tuple, Dict
import numpy as np


class TemporalNegativeSampler:
    def __init__(self, max_node_id, is_directed=True):
        self.is_directed = is_directed
        self.max_node_id = max_node_id
        self.added_edges: Set[Tuple[int, int]] = set()
        self.added_nodes: Set[int] = set()
        self.adj: Dict[int, Set[int]] = {}

    def sample_negative_edges(self, batch_sources, batch_targets, _batch_timestamps,
                               num_negatives, historical_negative_percentage=0.5):
        batch_sources = np.asarray(batch_sources)
        batch_targets = np.asarray(batch_targets)

        # Build current batch edge set
        current_batch = set(zip(batch_sources, batch_targets))
        if not self.is_directed:
            current_batch |= set(zip(batch_targets, batch_sources))

        hist_k = int(num_negatives * historical_negative_percentage)
        rand_k = num_negatives - hist_k

        negatives = np.empty((len(batch_sources), num_negatives), dtype=np.int32)

        for i, src in enumerate(batch_sources):
            negs = []

            # === Historical negatives ===
            if hist_k > 0:
                hist_targets = [
                    dst for (s, dst) in self.added_edges
                    if s == src and (s, dst) not in current_batch
                ]

                if len(hist_targets) == 0:
                    hist_targets = self._get_random_candidates(src, num_negatives)

                sample = np.random.choice(hist_targets, size=hist_k, replace=len(hist_targets) < hist_k)
                negs.extend(sample)

            # === Random negatives ===
            if rand_k > 0:
                rand_candidates = self._get_random_candidates(src, num_negatives)
                sample = np.random.choice(rand_candidates, size=rand_k, replace=len(rand_candidates) < rand_k)
                negs.extend(sample)

            negatives[i] = negs

        self._update_state(current_batch)
        return negatives

    def _get_random_candidates(self, src, num_negatives):
        current_neighbors = self.adj.get(src, set())
        candidates = list(self.added_nodes - current_neighbors - {src})
        if not candidates:
            candidates = np.random.randint(0, self.max_node_id + 1, size=num_negatives, dtype=np.int32)
            return candidates
        return np.array(candidates, dtype=np.int32)

    def _update_state(self, current_batch):
        for src, dst in current_batch:
            self.added_edges.add((src, dst))
            self.added_nodes.update([src, dst])
            if src not in self.adj:
                self.adj[src] = set()
            self.adj[src].add(dst)


def get_negatives(
        all_sources,
        all_targets,
        all_timestamps,
        is_directed,
        num_negatives,
        historical_negative_percentage):
    temporal_random_sampler = TemporalNegativeSampler(is_directed=is_directed)

    all_sources = np.asarray(all_sources)
    all_targets = np.asarray(all_targets)
    all_timestamps = np.asarray(all_timestamps)

    unique_timestamps = np.unique(all_timestamps)
    neg_sources = []
    neg_targets = []

    for ts in unique_timestamps:
        ts_mask = all_timestamps == ts
        ts_sources = all_sources[ts_mask]
        ts_targets = all_targets[ts_mask]
        ts_timestamps = all_timestamps[ts_mask]

        negatives = temporal_random_sampler.sample_negative_edges(
            ts_sources,
            ts_targets,
            ts_timestamps,
            num_negatives,
            historical_negative_percentage
        )

        for i, src in enumerate(ts_sources):
            for neg_target in negatives[i]:
                neg_sources.append(src)
                neg_targets.append(neg_target)

    return np.array(neg_sources), np.array(neg_targets)


def combine_edges(positive_sources, positive_targets, negative_sources, negative_targets):
    pos_sources = np.asarray(positive_sources)
    pos_targets = np.asarray(positive_targets)
    neg_sources = np.asarray(negative_sources)
    neg_targets = np.asarray(negative_targets)

    n_pos = len(pos_sources)
    n_neg = len(neg_sources)
    k = n_neg // n_pos  # Number of negatives per positive

    # Total edges after interleaving
    total_edges = n_pos * (k + 1)

    # Initialize result arrays
    combined_sources = np.empty(total_edges, dtype=pos_sources.dtype)
    combined_targets = np.empty(total_edges, dtype=pos_targets.dtype)
    combined_labels = np.empty(total_edges, dtype=bool)  # True=positive, False=negative

    # Process each positive edge with its k negatives
    for i in range(n_pos):
        # Get the k negatives for this positive edge
        neg_start = i * k

        # Randomly choose position for positive edge (0 to k inclusive)
        pos_position = np.random.randint(0, k + 1)

        # Fill the group of k+1 edges
        group_start = i * (k + 1)

        # Place negatives and positive edge
        neg_idx = 0
        for j in range(k + 1):
            result_idx = group_start + j

            if j == pos_position:
                # Place positive edge at random position
                combined_sources[result_idx] = pos_sources[i]
                combined_targets[result_idx] = pos_targets[i]
                combined_labels[result_idx] = True
            else:
                # Place negative edge
                combined_sources[result_idx] = neg_sources[neg_start + neg_idx]
                combined_targets[result_idx] = neg_targets[neg_start + neg_idx]
                combined_labels[result_idx] = False
                neg_idx += 1

    return combined_sources, combined_targets, combined_labels
