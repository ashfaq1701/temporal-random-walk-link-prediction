from typing import Dict

import numpy as np
import torch

class GlobalEmbeddingStore:
    def __init__(self, max_node_id: int, embedding_dim: int, alpha: float = 0.75, device='cuda'):
        self.alpha = alpha
        self.device = device
        self.embeddings = torch.zeros(max_node_id + 1, embedding_dim, device=device)
        self.seen_mask = torch.zeros(max_node_id + 1, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, batch_embs: Dict[int, np.ndarray]):
        if not batch_embs:
            return

        node_ids = np.fromiter(batch_embs.keys(), dtype=np.int64)
        emb_array = np.stack([batch_embs[n] for n in node_ids])  # shape (N, D)

        node_ids = torch.tensor(node_ids, dtype=torch.long, device=self.device)
        batch_embs_tensor = torch.tensor(emb_array, dtype=torch.float32, device=self.device)

        # Determine which nodes have already been seen
        seen = self.seen_mask[node_ids]
        unseen = ~seen

        # 1. Directly assign embeddings to unseen nodes
        self.embeddings[node_ids[unseen]] = batch_embs_tensor[unseen]

        # 2. EMA update only for previously seen nodes
        if seen.any():
            old_emb = self.embeddings[node_ids[seen]]
            new_emb = batch_embs_tensor[seen]
            self.embeddings[node_ids[seen]] = self.alpha * old_emb + (1 - self.alpha) * new_emb

        # 3. Mark all nodes as seen
        self.seen_mask[node_ids] = True


    def get(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[node_ids.to(self.device)]
