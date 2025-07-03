import logger
import warnings
from contextlib import contextmanager


class EarlyStopping:
    def __init__(self, mode='min', patience=5, min_delta=0.0001, restore_best_weights=True):
        assert mode in ['min', 'max']
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, current_score, model):
        improved = (current_score < self.best_score - self.min_delta) if self.mode == 'min' else (
                current_score > self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


@contextmanager
def suppress_word2vec_output():
    # Save current logging levels
    gensim_logger = logging.getLogger('gensim')
    word2vec_logger = logging.getLogger('gensim.models.word2vec')
    kv_logger = logging.getLogger('gensim.models.keyedvectors')

    original_gensim_level = gensim_logger.level
    original_word2vec_level = word2vec_logger.level
    original_kv_level = kv_logger.level

    try:
        # Suppress logging
        gensim_logger.setLevel(logging.ERROR)
        word2vec_logger.setLevel(logging.ERROR)
        kv_logger.setLevel(logging.ERROR)

        # Also suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        # Restore original levels
        gensim_logger.setLevel(original_gensim_level)
        word2vec_logger.setLevel(original_word2vec_level)
        kv_logger.setLevel(original_kv_level)

