#ifndef NEGATIVEEDGESAMPLER_H
#define NEGATIVEEDGESAMPLER_H

#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <random>
#include <unordered_set>
#include <vector>
#include <utility>

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::size_t seed = std::hash<int>{}(p.first);
        seed ^= std::hash<int>{}(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

class NegativeEdgeSampler {
    bool is_directed;
    tbb::concurrent_unordered_set<int> all_nodes;
    tbb::concurrent_unordered_set<int> added_nodes;
    tbb::concurrent_unordered_set<std::pair<int, int>, PairHash> added_edges;
    tbb::concurrent_unordered_map<int, tbb::concurrent_unordered_set<int>> adj;

    std::vector<int> get_random_candidates(int src);
    void update_state(const tbb::concurrent_unordered_set<std::pair<int, int>, PairHash>& current_batch);

public:
    NegativeEdgeSampler(const std::unordered_set<int>& all_node_ids, bool is_directed);

    std::pair<std::vector<int>, std::vector<int>> sample_negative_edges_per_batch(
        const std::vector<int>& batch_sources,
        const std::vector<int>& batch_targets,
        int num_negatives_per_positive,
        double historical_negative_percentage
    );
};

std::pair<std::vector<int>, std::vector<int>> collect_all_negatives_by_timestamp(
    const std::vector<int>& sources,
    const std::vector<int>& targets,
    const std::vector<int64_t>& timestamps,
    int num_negatives_per_positive,
    double historical_negative_percentage,
    bool is_directed = false
);

#endif // NEGATIVEEDGESAMPLER_H
