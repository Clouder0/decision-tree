#include <cstdint>

extern "C" void *create_tree(uint64_t numerical_feat, uint64_t categorical_feat,
                             uint64_t labels);
extern "C" void *create_fit_options(int32_t max_depth, int32_t min_samples_leaf,
                                    int32_t min_samples_split,
                                    double min_purity_decrease);
extern "C" void delete_fit_options(void *options);
extern "C" void tree_fit(void *tree, void *sampleset,
                         void *options);
extern "C" void *create_sampleset();
extern "C" void create_sample(void *sampleset, int32_t label,
                              uint64_t numerical_num, double *nums,
                              uint64_t categorical_numbers, int32_t *cats);
extern "C" void delete_sampleset(void *sampleset);
extern "C" int32_t tree_predict(void *tree, void *sampleset);
extern "C" void delete_tree(void *tree);

extern "C" void show_tree(void*tree);