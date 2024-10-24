#include "interface.hpp"
#include "tree.hpp"
#include <cstring>
#include <vector>


extern "C" void *create_tree(uint64_t numerical_feat, uint64_t categorical_feat,
                             uint64_t labels) {
  std::vector<std::string> num_names;
  for (auto i = 0; i < numerical_feat; ++i) {
    num_names.push_back(std::to_string(i));
  }
  std::vector<std::string> cat_names;
  for (auto i = 0; i < categorical_feat; ++i) {
    cat_names.push_back(std::to_string(i));
  }
  auto *res = new DecisionTree(numerical_feat, categorical_feat, labels,
                               num_names, cat_names);
  return res;
}

extern "C" void *create_fit_options(int32_t max_depth, int32_t min_samples_leaf,
                                    int32_t min_samples_split,
                                    double min_purity_decrease) {
  return new DecisionTree::FitOptions{max_depth, min_samples_leaf,
                                      min_samples_split, min_purity_decrease};
}
extern "C" void tree_fit(void *tree, void *sampleset,
                         void*options) {
  DecisionTree *now = reinterpret_cast<DecisionTree *>(tree);
  auto *samples = reinterpret_cast<std::vector<Sample> *>(sampleset);
  auto *cur_options =
      reinterpret_cast<const DecisionTree::FitOptions *>(options);
  now->fit(samples->size(), samples->data(), *cur_options);
}

extern "C" int32_t tree_predict(void *tree, void *sample) {
  DecisionTree *now = reinterpret_cast<DecisionTree *>(tree);
  auto *s = reinterpret_cast<std::vector<Sample>*>(sample);
  return now->predict((*s).at(0));
}

extern "C" void delete_tree(void *tree) {
  delete reinterpret_cast<DecisionTree *>(tree);
}
extern "C" void* create_sampleset() {
  return new std::vector<Sample>;
}
extern "C" void create_sample(void* sampleset, int32_t label, uint64_t numerical_num, double* nums, uint64_t categorical_numbers, int32_t* cats) {
    // copy to create one sample
    Sample sample;
    sample.label = label;
    sample.nums = std::vector<double>(numerical_num);
    sample.cats = std::vector<int>(categorical_numbers);
    for(auto i = 0; i < numerical_num; ++i) {
        sample.nums[i] = nums[i];
    }
    for(auto i = 0; i < categorical_numbers; ++i) {
        sample.cats[i] = cats[i];
    }
    std::vector<Sample> *samples = reinterpret_cast<std::vector<Sample>*>(sampleset);
    samples->emplace_back(sample);
}
extern "C" void delete_sampleset(void *sampleset) {
  delete reinterpret_cast<std::vector<Sample> *>(sampleset);
}

extern "C" void delete_fit_options(void*options) {
  delete reinterpret_cast<DecisionTree::FitOptions *>(options);
}

extern "C" void show_tree(void*tree) {
  DecisionTree *now = reinterpret_cast<DecisionTree *>(tree);
  now->show();
}