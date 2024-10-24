#pragma once
#include <cstdio>
#include <memory>
#include <vector>
#include <string>

struct Sample {
  int label;
  std::vector<double> nums;
  std::vector<int> cats;
};
struct TreeNode {
  enum class Type { LEAF, NUM, CAT };
  Type type;
  union {
    int label;
    double threshold;
    int cat;
  } value;
  size_t feat_idx;
  std::unique_ptr<TreeNode> ls, rs;
};

inline TreeNode leaf_node(int label) {
  return TreeNode{.type = TreeNode::Type::LEAF,
                  .value = {label},
                  .feat_idx = 0,
                  .ls = nullptr,
                  .rs = nullptr};
}
inline TreeNode num_node(size_t feat_idx, double threshold) {
  return TreeNode{.type = TreeNode::Type::NUM,
                  .value = {.threshold = threshold},
                  .feat_idx = feat_idx,
                  .ls = nullptr,
                  .rs = nullptr};
}
inline TreeNode cat_node(size_t feat_idx, int cat) {
  return TreeNode{.type = TreeNode::Type::CAT,
                  .value = {.cat = cat},
                  .feat_idx = feat_idx,
                  .ls = nullptr,
                  .rs = nullptr};
}
class DecisionTree {
public:
  struct FitOptions {
    int max_depth;
    int min_samples_leaf{4};
    int min_samples_split{4};
    double min_purity_decrease{0.1};

  } options;
  size_t feat_nums, feat_cats, labels;
  std::vector<std::string> num_names, cat_names;
  void fit(size_t n, Sample *samples, FitOptions const &options);
  int predict(Sample const &samples);
  void show();
  DecisionTree(size_t numerical_feat, size_t categorical_feat, size_t labels, std::vector<std::string> num_names, std::vector<std::string> cat_names) : feat_nums(numerical_feat), feat_cats(categorical_feat), labels(labels) {}
  private:
  std::unique_ptr<TreeNode> _fit(size_t n, Sample *samples, int depth);
  std::unique_ptr<TreeNode> inner_;
  void _show(TreeNode &node, int depth);
};
