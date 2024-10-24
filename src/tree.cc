#include "tree.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <print>
#include <vector>


std::unique_ptr<TreeNode> DecisionTree::_fit(size_t n, Sample *samples,
                                             int depth) {
  #ifdef DEBUG
  puts("");
  puts("");
  printf("fitting with %lu samples\n", n);
  #endif
  std::vector<size_t> counts;
  std::vector<size_t> now_counts;
  counts.resize(this->labels);

  // count all labels
  for (size_t idx = 0; idx < n; ++idx) {
    counts[samples[idx].label]++;
  }

  // check if all labels are the same
  size_t most_freq = 0;
  for (size_t idx = 0; idx < this->labels; ++idx) {
    if (counts[idx] == n) {
      return std::make_unique<TreeNode>(leaf_node(idx));
    }
    if (counts[idx] > counts[most_freq]) {
      most_freq = idx;
    }
  }

  if (depth >= this->options.max_depth ||
      n <= this->options.min_samples_split) {
    return std::make_unique<TreeNode>(leaf_node(most_freq));
  }
  int split_type = 0; // 1 for num, 2 for cat
  size_t feat_idx = 0;
  double threshold = 0;
  int cate = 0;

  double minloss = 1e20;
  double init_loss = 0;
  for (size_t k = 0; k < this->labels; ++k) {
    init_loss -= (counts[k] * 1.0 / n) * log(counts[k] * 1.0 / n);
  }

  if (this->feat_nums > 0) {
    now_counts.resize(this->labels);
    for (size_t idx = 0; idx < this->feat_nums; ++idx) {
      // sort by this feature
      std::sort(samples, samples + n, [&](Sample const &a, Sample const &b) {
        return a.nums[idx] < b.nums[idx];
      });
      // now calculate loss dynamically
      double loss = 0;
      // reset to zero
      std::memset(now_counts.data(), 0, sizeof(size_t) * this->labels);
      double last = samples[0].nums[idx];
      for (size_t i = 0; i < n; ++i) {
        // split between [0, i] and (i, n)
        // only ends on unique values
        now_counts[samples[i].label]++;
        while (i + 1 < n && samples[i + 1].nums[idx] == samples[i].nums[idx]) {
          ++i;
          now_counts[samples[i].label]++;
        }
        // all nodes in left tree, break
        if (i >= n - 1)
          break;
        // left tree size i + 1, right tree size n - i - -1
        loss = 0;
        #ifdef DEBUG
        std::print("NUM split at feat {} num {}\n", idx, samples[i].nums[idx]);
        #endif 
        for (size_t k = 0; k < this->labels; ++k) {
          // left
        #ifdef DEBUG
          std::print("class {}, now counts {}\n", k, now_counts[k]);
        #endif 
          if (now_counts[k] > 0) {
            auto pmf_left = 1.0 * now_counts[k] / (i + 1);
            loss -= 1.0*(i+1)/n * pmf_left * std::log(pmf_left);
        #ifdef DEBUG
            std::print("left total {}, class {}, loss contri {}\n", i + 1,
                       now_counts[k], -pmf_left * std::log(pmf_left));
        #endif 
          }
          // right
          if (counts[k] - now_counts[k] > 0) {
            auto pmf_right = 1.0 * (counts[k] - now_counts[k]) / (n - i - 1);
            loss -= 1.0*(n-i-1)/n * pmf_right * std::log(pmf_right);
        #ifdef DEBUG
            std::print("right total {}, class {}, loss contri {}\n", n - i - 1,
                       n - now_counts[k], -pmf_right * std::log(pmf_right));
        #endif 
          }
        }
        #ifdef DEBUG
        std::print("init loss {} split at feat {} num {}, loss {}\n", init_loss,
                   idx, samples[i].nums[idx], loss);
        puts("");
        #endif 
        if (loss < minloss) {
          minloss = loss;
          split_type = 1;
          threshold = samples[i].nums[idx];
          feat_idx = idx;
        }
      }
    }
  }
  if (this->feat_cats > 0) {
    now_counts.resize(this->labels);
    for (size_t idx = 0; idx < this->feat_cats; ++idx) {
      // sort by this feature
      std::sort(samples, samples + n, [&](Sample const &a, Sample const &b) {
        return a.cats[idx] < b.cats[idx];
      });
      // now calculate loss dynamically
      double loss = 0;
      // reset to zero
      for (size_t i = 0; i < n; ++i) {
        #ifdef DEBUG
        std::print("CAT split at feat {} cate {}\n", idx, samples[i].cats[idx]);
        #endif
        std::memset(now_counts.data(), 0, sizeof(size_t) * this->labels);
        // calculate counts in this cateogry
        now_counts[samples[i].label]++;
        size_t to = i;
        while (to + 1 < n &&
               samples[to + 1].cats[idx] == samples[to].cats[idx]) {
          ++to;
          now_counts[samples[to].label]++;
        }
        auto now_size = to - i + 1;
        auto remain_size = n - now_size;
        i = to;
        loss = 0;
        for (size_t k = 0; k < this->labels; ++k) {
        #ifdef DEBUG
          std::print("nowcount[{}] = {}, counts[{}] = {}\n", k, now_counts[k], k, counts[k]);
          #endif
          if (now_counts[k] > 0) {
            auto pmf_now = 1.0 * now_counts[k] / now_size;
            loss -= 1.0 * now_size / n * pmf_now * std::log(pmf_now);
        #ifdef DEBUG
            std::print("left total {}, class {}, loss contri {}\n", now_size,
                       now_counts[k], -pmf_now * std::log(pmf_now));
            #endif
          }
          if (counts[k] - now_counts[k] > 0) {
            auto pmf_remain = 1.0 * (counts[k] - now_counts[k]) / remain_size;
            loss -= 1.0 * remain_size / n * pmf_remain * std::log(pmf_remain);
        #ifdef DEBUG
            std::print("right total {}, class {}, loss contri {}\n", remain_size,
                       counts[k] - now_counts[k], -pmf_remain * std::log(pmf_remain));
            #endif
          }
        }
        if (loss < minloss) {
          minloss = loss;
          split_type = 2;
          cate = samples[i].cats[idx];
          feat_idx = idx;
        }
      }
    }
  }
  if ((1.0*(init_loss - minloss)/init_loss <= this->options.min_purity_decrease)) {
    #ifdef DEBUG
    std::print("init {}, min {} purged by purity", init_loss, minloss);
    #endif
    return std::make_unique<TreeNode>(leaf_node(most_freq));
  }
  if (split_type == 1) {
    // partition samples by threshold
    size_t p = 0, q = 1;
    while (samples[p].nums[feat_idx] <= threshold)
      ++p;
    q = p + 1;
    while (q < n) {
      if (samples[q].nums[feat_idx] <= threshold) {
        std::swap(samples[p], samples[q]);
        ++p;
      }
      ++q;
    }
    #ifdef DEBUG
    std::print("Best NUM, threshold {} feat {} minloss {}\n", threshold, feat_idx, minloss);
    #endif
    if (p < this->options.min_samples_leaf ||
        n - p < this->options.min_samples_leaf) {
      return std::make_unique<TreeNode>(leaf_node(most_freq));
    }
    // now [0, p) <= threshold, [p, n) > threshold
    auto cur = std::make_unique<TreeNode>(num_node(feat_idx, threshold));
    cur->ls = this->_fit(p, samples, depth + 1);
    cur->rs = this->_fit(n - p, samples + p, depth + 1);
    return cur;
  }
  // partition sample by category
  size_t p = 0, q = 1;
  while (q < n) {
    if (samples[q].cats[feat_idx] == cate) {
      std::swap(samples[p], samples[q]);
      ++p;
    }
    ++q;
  }
  #ifdef DEBUG
  std::print("Best CAT, cateogry {} feat {} minloss {}\n", cate, feat_idx, minloss);
  #endif
  if (p < this->options.min_samples_leaf ||
      n - p < this->options.min_samples_leaf) {
    return std::make_unique<TreeNode>(leaf_node(most_freq));
  }
  auto cur = std::make_unique<TreeNode>(cat_node(feat_idx, cate));
  cur->ls = this->_fit(p, samples, depth + 1);
  cur->rs = this->_fit(n - p, samples + p, depth + 1);
  return cur;
}

void DecisionTree::fit(size_t n, Sample *samples, FitOptions const &options) {
  this->options = options;
  this->inner_ = this->_fit(n, samples, 0);
}

void DecisionTree::show() {
  // recursively show tree
  this->_show(*this->inner_, 0);
}
void DecisionTree::_show(TreeNode &node, int depth) {
  for (int i = 0; i < depth; ++i) {
    putchar(' ');
  }
  if (node.type == TreeNode::Type::LEAF) {
    std::print("LEAF label {}\n", node.value.label);
    return;
  }
  if (node.type == TreeNode::Type::NUM) {
    std::print("NUM feat {} < threshold {}\n", node.feat_idx,
               node.value.threshold);
    if (node.ls)
      _show(*node.ls, depth + 1);
    if (node.rs)
      _show(*node.rs, depth + 1);
    return;
  }
  std::print("CAT feat {} cate = {}\n", node.feat_idx, node.value.cat);
  if (node.ls)
    _show(*node.ls, depth + 1);
  if (node.rs)
    _show(*node.rs, depth + 1);
}

int DecisionTree::predict(Sample const &sample) {
  auto *now = this->inner_.get();
  while (now->type != TreeNode::Type::LEAF) {
    if (now->type == TreeNode::Type::NUM) {
      if (sample.nums[now->feat_idx] <= now->value.threshold) {
        now = now->ls.get();
      } else {
        now = now->rs.get();
      }
    } else {
      if (sample.cats[now->feat_idx] == now->value.cat) {
        now = now->ls.get();
      } else {
        now = now->rs.get();
      }
    }
  }
  return now->value.label;
}

/*int main() {
  Sample samples[10];
  for (size_t i = 0; i < 10; ++i) {
    samples[i].nums.push_back(i);
    samples[i].cats.push_back(i & 1);
    samples[i].cats.push_back(i < 7);
    samples[i].label = (i < 5) || (!(i & 1) && (i < 7));
  }
  DecisionTree dt(1, 2, 2, {"test"}, {});
  dt.fit(10, samples,
         {.max_depth = 4,
          .min_samples_leaf = 1,
          .min_samples_split = 1,
          .min_purity_decrease = 0.1});
  dt.show();
  for (size_t i = 0; i < 10; ++i) {
    std::print("origin {}, pred {}\n", samples[i].label,
               dt.predict(samples[i]));
  }
  return 0;
}
*/
