import pandas as pd
import numpy as np
import math
import cvxpy as cp # optimization framework
from time import time as timestamp
from tqdm import tqdm

try:
    import fast_implementations
except Exception as e:
    print("Can't load fast_implementations module: {}".format(e))

class RandomSurvivalForest():

    def __init__(self, n_trees=10, n_trees_in_subset=2, max_features=2, max_depth=5, min_samples_split=2,
                 random_J_subspace_size=3000, verbose=True, disable_opt=False, lam=0.01,
                 random_seed=None, split_type="median"):
        """
            Initialize the Weighted Random Survival Forest.

            :param int n_trees: Total number of trees in the forest.
            :param int n_trees_in_subset: The number of trees in a subset.
            :param int max_features: The number of features to consider for a split.
            :param int max_depth: The maximum depth of the tree.
            :param int min_samples_split: The minimum number of samples required to split at a node.
            :param int random_J_subspace_size: The maximum number of optimization restrictions.
            :param bool verbose: Enable log printing.
            :param bool disable_opt: Disable optimization and weights generation.
                                     Weights values will be set to 1/(number of subsets).
            :param float lam: Optimization regularization parameter.
            :param int random_seed: Random seed.
            :param str | float split_type: Splitting rule ("best" or "median").
                                           If `split_type` is a float value, then randomly selected part
                                           of samples will be used, where `split_type` is a ratio of randomly
                                           selected samples size to whole samples number.
                                        
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_trees_in_subset = n_trees_in_subset
        self.n_subsets = n_trees // n_trees_in_subset
        self.all_events_time = None
        self.fix_tree_features = False # Randomly select features for each tree, or select randomly at each split

        self.simplex_lower_boundary = 1.0 / (self.n_trees * 5.0) # 1.0 / (self.n_subsets ** 2)
        self.optimization_lambda = lam # 0.01 # 1.0 # 10
        self.reduce_problem_dimensionality = False # True
        self.random_J_subspace_size = random_J_subspace_size # -1 to disable

        self.disable_opt = disable_opt # disable weights generation

        self.random_seed = int(timestamp()) if random_seed is None else random_seed
        self.rs = np.random.RandomState(self.random_seed + 293847) # or self.rs = np.random

        self.split_type = split_type # "best", "median", "random"

        self.verbose = verbose
        if self.verbose:
            self.print = print
        else:
            self.print = lambda x: x

    def logrank(self, x, feature):
        """
            Perform logrank test.

            :param DataFrame x: Samples data frame.
            :param str feature: Column name.

            :return: logrank score.
        """
        start_time = timestamp()
        c = x[feature].median()
        if x[x[feature] <= c].shape[0] < self.min_samples_split or x[x[feature] > c].shape[0] < self.min_samples_split:
            return 0
        t = list(set(x["time"]))
        get_time = {t[i]:i for i in range(len(t))}
        N = len(t)
        y = np.zeros((3, N), dtype=np.int)
        d = np.zeros((3, N), dtype=np.int)
        feature_inf = x[x[feature] <= c]
        feature_sup = x[x[feature] > c]
        count_sup = np.zeros((N, 1), dtype=np.int)
        count_inf = np.zeros((N, 1), dtype=np.int)
        for _, r in feature_sup.iterrows():
            t_idx = get_time[r["time"]]
            count_sup[t_idx] = count_sup[t_idx] + 1
            if r["event"]:
                d[2][t_idx] = d[2][t_idx] + 1
        for _, r in feature_inf.iterrows():
            t_idx = get_time[r["time"]]
            count_inf[t_idx] = count_inf[t_idx] + 1
            if r["event"]:
                d[1][t_idx] = d[1][t_idx] + 1
        nb_inf = feature_inf.shape[0]
        nb_sup = feature_sup.shape[0]

        L = fast_implementations.compute_logrank_sum(N, nb_inf, nb_sup, count_inf, count_sup, y, d)

        end_time = timestamp()
        # self.print("   Logrank time: {}".format(end_time - start_time))
        return abs(L)

    def logrank_non_median_split(self, x, feature, split_value):
        """
            Perform logrank test for specified split value.

            :param DataFrame x: Samples data frame.
            :param str feature: Column name.
            :param float split_value: The value for splitting.
                                      Logrank score will be calculated for two subsets after splitting.

            :return: logrank score.
        """
        start_time = timestamp()
        c = split_value
        if x[x[feature] <= c].shape[0] < self.min_samples_split or x[x[feature] > c].shape[0] < self.min_samples_split:
            return 0
        t = list(set(x["time"]))
        get_time = {t[i]:i for i in range(len(t))}
        N = len(t)
        y = np.zeros((3, N), dtype=np.int)
        d = np.zeros((3, N), dtype=np.int)
        feature_inf = x[x[feature] <= c]
        feature_sup = x[x[feature] > c]
        count_sup = np.zeros((N, 1), dtype=np.int)
        count_inf = np.zeros((N, 1), dtype=np.int)
        for _, r in feature_sup.iterrows():
            t_idx = get_time[r["time"]]
            count_sup[t_idx] = count_sup[t_idx] + 1
            if r["event"]:
                d[2][t_idx] = d[2][t_idx] + 1
        for _, r in feature_inf.iterrows():
            t_idx = get_time[r["time"]]
            count_inf[t_idx] = count_inf[t_idx] + 1
            if r["event"]:
                d[1][t_idx] = d[1][t_idx] + 1
        nb_inf = feature_inf.shape[0]
        nb_sup = feature_sup.shape[0]

        L = fast_implementations.compute_logrank_sum(N, nb_inf, nb_sup, count_inf, count_sup, y, d)

        end_time = timestamp()
        # self.print("   Logrank time: {}".format(end_time - start_time))
        return abs(L)

    def find_best_feature_and_split(self, x, look_ratio=1.0):
        """
            Find best feature and split value.

            :param DataFrame x: Samples data frame.
            :param float look_ratio: Ratio of randomly selected subset size to `x` size

            :return: Tuple of (feature, split_value).
        """
        start_time = timestamp()
        features = [f for f in x.columns if f not in ["time", "event"]]
        if not self.fix_tree_features:
            # features = list(np.random.permutation(features))[:self.max_features]
            features = list(self.rs.permutation(features))[:self.max_features]
        feature_split_pairs = [] # zip([], [])
        for feature in features:
            values = x[feature].unique()
            if not (len(values) < 2 * self.min_samples_split + 1):
                values.sort()
                values = values[self.min_samples_split:len(values) - self.min_samples_split]
                values = list(self.rs.permutation(values))[:round(len(values) * look_ratio)]
            for val in values:
                feature_split_pairs.append((feature, val))
        information_gains = [self.logrank_non_median_split(x, f[0], f[1]) for f in feature_split_pairs]
        highest_ig = max(information_gains)
        end_time = timestamp()
        # self.print("    Find best feature time: {}".format(end_time - start_time))
        if highest_ig == 0:
            return None
        else:
            return feature_split_pairs[information_gains.index(highest_ig)]

    def find_best_feature(self, x):
        """
            Find the best feature for a median split value.

            :param DataFrame x: Samples data frame.

            :return: The best feature name.
        """
        start_time = timestamp()
        features = [f for f in x.columns if f not in ["time", "event"]]
        if not self.fix_tree_features:
            # features = list(np.random.permutation(features))[:self.max_features]
            features = list(self.rs.permutation(features))[:self.max_features]
        information_gains = [self.logrank(x, feature) for feature in features]
        highest_ig = max(information_gains)
        end_time = timestamp()
        # self.print("    Find best feature time: {}".format(end_time - start_time))
        if highest_ig == 0:
            return None
        else:
            return features[information_gains.index(highest_ig)]

    def compute_leaf(self, x, tree):
        """
            Compute counts for the leaf.

            :param DataFrame x: Samples data frame.
            :param dict tree: Leaf subtree.
        """
        count_times = []
        count_0 = []
        count_1 = []
        x_sorted = x.sort_values(by="time")
        prev_time = -1
        for _, r in x_sorted.iterrows():
            if prev_time != r["time"]:
                prev_time = r["time"]
                count_times.append(r["time"])
                if r["event"]:
                    count_0.append(0)
                    count_1.append(1)
                else:
                    count_0.append(1)
                    count_1.append(0)
            else:
                if r["event"]:
                    count_1[-1] += 1
                else:
                    count_0[-1] += 1
        t = count_times
        t.sort()
        total = x.shape[0]
        tree["count_0"] = np.array(count_0)
        tree["count_1"] = np.array(count_1)
        tree["t"] = np.array(t)
        tree["total"] = total

    def build(self, in_x, in_tree, in_depth):
        """
            Build a new tree.

            :param DataFrame in_x: Samples data frame.
            :param dict in_tree: Preallocated tree object.
            :param in_depth: Initial depth of tree.
        """
        tree = in_tree
        depth = in_depth
        x = in_x

        stack = [(x, tree, depth)]

        while len(stack) > 0:
            x, tree, depth = stack.pop()
            unique_targets = pd.unique(x["time"])

            if len(unique_targets) == 1 or depth == self.max_depth:
                self.compute_leaf(x, tree)
                continue

            if self.split_type == "median":
                best_feature = self.find_best_feature(x)

                if best_feature == None:
                    self.compute_leaf(x, tree)
                    continue

                feature_median = x[best_feature].median()
            elif self.split_type == "best":
                best_feature_and_split = self.find_best_feature_and_split(x)

                if best_feature_and_split == None:
                    self.compute_leaf(x, tree)
                    continue

                best_feature, feature_median = best_feature_and_split
            elif type(self.split_type) == float:
                best_feature_and_split = self.find_best_feature_and_split(x, look_ratio=self.split_type)

                if best_feature_and_split == None:
                    self.compute_leaf(x, tree)
                    continue

                best_feature, feature_median = best_feature_and_split
            else:
                raise Exception("Incorrect split type")
        
            tree["feature"] = best_feature
            tree["median"] = feature_median

            left_split_x = x[x[best_feature] <= feature_median]
            right_split_x = x[x[best_feature] > feature_median]
            split_dict = [["left", left_split_x], ["right", right_split_x]]
        
            for name, split_x in split_dict:
                tree[name] = {}
                stack.append((split_x, tree[name], depth + 1))

    def fit(self, x, event):
        """
            Fit the Random Survival Forest and perform optimization to compute weights.

            :param DataFrame x: Samples data frame.
            :param DataFrame event: Events data frame, containing time and censorship labels.

            :return: Random Survival Forest.
        """
        self.trees = [{} for i in range(self.n_trees)]
        event.columns = ["time", "event"]
        features = list(x.columns)
        x = pd.concat((x, event), axis=1)
        x = x.sort_values(by="time")
        x.index = range(x.shape[0])
        rs = np.random.RandomState(self.random_seed)
        for i in range(self.n_trees): # tqdm(range(self.n_trees)):
            sampled_x = x.sample(frac = 1, replace = True, random_state=rs)
            sampled_x.index = range(sampled_x.shape[0])
            if self.fix_tree_features:
                # sampled_features = list(np.random.permutation(features))[:self.max_features] + ["time","event"]
                sampled_features = list(self.rs.permutation(features))[:self.max_features] + ["time","event"]
                # self.print("Build {} tree".format(i))
                start_time = timestamp()
                self.build(sampled_x[sampled_features], self.trees[i], 0)
                end_time = timestamp()
            else:
                self.build(sampled_x, self.trees[i], 0)
            # self.print("  Build time: {}".format(end_time - start_time))
        self.make_weights(x)
        self.all_events_time = event[["time"]].copy().sort_values(by="time")["time"]

        return self


    def make_weights(self, x):
        """
            Compute weights for subsets of trees according to Concordance index minimization.

            :param DataFrame x: Samples data frame.
        """
        self.weights = [(1.0 / self.n_subsets) for i in range(self.n_subsets)]
        if self.disable_opt:
            return False
        
        J = []
        row_count = 0
        for i in range(len(x)):
            if x["event"][i]:
                row_count += 1
                for j in range(i + 1, len(x)):
                    J.append((i, j))
                    if self.reduce_problem_dimensionality:
                        if x["event"][j]:
                            break

        self.print("i indices: {}".format(row_count))
        self.print("Indices computed: #J = {}".format(len(J)))
        if self.random_J_subspace_size > 0:
            tmp_J = np.array(J)
            indices = np.random.choice(len(J), min(self.random_J_subspace_size, len(J)), replace=False)
            J = list(tmp_J[list(indices)])
            self.print("Indices after subspace selection: #J = {}".format(len(J)))


        x = x.drop(columns=["event"])
        self.time_column = list(x.columns)[-1] # for predicting
        deltas = []
        for q in range(len(self.weights)):
            deltas_row = []
            for (i, j) in J:
                left = self.predict_trees_subset_row(q, x.loc[i])
                right = self.predict_trees_subset_row(q, x.loc[j])
                delta = left - right
                deltas_row.append(delta)
            deltas.append(deltas_row)

        self.print("Deltas computed: #deltas = {}".format(len(deltas)))

        deltas = cp.Constant(value=deltas)

        w = cp.Variable(len(self.weights))
        xi = cp.Variable(len(J))
        lam = self.optimization_lambda # 0.001
        M = len(J)

        objective = cp.Minimize(cp.sum(xi) / M + lam * cp.norm2(w))
        constraints = [xi >= 0, xi >= deltas * w, cp.sum(w) == 1, w >= self.simplex_lower_boundary]
        prob = cp.Problem(objective, constraints)

        self.print("Problem is prepared")

        result = prob.solve()
        self.print("Problem status: {}".format(prob.status))
        self.weights = list(w.value.flatten())
        self.print("Weights: {}".format(self.weights))

    def compute_survival(self, row, tree):
        """
            Compute survival function for a tree leaf.

            :param DataFrame row: Sample row.
                                  Only `self.time_column` column is used.
            :param dict tree: Tree leaf.
        """
        count = tree["count"]
        t = tree["t"]
        total = tree["total"]
        h = 1
        survivors = float(total)
        for ti in t:
            if ti <= row[self.time_column]:
                h = h * (1 - count[(ti, 1)] / survivors)
            survivors = survivors - count[(ti, 1)] - count[(ti, 0)]
        return h

    def compute_hazard(self, row, tree):
        """
            Compute hazard function for a tree leaf.

            :param DataFrame row: Sample row.
                                  Only `self.time_column` column is used.
            :param dict tree: Tree leaf.
        """
        count = tree["count"]
        t = tree["t"]
        total = tree["total"]
        h = 0
        survivors = float(total)
        for ti in t:
            if ti <= row[self.time_column]:
                h += count[(ti,1)] / survivors
            survivors = survivors - count[(ti,1)] - count[(ti,0)]
        return h

    def compute_hazard_by_time(self, time, tree):
        """
            Compute survival function for a tree leaf.

            :param float time: Time moment.
            :param dict tree: Tree leaf.
        """
        count = tree["count"]
        t = tree["t"]
        total = tree["total"]
        h = 0
        survivors = float(total)
        for ti in t:
            if ti <= time:
                h += count[(ti,1)] / survivors
            survivors = survivors - count[(ti,1)] - count[(ti,0)]
        return h
    
    def predict_row_recursive(self, tree, row):
        """
            Make prediction for `tree` for `row` sample.

            :param dict tree: Tree object.
            :param DataFrame row: Data sample.
        """
        if "count_0" in tree:
            return self.compute_hazard(row, tree)
    
        if row[tree["feature"]] > tree["median"]:
            return self.predict_row(tree["right"], row)
        else:
            return self.predict_row(tree["left"], row)

    def predict_row(self, tree, row):
        """
            Make non-recursive prediction for `tree` for `row` sample.

            :param dict tree: Tree object.
            :param DataFrame row: Data sample.
        """
        cur_tree = tree

        while True: # assuming that tree is built correctly
            if "count_0" in cur_tree:
                #return self.compute_hazard_by_time(row[self.time_column], cur_tree)
                return fast_implementations.compute_hazard(row[self.time_column],
                                                           cur_tree["count_0"],
                                                           cur_tree["count_1"],
                                                           cur_tree["t"],
                                                           cur_tree["total"])
        
            if row[cur_tree["feature"]] > cur_tree["median"]:
                cur_tree = cur_tree["right"]
            else:
                cur_tree = cur_tree["left"]

    def predict_trees_subset_row(self, subset, row):
        """
            Make prediction for trees subset for `row` sample.

            :param int subset: The number of trees subset.
            :param DataFrame row: Data sample.
        """
        fst = subset * self.n_trees_in_subset
        subset_prediction = [self.predict_row(self.trees[i], row) for i in range(fst, fst + self.n_trees_in_subset)]
        return sum(subset_prediction) / self.n_trees_in_subset

    def predict_proba(self, x):
        """
            Make prediction for RSF for `x` samples.

            :param DataFrame x: Data samples.
        """
        self.time_column = list(x.columns)[-1]
        compute_trees = [x.apply(lambda u: self.predict_row(self.trees[i], u), axis=1) for i in range(self.n_trees)]
        return sum(compute_trees) / self.n_trees

    def predict_weighted_proba(self, x):
        """
            Make prediction for weighted RSF for `x` samples.

            :param DataFrame x: Data samples.
        """
        self.time_column = list(x.columns)[-1]
        Hs = [x.apply(lambda u: self.predict_trees_subset_row(i, u), axis=1) for i in range(self.n_subsets)]
        weighted = [hs * self.weights[i] for i, hs in enumerate(Hs)]
        return sum(weighted)

    def predict_survival(self, x):
        """
            Predict survival function for `x` samples

            :param DataFrame x: Data samples.
        """
        cumulative_hazard = self.predict_proba(x)
        return math.e ** (-cumulative_hazard)

    def hazard_to_survival(self, hazard):
        return math.e ** (-hazard)

    def predict_weighted_survival(self, x):
        """
            Predict weighted survival function for `x` samples

            :param DataFrame x: Data samples.
        """
        cumulative_hazard = self.predict_weighted_proba(x)
        return self.hazard_to_survival(cumulative_hazard) # math.e ** (-cumulative_hazard)

    def predict(self, x):
        return self.predict_weighted_survival(x)

    def predict_survival_function(self, x):
        test_data = x.iloc[[0]]
        test_data = test_data.append([test_data] * (self.all_events_time.count() - 1), ignore_index=True)
        test_data["time"] = self.all_events_time
        return self.predict_weighted_survival(test_data)
