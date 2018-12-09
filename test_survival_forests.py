import numpy as np
import lifelines
import pandas as pd
from sksurv.preprocessing import OneHotEncoder # from preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split as tts

import lifelines.datasets
import time

def train_test_split(data, test_size=0.25):
    return tts(data, test_size=test_size) # shuffle=False)

def get_blcd(test_size=0.2):
    blcd = pd.read_csv("datasets/blcd.csv", engine='python')
    
    dataset_train, dataset_test = train_test_split(blcd, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "censor"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "time"]
        
        data_y = dataset[["time", "censor"]]
        data_y = data_y.reindex(columns=["censor", "time"])
        data_y["censor"] = data_y["censor"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"censor": "event"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_lnd(test_size=0.2):
    lnd = pd.read_csv("datasets/lnd.csv", engine='python')
    # lnd.drop([lnd.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(lnd, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "status"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "time"]
        
        data_y = dataset[["time", "status"]]
        data_y = data_y.reindex(columns=["status", "time"])
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_htd(test_size=0.2):
    htd = pd.read_csv("datasets/htd.csv", engine='python')
    htd.drop([htd.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(htd, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "status"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "time"]
        
        data_y = dataset[["time", "status"]]
        data_y = data_y.reindex(columns=["status", "time"])
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_veteran(test_size=0.2):
    veteran = pd.read_csv("datasets/veteran.csv", engine='python')
    veteran.drop([veteran.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(veteran, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "status"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "time"]
        # convert string columns to categorical type
        for col in data_x_numeric.columns:
            if str(data_x_numeric[col].dtype) == "object":
                data_x_numeric[col] = data_x_numeric[col].astype('category')
        data_x_numeric = OneHotEncoder().fit_transform(data_x_numeric)
        
        data_y = dataset[["time", "status"]]
        data_y = data_y.reindex(columns=["status", "time"])
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_pbc(test_size=0.2):
    pbc = pd.read_csv("datasets/pbc.csv", engine='python')
    pbc.drop([pbc.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(pbc, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "status"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "days"]
        # convert string columns to categorical type
        for col in data_x_numeric.columns:
            if str(data_x_numeric[col].dtype) == "object":
                data_x_numeric[col] = data_x_numeric[col].astype('category')
        data_x_numeric = OneHotEncoder().fit_transform(data_x_numeric)
        
        data_y = dataset[["days", "status"]]
        data_y = data_y.reindex(columns=["status", "days"])
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.rename(index=int, columns={"days": "time"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_gcd(test_size=0.2):
    gcd = pd.read_csv("datasets/gcd.csv", engine='python')
    gcd.drop([gcd.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(gcd, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "status"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "time"]
        
        data_y = dataset[["time", "status"]]
        data_y = data_y.reindex(columns=["status", "time"])
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_mbc(test_size=0.2):
    pbc = pd.read_csv("datasets/mbc.csv", engine='python')
    pbc.drop([pbc.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(pbc, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "Censoring"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "Time"]
        
        data_y = dataset[["Time", "Censoring"]]
        data_y = data_y.reindex(columns=["Censoring", "Time"])
        data_y = data_y.rename(index=int, columns={"Censoring": "status"})
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.rename(index=int, columns={"Time": "time"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_cml(test_size=0.2):
    cml = pd.read_csv("datasets/cml.csv", engine='python')
    cml.drop([cml.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(cml, test_size=test_size)
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "status"]
        data_x_numeric = data_x_numeric.loc[:, data_x_numeric.columns != "time"]
        # convert string columns to categorical type
        for col in data_x_numeric.columns:
            if str(data_x_numeric[col].dtype) == "object":
                data_x_numeric[col] = data_x_numeric[col].astype('category')
        data_x_numeric = OneHotEncoder().fit_transform(data_x_numeric)
        
        data_y = dataset[["status", "time"]]
        data_y = data_y.reindex(columns=["status", "time"])
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def get_breast_wpbc(test_size=0.2):
    breast = pd.read_csv("datasets/breast_wpbc.csv", engine='python')
    breast.drop([breast.columns[0]], axis = 1, inplace = True)
    
    dataset_train, dataset_test = train_test_split(breast, test_size=test_size)
    print("No time information is provided in dataset: the row number is used as time")
    
    def convert_dataset(dataset):
        data_x_numeric = dataset.loc[:, dataset.columns != "status"]
        dataset["days"] = pd.Series([i for i in range(len(dataset.index))], index=dataset.index)
        # no time information provided
        
        data_y = dataset[["days", "status"]]
        data_y = data_y.reindex(columns=["status", "days"])
        data_y["status"] = data_y["status"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"status": "event"})
        pd_y_values = pd_y_values.rename(index=int, columns={"days": "time"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))



def get_gbsg2(test_size=0.2):
    # load German Breast Cancer Study Group 2 Dataset
    gbsg2 = lifelines.datasets.load_gbsg2()
    dataset_train, dataset_test = train_test_split(gbsg2, test_size=test_size)
    
    def convert_dataset(dataset):
        # convert string columns to categorical type
        for col in dataset.columns:
            if str(dataset[col].dtype) == "object":
                dataset.loc[:, col] = dataset[col].astype('category')

        data_x_numeric = OneHotEncoder().fit_transform(dataset[["horTh", 
                                                             "age",
                                                             "menostat",
                                                             "tsize",
                                                             "tgrade",
                                                             "pnodes",
                                                             "progrec",
                                                             "estrec"]])
        data_y = dataset[["time", "cens"]]
        data_y = data_y.reindex(columns=["cens", "time"])
        data_y["cens"] = data_y["cens"].astype('bool')

        pd_y_values = data_y.copy()
        pd_y_values = pd_y_values.rename(index=int, columns={"cens": "event"})
        pd_y_values = pd_y_values.reindex(columns=["time", "event"])

        # test on sorted input data
        test_data = data_x_numeric.copy()
        test_timed_data = test_data
        test_timed_data['time'] = pd_y_values["time"]
        
        return data_x_numeric, pd_y_values, test_timed_data
    
    return (convert_dataset(dataset_train), convert_dataset(dataset_test))

def test_on(dataset, n_trees=50, n_trees_in_subset=5, max_features=2, max_depth=5, j_subspace=3000):
    train, test = dataset
    data_x_numeric, pd_y_values, _ = train
    _, test_pd_y_values, test_timed_data = test
    
    from survival import RandomSurvivalForest
    rf = RandomSurvivalForest(n_trees=n_trees,
                              n_trees_in_subset=n_trees_in_subset,
                              max_features=max_features,
                              max_depth=max_depth,
                              random_J_subspace_size=j_subspace)

    a_time = time.time()
    rf.fit(data_x_numeric, pd_y_values)
    b_time = time.time()
    fit_time = b_time - a_time
    print("Fit time: {}".format(fit_time))

    a_time = time.time()
    pred_hazards = rf.predict_weighted_proba(test_timed_data)
    b_time = time.time()
    print("Predict hazards (with weights): {}".format(b_time - a_time))

    
    ci = lifelines.utils.concordance_index(event_times=test_timed_data['time'],
                                           event_observed=test_pd_y_values["event"],
                                           predicted_scores=pred_hazards)
    print("Concordance index: {}".format(ci))

    a_time = time.time()
    pred_unweighted_hazards = rf.predict_proba(test_timed_data)
    b_time = time.time()
    print("Predict hazards (without weights): {}".format(b_time - a_time))

    ci2 = lifelines.utils.concordance_index(event_times=test_timed_data['time'],
                                            event_observed=test_pd_y_values["event"],
                                            predicted_scores=pred_unweighted_hazards)
    print("Concordance index: {}".format(ci2))
    
    return fit_time, ci, ci2
    
if __name__ == "__main__":
    # test_on()
    datasets = {
        #"cml": get_cml,
        #"gbsg2": get_gbsg2,
        #"pbc": get_pbc,
        #"mbc": get_mbc,
        #"wpbc": get_breast_wpbc,
        #"gcd": get_gcd,
        #"htd": get_htd,
        "veteran": get_veteran,
        #"lnd": get_lnd,
        #"blcd": get_blcd
    }
    test_sizes = [0.25]
    test_n_trees = [200]
    test_n_trees_in_subset = [2]
    test_max_depth = [2]
    test_max_features = [2]
    test_j_subspaces = [3000]
    names = []
    times = []
    weighted = []
    without_weights = []
    sizes = []
    all_j_subspaces = []

    all_n_trees = []
    all_n_trees_in_subset = []
    all_max_features = []
    all_max_depth = []
    for name, dataset in datasets.items():
        for n_trees in test_n_trees:
            for n_trees_in_subset in test_n_trees_in_subset:
                if n_trees_in_subset > n_trees_in_subset:
                    continue
                for max_features in test_max_features:
                    for max_depth in test_max_depth:
                        for test_size in test_sizes:
                            for test_j in test_j_subspaces:
                                all_n_trees.append(n_trees)
                                all_n_trees_in_subset.append(n_trees_in_subset)
                                all_max_features.append(max_features)
                                all_max_depth.append(max_depth)

                                names.append(name)
                                sizes.append(test_size)
                                all_j_subspaces.append(test_j)
    rows_number = len(names)
    
    for i in range(rows_number):
        test_size = sizes[i]
        n_trees = all_n_trees[i]
        n_trees_in_subset = all_n_trees_in_subset[i]
        max_features = all_max_features[i]
        max_depth = all_max_depth[i]
        j_subspace = all_j_subspaces[i]
        fit_time, ci_weighted, ci_without_weights = test_on(dataset(test_size=test_size),
                                                            n_trees=n_trees,
                                                            n_trees_in_subset=n_trees_in_subset,
                                                            max_features=max_features,
                                                            max_depth=max_depth,
                                                            j_subspace=j_subspace)
        times.append(fit_time)
        weighted.append(ci_weighted)
        without_weights.append(ci_without_weights)

    results = pd.DataFrame({"dataset_name": names[:len(times)],
                            "train_test_split_ratio": sizes[:len(times)],
                            "fit_time": np.array(times),
                            "ci_weighted": np.array(weighted),
                            "ci_rf": np.array(without_weights),
                            "n_trees": all_n_trees[:len(times)],
                            "n_trees_in_subset": all_n_trees_in_subset[:len(times)],
                            "max_features": all_max_features[:len(times)],
                            "max_depth": all_max_depth[:len(times)],
                            "j_subspace": all_j_subspaces[:len(times)]})
    print(results)
    results.to_csv("./results_" + str(time.time()) + ".csv")