# libary imports
import csv, operator, sys, os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, \
    average_precision_score
from pandas.core.common import flatten
import joblib
import argparse
import time

# own file imports
from Utils import create_exp_folder, store_exp_data_dict, store_exp_data_write, bit_error_rates_generator, parse_args
from bfi_evaluation import bfi_tree, bfi_forest
from prepareExpData import getData


def main():
    ### Preparations and configs
    # paths to train and test
    this_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parse_args(parser)
    args = parser.parse_args()

    # command line arguments, use argparse here later
    datasets = args.dataset

    # DT/RF configs
    DT_RFs = args.model  # DT or RF (needs to be correctly specified when loading a model)
    depths = args.depth  # DT/RF depth (single value for DT, list for RFs)
    estimss = args.estims  # number of DTs in RF (does not matter for DT)
    split_inj = args.splitval_inj  # activate split value injection with 1
    feature_inj = args.featval_inj  # activate feature value injection with 1
    feature_idx_inj = args.featidx_inj  # activate feature idx injection with 1
    child_idx_inj = args.chidx_inj  # activate child idx injection with 1
    nr_bits_split = args.nr_bits_split  # nr of bits in split value
    nr_bits_feature = args.nr_bits_feature  # nr of bits in feature value
    int_split = args.int_split  # whether to use integer split (use 1 to activate)
    reps = args.trials  # how many times to evaluate for one bit error rate
    export_accuracy = args.export_accuracy  # 1 if accuracy list for a bit error rate should be exported as .npy, else None
    all_data = []
    store_model = args.store_model
    load_model = args.load_model_path
    true_majority = args.true_majority
    random_state = args.seed  # np.random.randint(low=1, high=100)
    np.random.seed(random_state)
    # p2exp = 6 # error rates for evaluation start at 2^(-p2exp)
    # bers = bit_error_rates_generator(p2exp)
    bers = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    #bers = [0.1, 0.2, 0.4]
    #bers = [0]
    plot_histogram = None  # plots histogram of input data (useful for quantization)

    #Added for complete trees/whatever
    complete_trees = args.complete_trees        # 1 if complete trees should be used
    complete_redundant_trees = args.complete_redundant_trees        # 1 if complete redundant trees should be used
    if(complete_redundant_trees == 1):
        complete_trees = 1
    #complete_trees = 1
    rsdts = args.rsdt
    resilience = args.resilience
    timing = args.timing
    time_start = 0
    time_end = 0

    summarize = args.summarize                  # 1 if accuracy scores should be summarized over all bers
    #exact_chidx_error = args.exact_chidx_error  # 1 if chidx shouldn't be aborted immediately

    #print("run_exp.py().complete_trees = ", complete_trees)
    #print("random state: ", random_state)

    #print(datasets)

    # create experiment folder and return the path to it
    exp_path = create_exp_folder(this_path)

    for dataset in datasets:

        # read data
        #print(dataset)
        X_train, y_train, X_test, y_test = getData(dataset, this_path, nr_bits_split, nr_bits_feature, random_state)



        for DT_RF in DT_RFs:

            for depth in depths:

                if DT_RF == "RF":
                    estims_list = estimss
                else:
                    estims_list = {1}

                for estims in estims_list:

                    complete_redundant = "/"

                    if complete_redundant_trees==1:
                        complete_redundant = "/CR_"
                    elif complete_redundant==1:
                        complete_redundant = "/C_"


                    rsdts_path = ""
                    if DT_RF == "RF":
                        rsdts_path = exp_path + complete_redundant + f"{dataset}_D{depth}_T{estims}_rsdts.txt"
                        #rsdts_path = exp_path + f"/{dataset}_D{depth}_T{estims}_rsdts.txt"
                        print(f"/{dataset}_D{depth}_T{estims}")
                    else:
                        rsdts_path = exp_path + complete_redundant + f"{dataset}_D{depth}_rsdts.txt"
                        #rsdts_path = exp_path + f"/{dataset}_D{depth}_rsdts.txt"
                        print(f"/{dataset}_D{depth}")

                    output = [
                        "BERs,  ",
                        "0.0000", "0.0001", "0.0010", "0.0100", "0.1000", "0.2000", "0.4000", "0.6000", "0.8000",
                        "1.0000"
                    ]
                    for rsdt in rsdts:
                        output[0] += "rsdt" + str(rsdt) + ","
                        if(rsdt==0 or rsdt==5):
                            output[0] += " "

                    """
                    output = [
                        "BERs,  rsdt0, rsdt5, rsdt10,rsdt15,rsdt20,rsdt25,rsdt30,rsdt35,rsdt40,rsdt45,rsdt50",
                        "0.0000", "0.0001", "0.0010", "0.0100", "0.1000", "0.2000", "0.4000", "0.6000", "0.8000",
                        "1.0000"
                    ]
                    """
                    with open(rsdts_path, 'w+') as f:
                        for line in output:
                            f.write(f"{line}\n")

                    for rsdt in rsdts:

                        # TODO: loop over models here, and use multiprocessing
                        # train or load tree / forest
                        if load_model is not None:
                            model = joblib.load(load_model)
                        else:
                            if DT_RF == "DT":
                                clf = DecisionTreeClassifier(max_depth=depth, rsdt=rsdt,
                                                             complete_trees=complete_trees,
                                                             complete_redundant_trees=complete_redundant_trees,
                                                             random_state=random_state)
                                if timing:
                                    time_start = time.time()
                                model = clf.fit(X_train, y_train, rsdt=rsdt, complete_trees=complete_trees,
                                                complete_redundant_trees=complete_redundant_trees)
                                if timing:
                                    time_end = time.time()
                                    print("Building time: ", time_end - time_start)
                                if store_model is not None:
                                    joblib.dump(model, exp_path + f"/{dataset}_D{depth}_rsdt{rsdt}.pkl", compress=9)

                            elif DT_RF == "RF":
                                clf = RandomForestClassifier(max_depth=depth, n_estimators=estims, rsdt=rsdt, complete_trees=complete_trees,
                                                             complete_redundant_trees=complete_redundant_trees,
                                                             random_state=random_state)
                                if timing:
                                    time_start = time.time()
                                model = clf.fit(X_train, y_train, rsdt=rsdt, complete_trees=complete_trees,
                                                complete_redundant_trees=complete_redundant_trees)
                                if timing:
                                    time_end = time.time()
                                    print("Building time: ", time_end - time_start)
                                if store_model is not None:
                                    joblib.dump(model, exp_path + f"/{dataset}_D{depth}_T{estims}_rsdt{rsdt}.pkl", compress=9)
                            """
                            if DT_RF == "RSD_TEST":
                                with open('output.txt', 'w+') as f:
                                    output = [
                                        "BERs,  rsdt0, rsdt5, rsdt10,rsdt15,rsdt20,rsdt25,rsdt30,rsdt35,rsdt40,rsdt45,rsdt50",
                                        "0.0000", "0.0001", "0.0010", "0.0100", "0.1000", "0.2000", "0.4000", "0.6000", "0.8000",
                                        "1.0000"
                                        ]
                                    for line in output:
                                        f.write(f"{line}\n")
                                for rsdt in range(0, 51, 5):
                                    clf = RandomForestClassifier(max_depth=depth, n_estimators=estims, rsdt=rsdt,
                                                                 complete_trees=complete_trees,
                                                                 complete_redundant_trees=complete_redundant_trees,
                                                                 random_state=random_state)
                                    model = clf.fit(X_train, y_train, rsdt, complete_trees=complete_trees,
                                                    complete_redundant_trees=complete_redundant_trees)
                                    if store_model is not None:
                                        joblib.dump(model, exp_path + f"/D{depth}_T{estims}_{dataset}.pkl", compress=9)
                                    expdata_dict = {
                                        "DT_RF": DT_RF,
                                        "model": model,
                                        "depth": depth,
                                        "estims": estims,
                                        "dataset_name": dataset,
                                        "X_train": X_train,
                                        "X_test": X_test,
                                        "y_train": y_train,
                                        "y_test": y_test,
                                        "experiment_path": exp_path,
                                        "split_inj": split_inj,
                                        "int_split": int_split,
                                        "feature_inj": feature_inj,
                                        "nr_bits_split": nr_bits_split,
                                        "nr_bits_feature": nr_bits_feature,
                                        "feature_idx_inj": feature_idx_inj,
                                        "child_idx_inj": child_idx_inj,
                                        "reps": reps,
                                        "bers": bers,
                                        "export_accuracy": export_accuracy,
                                        "true_majority": true_majority,
                                        "random_state": random_state,
                                        #Added by me
                                        "complete_trees": complete_trees,
                                        "complete_redundant_trees": complete_redundant_trees,
                                        "summarize": summarize,
                                        "rsdt": rsdt,
                                        "resilience": resilience
                                    }
                                    bfi_forest(expdata_dict)
                            """
                        # create data file to store experiment results
                        exp_data = open(exp_path + "/results.txt", "a")
                        exp_data.close()

                        # dictionary for experiment data
                        expdata_dict = {
                            "DT_RF": DT_RF,
                            "model": model,
                            "depth": depth,
                            "estims": estims,
                            "dataset_name": dataset,
                            "X_train": X_train,
                            "X_test": X_test,
                            "y_train": y_train,
                            "y_test": y_test,
                            "experiment_path": exp_path,
                            "split_inj": split_inj,
                            "int_split": int_split,
                            "feature_inj": feature_inj,
                            "nr_bits_split": nr_bits_split,
                            "nr_bits_feature": nr_bits_feature,
                            "feature_idx_inj": feature_idx_inj,
                            "child_idx_inj": child_idx_inj,
                            "reps": reps,
                            "bers": bers,
                            "export_accuracy": export_accuracy,
                            "true_majority": true_majority,
                            "random_state": random_state,
                            #Added by me
                            "complete_trees": complete_trees,
                            "complete_redundant_trees": complete_redundant_trees,
                            "summarize": summarize,
                            "rsdt": rsdt,
                            "resilience": resilience,
                            "rsdts_path": rsdts_path
                        }

                        if timing:
                            time_start = time.time()
                        # call evaluation function
                        if DT_RF == "DT":
                            bfi_tree(expdata_dict)
                        if DT_RF == "RF":
                            bfi_forest(expdata_dict)
                        if timing:
                            time_end = time.time()
                            print("Evaluation time: ", time_end - time_start)

                        # dump experiment settings to file, but first remove unserializable elements
                        keys_to_remove = ["model", "X_train", "X_test", "y_train", "y_test"]
                        for key in keys_to_remove:
                            expdata_dict.pop(key)
                        to_dump_data = expdata_dict
                        to_dump_path = exp_path + "/results.txt"
                        store_exp_data_write(to_dump_path, to_dump_data)

                    if plot_histogram is not None:
                        # print("Xtrain", X_train)

                        # plot histogram of values
                        mu, std = norm.fit(X_train.flatten())
                        s = np.random.normal(mu, std, 100)
                        plt.hist(X_train.flatten(), bins=50, color='g')
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 1000)
                        # p = norm.pdf(x, mu, std)
                        # plt.plot(x, p, 'k', linewidth=2)
                        plt.semilogy()
                        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
                        plt.title(title)
                        plt.savefig("input_distr.pdf", format="pdf")

                        min = np.abs(X_train).min()
                        max = np.abs(X_train).max()
                        # print(f"min: {min}, max: {max}")

                    # visualize model (for tree)
                    # fig = plt.figure(figsize=(25,20))
                    # _ = tree.plot_tree(clf,
                    #                    feature_names=iris.feature_names,
                    #                    class_names=iris.target_names,
                    #                    filled=True)
                    # fig.savefig("DT{}_{}.png".format(depth, dataset))


if __name__ == '__main__':
    main()
