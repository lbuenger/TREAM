import os

import joblib
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, \
    average_precision_score


def main():

    src_path = "./results/raw/nt/split/"
    des_path = "./results/clean/nt/memory.txt"

    #src_path = "./results/complete_trees/c_split/"
    #des_path = "./results/clean/ct/memory.txt"

    #src_path = "./results/raw/crt/split/"
    #des_path = "./results/clean/crt/memory_test.txt"

    out = 0

    nodes = 0

    counter = 0
    file_counter = 0

    for filename in os.listdir(src_path):
        if (filename.find("pkl") != -1):
            # print(filename)
            model = joblib.load(src_path + filename)
            if (filename.find("_T") != -1):
                nodes = 0
                for tree in model.estimators_:
                    # print(tree.tree_.node_count)
                    nodes += tree.tree_.node_count
                out += nodes
                counter += 1
                print(filename, " ", nodes)
            else:
                #print(model.tree_.node_count)
                out += model.tree_.node_count
                counter += 1
                print(filename, " ", model.tree_.node_count)

    out = round(out / counter, 4)

    # print(out)

    with open(des_path, 'w+') as w:
        w.write("crt,{:.4f}".format(out))

    # print(out)


# print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))


if __name__ == '__main__':
    main()