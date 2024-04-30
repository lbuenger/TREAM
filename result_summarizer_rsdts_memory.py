import os

import joblib
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, \
    average_precision_score


def main():

    src_path = "./results/raw/rsdt/fidx_half/"
    des_path = "./results/clean/rsdt/memory.txt"
    out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    nodes = 0

    counter = 0
    file_counter = 0

    for filename in os.listdir(src_path):
        if(filename.find("pkl") != -1):
            #print(filename)
            model = joblib.load(src_path + filename)
            if(filename.find("_T") != -1):
                nodes = 0
                for tree in model.estimators_:
                    #print(tree.tree_.node_count)
                    nodes += tree.tree_.node_count
                #print("forest ", nodes)
            else:
                #print(model.tree_.node_count)
                nodes = model.tree_.node_count
                #print("tree ", nodes)
            if (filename.find("rsdt0") != -1):
                out[0] += nodes
                counter += 1
            elif (filename.find("rsdt5.") != -1):
                out[1] += nodes
            elif (filename.find("rsdt10") != -1):
                out[2] += nodes
            elif (filename.find("rsdt15") != -1):
                out[3] += nodes
            elif (filename.find("rsdt20") != -1):
                out[4] += nodes
            elif (filename.find("rsdt25") != -1):
                out[5] += nodes
            elif (filename.find("rsdt30") != -1):
                out[6] += nodes
            elif (filename.find("rsdt35") != -1):
                out[7] += nodes
            elif (filename.find("rsdt40") != -1):
                out[8] += nodes
            elif (filename.find("rsdt45") != -1):
                out[9] += nodes
            elif (filename.find("rsdt50") != -1):
                out[10] += nodes
                #print("hallooooo?")
            else:
                print("hä")

    for idx, sum in enumerate(out):
        out[idx] = round(sum / counter, 4)

    #print(out)

    with open(des_path, 'w+') as w:
        w.write("rsdt0,rsdt10,rsdt15,rsdt20,rsdt25,rsdt30,rsdt35,rsdt40,rsdt45,rsdt50,\n")
        for idx, num in enumerate(out):
            w.write("{:.4f}".format(num))
            if(idx<10):
                w.write(",")

    #print(out)

#print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))










if __name__ == '__main__':
    main()