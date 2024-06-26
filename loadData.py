# Functions for loading the datasets are based on code from
# https://github.com/tudo-ls8/arch-forest
from datetime import datetime

import numpy as np

import sys
#np.set_printoptions(threshold=sys.maxsize)

def readFileMNIST(path):
    X = []
    Y = []

    f = open(path,'r')

    for row in f:
        entries = row.strip("\n").split(",")

        Y.append(int(entries[0])-1)
        x = [int(e) for e in entries[1:]]
        X.append(x)

    Y = np.array(Y)-min(Y)
    return np.array(X).astype(dtype=np.int32), Y

def getFeatureVectorAdult(entries):
    x = []
    x.append(float(entries[0])) # age = continous

    workclass = [0 for i in range(8)]
    if entries[1] == "Private":
        workclass[0] = 1
    elif entries[1] == "Self-emp-not-inc":
        workclass[1] = 1
    elif entries[1] == "Self-emp-inc":
        workclass[2] = 1
    elif entries[1] == "Federal-gov":
        workclass[3] = 1
    elif entries[1] == "Local-gov":
        workclass[4] = 1
    elif entries[1] == "State-gov":
        workclass[5] = 1
    elif entries[1] == "Without-pay":
        workclass[6] = 1
    else: #Never-worked
        workclass[7] = 1
    x.extend(workclass)
    x.append(float(entries[2]))

    education = [0 for i in range(16)]
    if entries[3] == "Bachelors":
        education[0] = 1
    elif entries[3] == "Some-college":
        education[1] = 1
    elif entries[3] == "11th":
        education[2] = 1
    elif entries[3] == "HS-grad":
        education[3] = 1
    elif entries[3] == "Prof-school":
        education[4] = 1
    elif entries[3] == "Assoc-acdm":
        education[5] = 1
    elif entries[3] == "Assoc-voc":
        education[6] = 1
    elif entries[3] == "9th":
        education[7] = 1
    elif entries[3] == "7th-8th":
        education[8] = 1
    elif entries[3] == "12th":
        education[9] = 1
    elif entries[3] == "Masters":
        education[10] = 1
    elif entries[3] == "1st-4th":
        education[11] = 1
    elif entries[3] == "10th":
        education[12] = 1
    elif entries[3] == "Doctorate":
        education[13] = 1
    elif entries[3] == "5th-6th":
        education[14] = 1
    if entries[3] == "Preschool":
        education[15] = 1

    x.extend(education)
    x.append(float(entries[4]))

    marital = [0 for i in range(7)]
    if entries[5] == "Married-civ-spouse":
        marital[0] = 1
    elif entries[5] == "Divorced":
        marital[1] = 1
    elif entries[5] == "Never-married":
        marital[2] = 1
    elif entries[5] == "Separated":
        marital[3] = 1
    elif entries[5] == "Widowed":
        marital[4] = 1
    elif entries[5] == "Married-spouse-absent":
        marital[5] = 1
    else:
        marital[6] = 1
    x.extend(marital)

    occupation = [0 for i in range(14)]
    if entries[6] == "Tech-support":
        occupation[0] = 1
    elif entries[6] == "Craft-repair":
        occupation[1] = 1
    elif entries[6] == "Other-service":
        occupation[2] = 1
    elif entries[6] == "Sales":
        occupation[3] = 1
    elif entries[6] == "Exec-managerial":
        occupation[4] = 1
    elif entries[6] == "Prof-specialty":
        occupation[5] = 1
    elif entries[6] == "Handlers-cleaners":
        occupation[6] = 1
    elif entries[6] == "Machine-op-inspct":
        occupation[7] = 1
    elif entries[6] == "Adm-clerical":
        occupation[8] = 1
    elif entries[6] == "Farming-fishing":
        occupation[9] = 1
    elif entries[6] == "Transport-moving":
        occupation[10] = 1
    elif entries[6] == "Priv-house-serv":
        occupation[11] = 1
    elif entries[6] == "Protective-serv":
        occupation[12] = 1
    else:
        occupation[13] = 1
    x.extend(occupation)

    relationship = [0 for i in range(6)]
    if entries[7] == "Wife":
        relationship[0] = 1
    elif entries[7] == "Own-child":
        relationship[1] = 1
    elif entries[7] == "Husband":
        relationship[2] = 1
    elif entries[7] == "Not-in-family":
        relationship[3] = 1
    elif entries[7] == "Other-relative":
        relationship[4] = 1
    else:
        relationship[5] = 1
    x.extend(relationship)

    race = [0 for i in range(5)]
    if entries[8] == "White":
        race[0] = 1
    elif entries[8] == "Asian-Pac-Islander":
        race[1] = 1
    elif entries[8] == "Amer-Indian-Eskimo":
        race[2] = 1
    elif entries[8] == "Other":
        race[3] = 1
    else:
        race[4] = 1
    x.extend(race)

    if (entries[9] == "Male"):
        x.extend([1,0])
    else:
        x.extend([0,1])

    x.append(float(entries[10]))
    x.append(float(entries[11]))
    x.append(float(entries[12]))

    native = [0 for i in range(42)]
    if entries[14] == "United-States":
        native[1] = 1
    elif entries[14] == "Cambodia":
        native[2] = 1
    elif entries[14] == "England":
        native[3] = 1
    elif entries[14] == "Puerto-Rico":
        native[4] = 1
    elif entries[14] == "Canada":
        native[5] = 1
    elif entries[14] == "Germany":
        native[6] = 1
    elif entries[14] == "Outlying-US(Guam-USVI-etc)":
        native[7] = 1
    elif entries[14] == "India":
        native[8] = 1
    elif entries[14] == "Japan":
        native[9] = 1
    elif entries[14] == "Greece":
        native[10] = 1
    elif entries[14] == "South":
        native[11] = 1
    elif entries[14] == "China":
        native[12] = 1
    elif entries[14] == "Cuba":
        native[13] = 1
    elif entries[14] == "Iran":
        native[14] = 1
    elif entries[14] == "Honduras":
        native[15] = 1
    elif entries[14] == "Philippines":
        native[16] = 1
    elif entries[14] == "Italy":
        native[17] = 1
    elif entries[14] == "Poland":
        native[18] = 1
    elif entries[14] == "Jamaica":
        native[19] = 1
    elif entries[14] == "Vietnam":
        native[20] = 1
    elif entries[14] == "Mexico":
        native[21] = 1
    elif entries[14] == "Portugal":
        native[22] = 1
    elif entries[14] == "Ireland":
        native[23] = 1
    elif entries[14] == "France":
        native[24] = 1
    elif entries[14] == "Dominican-Republic":
        native[25] = 1
    elif entries[14] == "Laos":
        native[26] = 1
    elif entries[14] == "Ecuador":
        native[27] = 1
    elif entries[14] == "Taiwan":
        native[28] = 1
    elif entries[14] == "Haiti":
        native[29] = 1
    elif entries[14] == "Columbia":
        native[30] = 1
    elif entries[14] == "Hungary":
        native[31] = 1
    elif entries[14] == "Guatemala":
        native[32] = 1
    elif entries[14] == "Nicaragua":
        native[33] = 1
    elif entries[14] == "Scotland":
        native[34] = 1
    elif entries[14] == "Thailand":
        native[35] = 1
    elif entries[14] == "Yugoslavia":
        native[36] = 1
    elif entries[14] == "El-Salvador":
        native[37] = 1
    elif entries[14] == "Trinadad&Tobago":
        native[38] = 1
    elif entries[14] == "Peru":
        native[39] = 1
    elif entries[14] == "Hong":
        native[40] = 1
    else:
        native[41] = 1

    return x

def readFileAdult(path):
    X = []
    Y = []
    f = open(path, 'r')
    for row in f:
        if (len(row) > 1):
            entries = row.replace("\n", "").replace(" ", "").split(",")

            x = getFeatureVectorAdult(entries)

            if (entries[-1] == "<=50K"):
                y = 0
            else:
                y = 1

            Y.append(y)
            X.append(x)

    X = np.array(X).astype(dtype=np.int32)
    Y = np.array(Y)
    return X, Y

def readFileSensorless(dataset_path):
    data = np.genfromtxt(dataset_path, delimiter=' ')
    idx = np.random.permutation(len(data))

    X = data[idx,0:-1].astype(dtype=np.float32)
    Y = data[idx,-1]
    Y = Y-min(Y)
    return X, Y

def readFileWinequality(dataset_path):
    dataset_path_red = dataset_path + "winequality-red.csv"
    dataset_path_white = dataset_path + "winequality-white.csv"
    red = np.genfromtxt(dataset_path_red, delimiter=';', skip_header=1)
    white = np.genfromtxt(dataset_path_white, delimiter=';', skip_header=1)
    X = np.vstack((red[:,:-1],white[:,:-1])).astype(dtype=np.float32)
    Y = np.concatenate((red[:,-1], white[:,-1]))
    Y = Y-min(Y)
    return X, Y

def readFileWine(dataset_path):
    dataset_path = dataset_path + "wine.data"
    data = np.genfromtxt(dataset_path, delimiter=',')
    X = data[:, 1:]
    Y = data[:,0]
    Y = Y-min(Y)
    # print(Y)
    return X, Y

def readFileTicTacToe(dataset_path):
    dataset_path = dataset_path + "tic-tac-toe.data"
    numbers = ""

    X = []
    Y = []

    with open(dataset_path) as f:
        data = f.read()
        #print(data)
    for x in data.replace("\n", ",").split(","):
        #print(x)
        if x == "b":
            X.append(0)
            #print("b")
        elif x == "x":
            X.append(1)
            #print("x")
        elif x == "o":
            X.append(2)
            #print("o")
        elif x == "positive":
            #print("positive\n")
            Y.append(1)
        elif x == "negative":
            #print("negative\n")
            Y.append(0)


    #print(numbers)
    X = np.reshape(X, (int(len(X)/9), 9))
    #print(X)
    #print(Y)
    return np.array(X), np.array(Y)

def readFileOccupancy(dataset_path):
    dataset_path1 = dataset_path + "datatest.txt"
    dataset_path2 = dataset_path + "datatest2.txt"
    dataset_path3 = dataset_path + "datatraining.txt"
    X = []
    Y = []

    with open(dataset_path1) as f:
        data = f.read()
    with open(dataset_path2) as f:
        data += f.read()
    with open(dataset_path3) as f:
        data += f.read()

    for line in data[:-1].split("\n"):
        x = []
        values = line.split(",")

        if(values[0] == "\"date\""):
            continue

        x.append(datetime.strptime(values[1].replace("\"", ""), '%Y-%m-%d %H:%M:%S').strftime("%Y%m%d%H%M%S"))
        x.append(int(float(values[2])*100))
        x.append(int(float(values[3])*1000))
        x.append(int(float(values[4])*10))
        x.append(int(float(values[5])*100))
        Y.append(int(values[7]))

        X.append(x)

    # print(X)
    # print(Y)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def readFileSpamBase(path):
    f = open(path, 'r')
    X = []
    Y = []
    for row in f:
        entries = row.strip().split(",")
        x = [int(float(e)*100) for e in entries[0:-1]]

        y = int(entries[-1])
        X.append(x)
        Y.append(y)

    return np.array(X).astype(dtype=np.int32), np.array(Y)


def readFileWearable(path):
    f = open(path, 'r')
    header = next(f)
    X = []
    Y = []

    for row in f:
        entries = row.replace("\n","").split(";")

        if entries[-1] == 'sitting':
            y = 0
        elif entries[-1] == 'standing':
            y = 1
        elif entries[-1] == 'standingup':
            y = 2
        elif entries[-1] == 'walking':
            y = 3
        elif entries[-1] == 'sittingdown':
            y = 4
        else:
            print("ERROR READING CLASSES:", entries[-1])

        x = []
        if entries[1] == 'Man':
            x.append(0)
        else:
            x.append(1)

        for e in entries[2:-1]:
            x.append(int(float(e.replace(",","."))*100))
        X.append(x)
        Y.append(y)

    return np.array(X).astype(dtype=np.int32), np.array(Y)


def readFileLetter(path):
    f = open(path, 'r')
    X = []
    Y = []
    for row in f:
        entries = row.strip().split(",")
        x = [int(e) for e in entries[1:]]
        y = ord(entries[0]) - 65
    
        X.append(x)
        Y.append(y)

    return np.array(X).astype(dtype=np.int32), np.array(Y)

def readFileWeatherAus(path):
    f = open(path, 'r')
    X = []
    Y = []
    for row in f:
            entries = row.strip().split(",")

