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
import matplotlib.pyplot as plt
import sklearn
import argparse

from sklearn.model_selection import train_test_split
from loadData import readFileMNIST, readFileAdult, readFileSensorless, readFileWinequality, readFileWine, readFileTicTacToe, readFileOccupancy, readFileSpamBase, readFileWearable, readFileLetter
from sklearn.datasets import load_iris, fetch_olivetti_faces, fetch_covtype
from Utils import quantize_data
import numpy as np

model = joblib.load("D5_IRIS.pkl")
plt.figure(figsize=(50,50))
sklearn.tree.plot_tree(model)
plt.savefig("plot.png")