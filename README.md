# TREAM
A framework built on top of sklearn for error evaluations of tree-based models

Instructions:
- Uninstall the sklearn library installed on your device.
- Pull the [fork of sklearn](https://github.com/myay/BETRF) with error tolerance analysis extension.
- Check out the branch called "bet".
- For first time installation of the sklearn with the error evaluation extension, run the following command in the root folder of sklearn: `python3 -m pip install --editable .`.
- After the first time install, when the source of sklearn is modified, it can be compiled using: `python3 -m pip install --verbose --no-build-isolation --editable .`.
- To test TREAM, run the following: `python3 run_exp.py --model=RF --dataset=MNIST --depth=4 --estims=3 --store-model=1 --trials=3 --splitval-inj=1 --nr-bits-split=8 --nr-bits-feature=8 --int-split=1 --export-accuracy=1`.
- To check the experiment results, go to folder the `experiments` then go to the folder with the time stamp of the experiments, and view the `results.txt`. The model will also be stored here as a `.pkl`, which can be loaded again into the framework.

A list of the command line parameters for running evaluations with TREAM:
| Command line parameter | Options |
| ------------- |:-------------:|
| --model      | DT, RF |
| --dataset      | MNIST, IRIS, ADULT, SENSORLESS, WINEQUALITY, OLIVETTI, COVTYPE, SPAMBASE, WEARABLE, LETTER |
| --depths | Integer, maxmimum depth of the decision tree(s) |
| --estims | Integer, number of DTs in RF |
| --store-model | 0/1, whether to dump the model file |
| --trials | Integer, number of repetitions of the error evaluations for the same bit error rate |
| --splitval-inj | 0/1, whether to inject bit errors into the split values |
| --featval-inj | 0/1, whether to inject bit errors into the feature values |
| --featidx-inj | 0/1, whether to inject bit errors into the feature indices |
| --chidx-inj | 0/1, whether to inject bit errors into the child indices |
| --int-split | 0/1, whether to use integer representation for slits |
| --true-majority | 0/1, whether to use the true majority, instead of the standard weighted majority |
| --load-model | String, loads a model from the speficied path in the string |
| --seed | Integer, seed for the reproducability of experiments |

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
