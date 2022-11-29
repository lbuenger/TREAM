# TREAM
A framework built on top of sklearn for error evaluations of tree-based models

Instructions:
- Uninstall the sklearn library installed on your device (or use a new virtual environment).
- Pull the [fork of sklearn](https://github.com/myay/BETRF) with error tolerance analysis extension.
- Check out the branch called "bet".
- For first time installation of the sklearn with the error evaluation extension, run the following command in the root folder of sklearn: `python3 -m pip install --editable .`.
- After the first time install, when the source of sklearn is modified, it can be compiled using: `python3 -m pip install --verbose --no-build-isolation --editable .`.
- To test TREAM, first download the MNIST dataset to the folder `data/mnist` using the script and then run the following command: `python3 run_exp.py --model=RF --dataset=MNIST --depth=4 --estims=3 --store-model=1 --trials=3 --splitval-inj=1 --nr-bits-split=8 --nr-bits-feature=8 --int-split=1 --export-accuracy=1`.
- To check the experiment results, go to folder `experiments` then go to the folder with the time stamp of the experiments, and view the `results.txt`. The model will also be stored here as a `.pkl`, which can be loaded again into the framework.

Here is a list of the command line parameters for running the error evaluations with TREAM:
| Command line parameter | Options |
| :------------- |:-------------|
| --model      | DT, RF |
| --dataset      | MNIST, IRIS, ADULT, SENSORLESS, WINEQUALITY, OLIVETTI, COVTYPE, SPAMBASE, WEARABLE, LETTER |
| --depths | Integer, maxmimum depth of the decision tree(s) |
| --estims | Integer, number of DTs in RF |
| --trials | Integer, number of repetitions of the error evaluations for the same bit error rate, default: 5 |
| --seed | Integer, seed for the reproducability of experiments, default: 42 |
| --splitval-inj | 0/1, whether to inject bit errors into the split values, default: 0 |
| --featval-inj | 0/1, whether to inject bit errors into the feature values, default: 0 |
| --featidx-inj | 0/1, whether to inject bit errors into the feature indices, default: 0 |
| --chidx-inj | 0/1, whether to inject bit errors into the child indices, default: 0 |
| --int-split | 0/1, whether to use integer representation for splits, default: 0 |
| --true-majority | 0/1, whether to use the true majority, instead of the standard weighted majority, default: 0 |
| --store-model | 0/1, whether to dump the model file, default:0 |
| --load-model | String, loads a model from the speficied path in the string, default: 0 |

More information on the command line parameters can be found [here](https://github.com/myay/TREAM/blob/main/Utils.py#L7).

If you find TREAM useful in your work, please cite the following source:

[Mikail Yayla, Zahra Valipour Dehnoo, Mojtaba Masoudinejad, Jian-Jia Chen "TREAM: A Tool for Evaluating Error Resilience of Tree-Based Models Using Approximate Memory". Embedded Computer Systems: Architectures, Modeling, and Simulation (SAMOS), 2022.](https://link.springer.com/chapter/10.1007/978-3-031-15074-6_4)

```
@InProceedings{yayla-samos/etal/2022,
  author="Yayla, Mikail
  and Valipour Dehnoo, Zahra
  and Masoudinejad, Mojtaba
  and Chen, Jian-Jia",
  title="TREAM: A Tool for Evaluating Error Resilience of Tree-Based Models Using Approximate Memory",
  booktitle="Embedded Computer Systems: Architectures, Modeling, and Simulation (SAMOS)",
  year="2022",
}
```

Please contact me if you have any questions: mikail.yayla@tu-dortmund.de.
