# Residual-PAC-Privacy

This directory contains stand-alone Python scripts for the experiments in the paper.


Files
----------------------------

1. ag_news_s.py    – SR-PAC vs PAC on AG-News text classification.
2. CIFAR-10_s.py   – SR-PAC vs PAC on CIFAR-10 logits.
3. CIFAR-100_s.py  – SR-PAC vs PAC on CIFAR-100 logits.
4. Iris_s.py       – SR-PAC vs PAC on the Iris tabular dataset.
5. MNIST_s.py      – SR-PAC vs PAC on MNIST logits.
6. Rice_s.py       – SR-PAC vs PAC vs DP on the Rice Cammeo/Osmancik aggregate-query setting, plus LiRA auditing.
7. lira_eval.py    – Shared LiRA utilities (posterior success rate, theoretical PSR, comparison table). Used only by Rice_s.py.



Environment
----------------------------

The scripts assume a standard scientific Python environment with PyTorch
(e.g., Python 3.x with numpy, torch / torchvision, nflows, scikit-learn).
Install these with your usual package manager (pip or conda).


Datasets and required files
----------------------------

Most datasets are automatically downloaded or loaded by standard libraries:

- MNIST, CIFAR-10, CIFAR-100 – via torchvision.datasets.
- AG-News – via the loader inside ag_news_s.py.
- Iris – via a standard one-shot loader (e.g., sklearn.datasets.load_iris).

Rice dataset:

- Rice_s.py expects Rice_Cammeo_Osmancik.arff to be present in the same directory.
- If this file is missing, Rice_s.py raises a FileNotFoundError with the expected filename.


Pre-trained models:


The image/logit scripts try to load small pre-trained classifiers and will
“quick-train” them if not found:

- MNIST_s.py looks for mnist_cnn_non_gaussian.pth and trains/saves it if missing.
- The CIFAR scripts follow the same pattern with their own checkpoints.

No manual preparation of these checkpoints is required for basic reproduction.


How to run
----------------------------

From this directory, each experiment can be run directly, for example:

python MNIST_s.py
python CIFAR-10_s.py
python CIFAR-100_s.py
python ag_news_s.py
python Iris_s.py
python Rice_s.py

Each script

- prints progress and a summary table to standard output, and
- may write a small CSV file with summary statistics (e.g., noise power and accuracy vs β),
  with the output filename defined near the top of the script (for example, OUT_CSV in MNIST_s.py).


File dependencies summary
----------------------------

1. ag_news_s.py      – self-contained
2. CIFAR-10_s.py     – self-contained
3. CIFAR-100_s.py    – self-contained
4. Iris_s.py         – self-contained
5. MNIST_s.py        – self-contained
6. Rice_s.py         – depends on lira_eval.py
7. lira_eval.py      – helper module only


Minimal set of files to reproduce the experiments
----------------------------------------------------

ag_news_s.py
CIFAR-10_s.py
CIFAR-100_s.py
Iris_s.py
MNIST_s.py
Rice_s.py
lira_eval.py
Rice_Cammeo_Osmancik.arff   (Rice dataset file required by Rice_s.py)
