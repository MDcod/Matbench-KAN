# main.py
import pandas as pd
import torch
import numpy as np
from matbench.bench import MatbenchBenchmark
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from google.colab import files
from sklearn.metrics import make_scorer, mean_squared_error
from math import sqrt
from kan import *

from matbench.bench import MatbenchBenchmark

from KanWrapper import KanWrapper
from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False, subset=['matbench_steels'])
model = KanWrapper()

for task in mb.tasks:
    task.load()
    for fold in task.folds:

        # Inputs are either chemical compositions as strings
        # or crystal structures as pymatgen.Structure objects.
        # Outputs are either floats (regression tasks) or bools (classification tasks)
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
        # train and validate your model

        model.train_and_validate(train_inputs, train_outputs, test_inputs, test_outputs)

        # Predict on the testing data
        # Your output should be a pandas series, numpy array, or python iterable
        # where the array elements are floats or bools
        predictions = model.predict(test_inputs)

        # Record your data!
        task.record(fold, predictions)
# Save your results
mb.to_file("my_models_benchmark.json.gz")


if __name__ == "__main__":
    main()
