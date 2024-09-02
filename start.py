from datetime import datetime
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

from sklearn.model_selection import train_test_split

from matminer.featurizers.conversions import (
    StrToComposition,
    StructureToComposition,
)
from matminer.featurizers.composition.element import ElementFraction
from pymatgen.core.structure import Structure

import torch

class MYKAN:

    def train_and_validate(self, train_inputs, train_outputs, test_inputs, test_outputs):

      #  X = self.transformInput(train_inputs)
      #  y = list(train_outputs);

      #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       X_train = self.transformInput(train_inputs)
       X_test =  self.transformInput(test_inputs)
       y_train = train_outputs
       y_test = test_outputs

       self.kan = KAN(width=[X_train.shape[1], [65, 26],  1], grid=5, k=3, seed=0)

       print(len(X_train[0]))
       print(len(X_test[0]))

       dataset = {
          'train_input': torch.tensor(X_train, dtype=torch.float32),
          'train_label': torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
          'test_input': torch.tensor(X_test, dtype=torch.float32),
          'test_label': torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
       }

       self.kan(dataset['train_input'])
      #  self.kan.plot()

       self.kan.fit(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=2.7876383801674827)
      #  self.kan.plot()

    def predict(self, test_inputs):
      tensorInput = torch.tensor(self.transformInput(test_inputs), dtype=torch.float32)
      self.predictInput = tensorInput
      pred = self.kan(tensorInput).detach().numpy()
      return pred

    def transformInput(self, inputs):
      if (type(inputs[0]) == str):
        stc = StrToComposition(overwrite_data=True)
        df = stc.featurize_dataframe(inputs.to_frame(), "composition")

        featurizer = ElementFraction()
        df = featurizer.featurize_dataframe(df, col_id='composition')
        df = df.loc[:, (df != 0).any(axis=0)]
        self.df = df

        return df[df.columns[1:]].values;

      if (type(inputs[0]) == Structure):
        stc = StructureToComposition(overwrite_data=True)
        df = stc.featurize_dataframe(train_inputs.to_frame(), "structure")

        featurizer = ElementFraction()
        df = featurizer.featurize_dataframe(df, col_id='composition')

        return df[df.columns[1:]].values;

mb = MatbenchBenchmark(autoload=False, subset=['matbench_steels'])
model = MYKAN()

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
    
    print(task.scores.mae.mean)
# Save your results
mb.to_file(f'benches/my_models_benchmark-{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json')
