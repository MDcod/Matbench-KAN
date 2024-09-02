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
import json

class MYKAN:

  def train_and_validate(self, train_inputs, train_outputs, test_inputs, test_outputs):

      y_train = train_outputs
      y_test = test_outputs
      X_train_tuple = self.transformInput(train_inputs)
      X_test_tuple =  self.transformInput(test_inputs)
      
      if (len(X_train_tuple[0][0]) == len(X_test_tuple[0][0])):
        print('Одинаковые размерности! Берём сокращенный data frame')
        X_train = X_train_tuple[0]
        X_test =  X_test_tuple[0]
        self.fullInputDf = False
      else:
        print('Разные размерности! Берём полный data frame')
        X_train = X_train_tuple[1]
        X_test =  X_test_tuple[1]
        self.fullInputDf = True

      self.kan = KAN(width=[X_train.shape[1], 9, 1], grid=5, k=3, seed=0)

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

      self.kan.fit(dataset, steps=20, lamb=0.001, lamb_entropy=2.7876383801674827)
    #  self.kan.plot()

  def predict(self, test_inputs):
    test_input_tuple = self.transformInput(test_inputs)
    tensorInput = torch.tensor(test_input_tuple[1] if self.fullInputDf else test_input_tuple[0], dtype=torch.float32)
    self.predictInput = tensorInput
    pred = self.kan(tensorInput).detach().numpy()
    return pred

  def transformInput(self, inputs):
    if (type(inputs[0]) == str):
      stc = StrToComposition(overwrite_data=True)
      df = stc.featurize_dataframe(inputs.to_frame(), "composition")

      featurizer = ElementFraction()
      df = featurizer.featurize_dataframe(df, col_id='composition')
      collapsedDf = df.loc[:, (df != 0).any(axis=0)]
      
      fullValues = df[df.columns[1:]].values
      collapsedValues = collapsedDf[collapsedDf.columns[1:]].values
      
      return (collapsedValues, fullValues);

    if (type(inputs[0]) == Structure):
      stc = StructureToComposition(overwrite_data=True)
      df = stc.featurize_dataframe(inputs.to_frame(), "structure")

      featurizer = ElementFraction()
      df = featurizer.featurize_dataframe(df, col_id='composition')
      collapsedDf = df.loc[:, (df != 0).any(axis=0)] # тут это не работает потому что получаются разные размерности в тестовых и тренировочных данных

      fullValues = df[df.columns[2:]].values
      collapsedValues = collapsedDf[collapsedDf.columns[2:]].values
      
      return (collapsedValues, fullValues);

set='matbench_jdft2d'
mb = MatbenchBenchmark(autoload=False)
model = MYKAN()

for task in mb.tasks:
    if task.metadata["task_type"] == "classification":
      print(f'task {task.dataset_name} is classification, skip it')
      continue
    
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
    
    print(task.is_recorded)
    print(json.dumps(task.scores, indent=4))
    print(task.scores.mae.mean)
    
    with open(f'benches/{task.dataset_name}-result.txt', 'w', encoding='utf-8') as file:
      file.write(str(task.is_recorded) + '\n')
      file.write(json.dumps(task.scores, indent=4) + '\n')
      file.write(str(task.scores.mae.mean) + '\n')
      
      
# Save your results
os.makedirs('benches', exist_ok = True)
mb.to_file(f'benches/KAN_benchmark-{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json')
