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

import json
import pprint

from sklearn.model_selection import train_test_split

from matminer.featurizers.conversions import (
    StrToComposition,
    StructureToComposition,
)
from matminer.featurizers.composition.element import ElementFraction
from pymatgen.core.structure import Structure

import torch
import optuna

def getTask():
  mb = MatbenchBenchmark(autoload=False, subset=['matbench_steels'])
  task = list(mb.tasks)[0]
  task.load()
  
  return task

def getDataset(task, fold):
  
  train_inputs, train_outputs = task.get_train_and_val_data(fold)
  test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
  
  return toDataset(train_inputs, train_outputs, test_inputs, test_outputs)
  
def toDataset(train_inputs, train_outputs, test_inputs, test_outputs):
   
    X_train = transformInput(train_inputs)
    X_test =  transformInput(test_inputs)
    y_train = train_outputs
    y_test = test_outputs

    dataset = {
      'train_input': torch.tensor(X_train, dtype=torch.float32),
      'train_label': torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
      'test_input': torch.tensor(X_test, dtype=torch.float32),
      'test_label': torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    }
    
    return dataset

def transformInput(inputs):
    if (type(inputs[0]) == str):
      stc = StrToComposition(overwrite_data=True)
      df = stc.featurize_dataframe(inputs.to_frame(), "composition")

      featurizer = ElementFraction()
      df = featurizer.featurize_dataframe(df, col_id='composition')
      df = df.loc[:, (df != 0).any(axis=0)]

      return df[df.columns[1:]].values

    if (type(inputs[0]) == Structure):
      stc = StructureToComposition(overwrite_data=True)
      df = stc.featurize_dataframe(inputs.to_frame(), "structure")

      featurizer = ElementFraction()
      df = featurizer.featurize_dataframe(df, col_id='composition')

      return df[df.columns[1:]].values
    
    
    
    
def objective(trial):
  mb = MatbenchBenchmark(autoload=False, subset=['matbench_steels'])

  optimizer = trial.suggest_categorical("optimizer", ["LBFGS"])
  lamb_entropy = trial.suggest_float('lamb_entropy', 2, 10, step=0.1)
  lamb = trial.suggest_categorical("lamb", [0.001])
  n_layers = trial.suggest_int('n_layers', 1, 3, log=True)
  hidden = []
  for i in range(n_layers):
    num_hidden = trial.suggest_int(f'n_units_l{i}', 8, 128, log=True)
    hidden.append(num_hidden)

  if n_layers == 1:
      hidden = [hidden[0], 0] # если подаешь в кан число а не массив он внутри превращается в массив с вторым элементом 0. ну как же так ну как же так

  try:
    for task in mb.tasks:
        task.load()
        for fold in task.folds:

          dataset = getDataset(task, fold)
          model = KAN(width=[dataset['train_input'].shape[1], hidden, 1], grid=5, k=3, seed=0)
          model.fit(dataset, opt=optimizer, steps=20, lamb=lamb, lamb_entropy=lamb_entropy)
          # model = model.prune(node_th=2e-1)
          # model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=lamb_entropy)
          # model = model.refine(10)
          # model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=lamb_entropy)
          # model.auto_symbolic(lib=['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs'])
          # model.fit(dataset, opt="LBFGS", steps=20)
          
          print('start predicting')
          prediction = model(dataset['test_input']).detach().numpy()
          
          print('start recording')
          task.record(fold, prediction)  
          print('stop recording')
          
          mae = task.results[f'fold_{str(fold)}'].scores.mae
          print(f'fold_{fold} mae: {mae}')

        print(task.scores.mae.mean)
        print(json.dumps(trial.params, indent=4))
        trial.report(task.scores.mae.mean, 0)
        
        return task.scores.mae.mean
  except Exception as e:
    print(e)
    return 999999


study = optuna.create_study()
try:
  study.optimize(lambda trial: objective(trial), n_trials=50, show_progress_bar=True)
except Exception as e:
  print(e)
  print(json.dumps(study.best_trial, indent=4))
  print(json.dumps(study.best_params, indent=4))
  print(json.dumps(study.best_value, indent=4))
    
    
    
    
    
    
# mb = MatbenchBenchmark(autoload=False, subset=['matbench_steels'])
# model = None

# for task in mb.tasks:
#     task.load()
#     for fold in task.folds:

#       dataset = getDataset(task, fold)
#       if model is None:
#         model = KAN(width=[dataset['train_input'].shape[1], dataset['train_input'].shape[1] * 2 - 1,  1], grid=5, k=3, seed=0)
#       model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
#       prediction = model(dataset['test_input']).detach().numpy()

#       task.record(fold, prediction)  

#     print(task.is_recorded)
#     print(json.dumps(task.scores, indent=4))
#     print(task.scores.mae.mean)