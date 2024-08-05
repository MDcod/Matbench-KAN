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
    
    
    
    
    
mb = MatbenchBenchmark(autoload=False, subset=['matbench_steels'])

for task in mb.tasks:
    task.load()
    for fold in task.folds:

      dataset = getDataset(task, fold)
      
      model = KAN(width=[dataset['train_input'].shape[1], 9,  1], grid=5, k=3, seed=0)
      model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=2.7876383801674827)
      model = model.prune(node_th=2e-1)
      model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=2.7876383801674827)
      model = model.refine(10)
      model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=2.7876383801674827)
      # model.auto_symbolic(lib=['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs'])
      # model.fit(dataset, opt="LBFGS", steps=20)
      
      print('start predicting')
      prediction = model(dataset['test_input']).detach().numpy()

      print('start recording')
      task.record(fold, prediction)  
      print('stop recording')

    print(task.is_recorded)
    print(json.dumps(task.scores, indent=4))
    print(task.scores.mae.mean)
    
mb.to_file(f'my_models_benchmark-{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json')