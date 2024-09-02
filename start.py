import torch
import json
import torch
import numpy as np

from datetime import datetime
from matbench.bench import MatbenchBenchmark
from kan import *

from matminer.featurizers.conversions import (StrToComposition, StructureToComposition)
from matminer.featurizers.composition.element import ElementFraction
from pymatgen.core.structure import Structure

class MYKAN:

  def train_and_validate(self, train_inputs, train_outputs, test_inputs, test_outputs):

      y_train = train_outputs
      y_test = test_outputs
      X_train_tuple = self.transformInput(train_inputs)
      X_test_tuple =  self.transformInput(test_inputs)
      
      if (len(X_train_tuple[0][0]) == len(X_test_tuple[0][0])):
        print(f'Одинаковые размерности ({len(X_train_tuple[0][0])} == {len(X_test_tuple[0][0])})! Берём сокращенный data frame')
        X_train = X_train_tuple[0]
        X_test =  X_test_tuple[0]
      else:
        print(f'Разные размерности ({len(X_train_tuple[0][0])} != {len(X_test_tuple[0][0])})! Берём полный data frame')
        X_train = X_train_tuple[1]
        X_test =  X_test_tuple[1]

      self.kan = KAN(width=[X_train.shape[1], 20, 1], grid=5, k=3, seed=0)

      dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'train_label': torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        'test_input': torch.tensor(X_test, dtype=torch.float32),
        'test_label': torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
      }
      
      self.tensorTestInputs = dataset['test_input']

      self.kan(dataset['train_input'])
      self.kan.fit(dataset, steps=20, lamb=0.01, lamb_entropy=2.7876383801674827)

  def predict(self):
    return self.kan(self.tensorTestInputs).detach().numpy()

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
      collapsedDf = df.loc[:, (df != 0).any(axis=0)]

      fullValues = df[df.columns[2:]].values
      collapsedValues = collapsedDf[collapsedDf.columns[2:]].values
      
      return (collapsedValues, fullValues);


###########################################################################################
##################################### Entry point #########################################
###########################################################################################


os.makedirs('benches', exist_ok = True)
mb = MatbenchBenchmark(autoload=False)
model = MYKAN()

for task in mb.tasks:
    task.load()
    for fold in task.folds:

      train_inputs, train_outputs = task.get_train_and_val_data(fold)
      test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

      model.train_and_validate(train_inputs, train_outputs, test_inputs, test_outputs)

      predictions = model.predict()

      if task.metadata["task_type"] == "classification":
        predictions = np.where(np.abs(predictions) < 1, np.round(np.abs(predictions)), 1).astype(int)

      # Record your data!
      task.record(fold, predictions)
    
    print(json.dumps(task.scores, indent=4))
    with open(f'benches/{task.dataset_name}-result.txt', 'w', encoding='utf-8') as file:
      file.write(json.dumps(task.scores, indent=4))
      
# Save your results
mb.to_file(f'benches/KAN_benchmark-{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json')
