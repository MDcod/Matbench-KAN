from matminer.featurizers.conversions import (
    StrToComposition,
    StructureToComposition,
)
from matminer.featurizers.composition.element import ElementFraction
from pymatgen.core.structure import Structure
import torch
from kan import *


# noinspection PyPep8Naming
class KanWrapper:
    def __init__(self):
        self.df = None
        self.predictInput = None
        self.kan = None

    def train_and_validate(self, train_inputs, train_outputs, test_inputs, test_outputs):

        X_train = self.transformInput(train_inputs)
        X_test = self.transformInput(train_outputs)
        y_train = self.transformInput(test_inputs)
        y_test = self.transformInput(test_outputs)
        self.kan = KAN(width=[X_train.shape[1], X_train.shape[1] * 2 - 1, 1], grid=3, k=2, seed=0)

        print(len(X_train[0]))
        print(X_train.shape[1])

        train_input = torch.tensor(X_train, dtype=torch.float32)
        train_label = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        test_input = torch.tensor(X_test, dtype=torch.float32)
        test_label = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        dataset = {
            'train_input': train_input,
            'train_label': train_label,
            'test_input': test_input,
            'test_label': test_label
        }

        self.kan.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)

    def predict(self, test_inputs):
        tensorInput = torch.tensor(self.transformInput(test_inputs), dtype=torch.float32)
        self.predictInput = tensorInput
        pred = self.kan(tensorInput).detach().numpy()
        return pred

    def transformInput(self, inputs):
        if type(inputs[0]) == str:
            stc = StrToComposition(overwrite_data=True)
            df = stc.featurize_dataframe(inputs.to_frame(), "composition")

            featurizer = ElementFraction()
            df = featurizer.featurize_dataframe(df, col_id='composition')

            self.df = df

            return df[df.columns[1:]].values

        if type(inputs[0]) == Structure:
            stc = StructureToComposition(overwrite_data=True)
            df = stc.featurize_dataframe(inputs.to_frame(), "structure")

            featurizer = ElementFraction()
            df = featurizer.featurize_dataframe(df, col_id='composition')

            return df[df.columns[1:]].values
