from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import sys
from tqdm import tqdm


class bqaDataset(Dataset):
    def __init__(self, cases, questions):
        self.cases = cases  # self.cases with judgements
        self.questions = questions  # questions aligned with cases


    def __getitem__(self, index):
        return {'case': self.cases[index],
                'question': self.questions[index]}

    def __len__(self):
        return len(self.cases)


class bqakDataset(Dataset):
    pass
