from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import sys
from tqdm import tqdm


class bqaDataset(Dataset):
    def __init__(self, file):
        self.cases = []  # self.cases with judgements, type=list(str)
        self.questions = []  # questions aligned with cases, type=list(str)

        'process data from file'
        with open(file, 'r', encoding='utf-8') as f:
            print('reading from ' + file)
            for line in tqdm(f.readlines()):
                line = line[1:-2]
                case = []
                question = []
                for x in line.split(', '):
                    y = json.loads(x)
                    if 'question' in list(y.keys())[0]:
                        question.append(list(y.values())[0])
                    content = list(y.values())[0]
                    case.append(content)
                # move background to the first
                background = case.pop().strip()
                case.insert(0, background)
                case = ''.join(case)
                # one-case n-question = n samples with the same case
                for i in range(len(question)):
                    self.cases.append(case)
                self.questions.extend(question)

    def __getitem__(self, index):
        return {'case': self.cases[index],
                'question': self.questions[index]}

    def __len__(self):
        return len(self.cases)


class bqakDataset(Dataset):
    pass
