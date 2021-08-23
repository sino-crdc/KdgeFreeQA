# coding: utf-8
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import sys
reload(sys)
from tqdm import tqdm
from preprocess.text_utils import init_tokenizer, word_seg

sys.setdefaultencoding('utf-8')
# what is 'all data from here' and 'one data from here'
# - 'all data from here': we load all of the data when construct cqaDataset object, that is to say, we store all the data all the time.
# - 'one data from here': we just load one data (of index 'index of __getitem__') from file, we don't need to store all the data but the file path.
# between two annotations, that is the code.

class cqaDataset(Dataset):
    def __init__(self, train_cfg):
        self.train_cfg = train_cfg

        'all data from here'
        cases = []
        questions = []
        with open(self.train_cfg['cqa_train_path'], 'r', encoding='utf-8') as f:
            print('read data from ' + self.train_cfg['cqa_train_path'])
            # count = 0  # the following loop is too long, for quick debug, we only use some samples for trial.
            for line in tqdm(f.readlines()):
                # count += 1
                # if count == 100:  # todo : we load 100 cases for debugging.
                #     break
                line = line[1:-2]
                case = []
                question = []
                for x in line.split(', {'):
                    try:
                        if x[0] != '{':
                            x = '{' + x
                        y = json.loads(x)
                    except Exception as e:
                        print(line)
                        print(x)
                        sys.exit(0)
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
                    cases.append(case)
                questions.extend(question)
        self.cases = cases
        self.questions = questions
        'all data to here'

    def __getitem__(self, index):
        train_cfg = self.train_cfg
        cqa_train_path = train_cfg['cqa_train_path']

        'one data from here'
        # 'process text from file'
        # print('get item')
        # with open(cqa_train_path, 'r', encoding='utf-8') as f:
        #     for i in range(index-1):
        #         f.readline()
        #     line = f.readline()[1:-2]
        #     case = []
        #     question = []
        #     for x in line.split(', {'):
        #         try:
        #             if x[0] != '{':
        #                 x = '{' + x
        #             y = json.loads(x)
        #         except Exception as e:
        #             print(line)
        #             print(x)
        #             sys.exit(0)
        #         if 'question' in list(y.keys())[0]:
        #             question.append(list(y.values())[0])
        #         content = list(y.values())[0]
        #         case.append(content)
        #     # move background to the first
        #     background = case.pop().strip()
        #     case.insert(0, background)
        #     case = ''.join(case)
        #     # one-case n-question = n samples with the same case
        #     question = question[0]
        'one data to here'

        'all data from here'
        case = self.cases[index]
        question = self.questions[index]
        'all data to here'

        'tokenization'
        if train_cfg['is_train_tokenized']:
            # todo : load tokenized cases and questions from files.
            pass
        else:
            case = word_seg(case)
            question = word_seg(question)

        return {'case': case, 'question': question}
            # cases = list(map(word_seg, cases))  # then cases=[['第', '一句', '话'], ..., ['第', 'N句', '话']]
            # questions = list(map(word_seg, questions))  # then cases = [['第', '一个', '问题'], ..., ['第', 'N个', '问题']]


    def __len__(self):
        'one data from here'
        # with open(self.train_cfg['cqa_train_path'], 'r', encoding='utf-8') as f:
        #     num = len(f.readlines())
        'one data to here'
        'all data from here'
        num = len(self.cases)
        'all data to here'
        return num


class bqakDataset(Dataset):
    pass
