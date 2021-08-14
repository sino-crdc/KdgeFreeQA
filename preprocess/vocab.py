'''
@author: Yakun
'''
import jieba

train_path = r'C:\pythonProject1\KdgeFreeQA\dataset\final_all_data\first_stage\question-answer-train.json'
test_path = r'C:\pythonProject1\KdgeFreeQA\dataset\final_all_data\first_stage\question-answer-test.json'
output_train_path = r'C:/pythonProject1/KdgeFreeQA/dataset/final_all_data/first_stage/train_vocab.json'
output_test_path = r'C:/pythonProject1/KdgeFreeQA/dataset/final_all_data/first_stage/test_vocab.json'


def is_chinese(uchar):  # remove garbled codes
    if '\u4e00' <= uchar <= '\u9fff':
        return True
    else:
        return False


def generate_vocab(input_file, output_file):  # generate lists with each word
    vocabulary = {}
    with open(input_file, encoding='utf-8') as f:
        counter = 0
        for line in f:
            counter += 1
            tokens = [word for word in line.strip()]
            for word in tokens:
                if is_chinese(word):
                    if word in vocabulary:
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1
        print("vocabulary: {}".format(vocabulary))
        vocabulary_list = sorted(vocabulary, key=vocabulary.get, reverse=True)

        print("vocabulary list: {}".format(vocabulary_list))
        print(input_file + " 词汇表大小:", len(vocabulary_list))
        with open(output_file, "w", encoding='utf-8') as ff:
            for word in vocabulary_list:
                ff.write(word + "\n")


generate_vocab(train_path, output_train_path)
generate_vocab(test_path, output_test_path)


# generate vocabulary file:
# arrange the words that appear in the dataset in order of their frequency.
# you can save the filepath in config/latentqa_cail2021.py by adding another dict
# return vocab.txt and save it, word2idx(dict), idx2word(array), counter or num_vocab

def generate_char(input_file):  # generate lists with each vocabulary
    with open(input_file, encoding='utf-8') as data:
        words = set()
        for line in data:
            line = line.strip().strip('\n')
            line = jieba.cut(line)
            for w in line:
                if is_chinese(w):
                    words.add(w)
        return list(words)


if __name__ == '__main__':
    output_train_dir = r'C:/pythonProject1/KdgeFreeQA/dataset/final_all_data/first_stage/train_char.json'
    output_test_dir = r'C:/pythonProject1/KdgeFreeQA/dataset/final_all_data/first_stage/test_char.json'
    words = generate_char(input_file=train_path)
    with open(output_train_dir, 'w', encoding='utf-8') as f1:
        f1.write('\n'.join(words))
    words = generate_char(input_file=test_path)
    with open(output_test_dir, 'w', encoding='utf-8') as f2:
        f1.write('\n'.join(words))

'''
@author: Yakun
'''


def word_to_idx(example, word2idx):
    # according to word2idx, transfer example to numerical form.
    wordlist = [one for one in example]
    idx = []
    for word in wordlist:
        id = 1
        with open(word2idx, encoding='utf-8') as f:
            for line in f:
                line1 = line.strip()
                if word == line1:
                    idx.append(id)
                    break
                id += 1
            else:
                idx.append(None)
    print(idx)
    return idx  # return array


'''
@author: Yakun
'''


def idx_to_word(example, idx2word):
    # according to idx2word, transfer example to textual form.
    # attention:the example must be down the form 'list',for exmaple:[1,2,3,24]
    wordlist = []
    for id in example:
        i = 1
        num = int(id)
        with open(idx2word, encoding='utf-8') as f:
            for line in f:
                if num == i:
                    wordlist.append(line.strip())
                    break
                i += 1
            else:
                wordlist.append(None)
    print(wordlist)
    return wordlist  # return text
