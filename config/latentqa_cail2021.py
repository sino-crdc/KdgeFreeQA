train_cfg = {
    # 'case_train_path': r'/data/wangzekun/KQA/case/train.json',
    # 'case_test_path': r'/data/wangzekun/KQA/case/test.json',
    # 'bqa_train_path': r'/data/wangzekun/KQA/question_answer_background/question-answer-train.json',
    # 'bqa_test_path': r'/data/wangzekun/KQA/question_answer_background/question-answer-test.json',
    'case_train_path': r'./asset/case/train.json',
    'case_test_path': r'./asset/case/test.json',
    'bqa_train_path': r'./asset/question_answer_background/question-answer-train.json',
    'bqa_test_path': r'./asset/question_answer_background/question-answer-test.json',
    'batch_size' : 16,
    'use_gpu' : True,
    'use_char' : True,
    'vocab_char_path' : r'./asset/vocab.txt',
    'vector_char_path' : r'./asset/zhs_wiki_glove.vectors.100d.txt.txt',
    'vocab_word_path' : None,
    'vector_word_path' : None,
    'LTP_config' : 'small',

}