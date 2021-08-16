model_cfg = {
    'embedding_dim': 100,  # if use GloVe, please make sure this equals to train_cfg['glove_dim']
    'encoder_dim': 256,
    'decoder_dim': 256,
    'num_encoder_layers': 1,
    'num_decoder_layers': 1,
    'latent_dim': 100,  # dimension of latent Representation
    'num_dist_word_selected': 3,
    'temperature_initial': 1e-3
    
}


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
    'use_gpu': True,
    'use_char': False,
    'vocab_char_path': None,
    'vector_char_path': None,
    'vocab_word_path': r'../asset/zhs_wiki_vocab.txt',
    'vector_word_path': r'./asset/zhs_wiki_glove.vectors.100d.txt.txt',
    'LTP_config': 'small',
    'max_length': 512,
    'batch_first': True,
    'glove_dim': 100,
    'is_train_tokenized': False,
    'tokenized_train_path': r'./asset/question_answer_background/tok-question-answer-train.json-answer-train.json',
}