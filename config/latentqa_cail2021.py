model_cfg = {
    'embedding_dim': 100,  # if use GloVe, please make sure this equals to train_cfg['glove_dim']
    'encoder_dim': 256,
    'decoder_dim': 256,
    'num_encoder_layers': 1,
    'num_decoder_layers': 1,
    'latent_dim': 100,  # dimension of latent Representation
    'num_dist_word_selected': 3,
    'temperature_initial': 1e-3,
    'mode': 'gaussian',
}


train_cfg = {
    'bqak_train_path': None,
    'bqak_test_path': None,
    'cqa_train_path': r'/data/wangzekun/KQA/question_answer_background/question-answer-train.json',
    'cqa_test_path': r'/data/wangzekun/KQA/question_answer_background/question-answer-test.json',
    'tokenized_train_path': None,
    'vocab_char_path': None,
    'vector_char_path': None,
    'vocab_word_path': r'/data/wangzekun/KQA/zhs_wiki_vocab.txt',
    'vector_word_path': r'/data/wangzekun/KQA/zhs_wiki_glove.vectors.100d.txt.txt',
#     'bqak_train_path': None,
#     'bqak_test_path': None,
#     'cqa_train_path': r'./asset/question_answer_background/question-answer-train.json',
#     'cqa_test_path': r'./asset/question_answer_background/question-answer-test.json',
#     'tokenized_train_path': None,
#     'vocab_char_path': None,
#     'vector_char_path': None,
#     'vocab_word_path': r'./asset/zhs_wiki_vocab.txt',
#     'vector_word_path': r'./asset/zhs_wiki_glove.vectors.100d.txt.txt',
    'batch_size' : 16,
    'use_gpu': True,
    'use_char': False,
    'vocab_size': int(964724 * 0.10), # how many words in vocab do we use
    'LTP_config': 'small',
    'max_length': 512,
    'batch_first': True,
    'glove_dim': 100,
    'tokenizer': 'ltp',  # 'ltp', 'jieba'
    'init_tokenizer_by_own_vocab': False,
    'is_train_tokenized': False,
}