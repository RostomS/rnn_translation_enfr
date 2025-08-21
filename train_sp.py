import sentencepiece as spm

# Entraîner le tokenizer anglais
spm.SentencePieceTrainer.train(input='english.txt', model_prefix='english_v2', vocab_size=16000, character_coverage=1.0,model_type='bpe')

# Entraîner le tokenizer français
spm.SentencePieceTrainer.train(input='french.txt',model_prefix='french_v2',vocab_size=16000,character_coverage=1.0,model_type='bpe')

print("Tokenizers entraînés avec succès.")
