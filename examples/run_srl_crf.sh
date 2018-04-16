#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python examples/SRLCRF.py --mode LSTM --num_epochs 1 --batch_size 32 --hidden_size 512 \
 --char_dim 30 --num_filters 30 --tag_space 128 \
 --learning_rate 0.005 --decay_rate 0.05 --schedule 1 --gamma 0.0 \
 --dropout std --p 0.5 --unk_replace 0.0 --bigram \
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.100d.gz" \
 --train "data/conll2012/english/train_wtriggers.txt" \
 --dev "data/conll2012/english/dev_wtriggers.txt" \
 --test "data/conll2012/english/test_wtriggers.txt" \
 --result_path "tmp/results_test.txt"
 --
#--train "data/conll2003/english/eng.train.bioes.conll" --dev "data/conll2003/english/eng.dev.bioes.conll" --test "data/conll2003/english/eng.test.bioes.conll"
