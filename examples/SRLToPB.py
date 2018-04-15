from __future__ import print_function
__author__ = 'nidhi'
"""
Implementation of Tranfer Learning Argument-Trigger Modelling from SRL CoNLL data using Bi-directional LSTM-CNNs-CRF model
"""


import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid

import numpy as np
import torch
from torch.optim import Adam, SGD
from neuronlp2 import utils

from neuronlp2.io import get_logger, srl_data, CoNLL03Writer



#Part 1

# train the first model
    # Keep saving the best model
    # Change the scoring script too http://www.lsi.upc.edu/~srlconll/examples.html

# part 1 should return the best saved model based on validation
# TODO: Also log accuracy and precision for graphs
def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', required=True)
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN')
    parser.add_argument('--tag_space', type=int, default=0, help='Dimension of tag space')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
    parser.add_argument('--trig_dim', type=int, default=100, help='Dimension of Trigger embeddings')
    parser.add_argument('--learning_rate', type=float, default=0.015, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--dropout', choices=['std', 'variational'], help='type of dropout', required=True)
    parser.add_argument('--p', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    # Arguments for provding where to get transfer learn data
    parser.add_argument('--t_train')
    parser.add_argument('--t_dev')
    parser.add_argument('--t_test')
    parser.add_argument('--transfer', type=bool, default=True, help='Flag to activate transfer learning') # flag to either run the transfer learning module or not


    args = parser.parse_args()

    logger = get_logger("SRLCRF")

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    transfer_train_path = args.t_train
    transfer_dev_path = args.t_dev
    transfer_test_path = args.t_test
    transfer = args.transfer
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule
    p = args.p
    unk_replace = args.unk_replace
    bigram = args.bigram
    embedding = args.embedding
    embedding_path = args.embedding_dict

    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)
    ###############################################################################################################################
    # Load Data from CoNLL task for SRL and the Transfer Data
    # Create alphabets from BOTH SLR and Process Bank
    ###############################################################################################################################
    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    chunk_alphabet, srl_alphabet, transfer_alphabet = srl_data.create_alphabets("data/alphabets/srl_crf/", train_path,
                                                                 data_paths=[dev_path, test_path],
                                                                 transfer_train_path = transfer_train_path,
                                                                 transfer_data_paths=
                                                                 [transfer_dev_path, transfer_test_path], transfer=transfer,
                                                                 embedd_dict=embedd_dict,
                                                                 max_vocabulary_size=55000
                                                                 )

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("SRL Alphabet Size: %d" % srl_alphabet.size())
    logger.info("Transfer Alphabet Size: %d" % transfer_alphabet.size())


    logger.info("Reading Data into Variables")
    use_gpu = torch.cuda.is_available()



#Part 2
# Update the new model - transfer learning part
    # Load data from Process Bank
    # Load previous best trained model
    # Change the last layer to incorporate new Softmax layer
    # Learn and Evaluate on that


if __name__ == '__main__':
    main()
