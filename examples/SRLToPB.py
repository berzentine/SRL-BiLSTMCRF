from __future__ import print_function
__author__ = 'Nidhi'
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
import copy
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from neuronlp2 import utils
from neuronlp2.models import BiRecurrentConvCRF
from neuronlp2.io import get_logger, srl_data, CoNLL03Writer, conll03_data
from span_based_f1_measure import proxy_eval

uid = uuid.uuid4().hex[:6]

def evaluate(output_file):
    # open the file and make list of lists
    pred_labels_string = []
    gold_labels_string = []
    pred_sample, gold_sample = [], []
    ignore_class = ["V"]
    all_labels = set()
    # tgt pred are last 2 columns in this file
    with open(output_file, 'r') as out:
         for lines in out:
             if len(lines.split())>1:


                 # Issue is here, going out of range
                 # 17  - 0 O O
                 tgt, pred = lines.split()[4], lines.split()[5]
                 if tgt!='O':
                     all_labels.add(tgt[2:])
                 if pred!='O':
                     all_labels.add(pred[2:])
                 pred_sample.append(pred)
                 gold_sample.append(tgt)
             else:
                 pred_labels_string.append(pred_sample)
                 gold_labels_string.append(gold_sample)
                 pred_sample, gold_sample = [], []


    stats = proxy_eval(None, None, pred_labels_string, gold_labels_string, ignore_class, list(all_labels))
    #print(stats)
    precision = stats["precision-overall"]
    recall = stats["recall-overall"]
    f1_measure  =  stats["f1-measure-overall"]
    return 0, 100*precision, 100*recall, 100*f1_measure
    # convert to get this
    ##pred_labels = seq_length X num_clases
    ##gold_labels = seq_length
    ##call the span based f1 class and return f1 measure
    #pass
    # should evaluate the input file and return a score for each metric

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
    parser.add_argument('--transfer_num_epochs', type=int, default=100, help='Number of training epochs after transfering the model')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    # Arguments for provding where to get transfer learn data

    parser.add_argument('--num_layers', type=int, default=1, help='Number of filters in CNN')
    parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')

    parser.add_argument('--t_train')
    parser.add_argument('--t_dev')
    parser.add_argument('--result_path',  type=str)
    parser.add_argument('--t_test')
    parser.add_argument('--alphabet_path', type=str, default="data/alphabets/transfer_srl_crf_12/")
    parser.add_argument('--transfer', type=bool, default=True, help='Flag to activate transfer learning') # flag to either run the transfer learning module or not
    parser.add_argument('--model_best_path', type=str, default="tmp/model_best.pth")

    args = parser.parse_args()

    logger = get_logger("SRLCRF")

    mode = args.mode
    model_best_path = args.model_best_path
    train_path = args.train
    transfer_num_epochs = args.transfer_num_epochs
    dev_path = args.dev
    test_path = args.test
    transfer_train_path = args.t_train
    transfer_dev_path = args.t_dev
    transfer_test_path = args.t_test
    transfer = args.transfer
    num_epochs = args.num_epochs
    alphabet_path = args.alphabet_path
    batch_size = args.batch_size
    res_path = args.result_path
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

    with open(res_path, 'w') as rp:
        #print(args)
        dic_args = vars(args)
        for d in dic_args:
            rp.write(d+':'+str(dic_args[d])+'\t')
        rp.write('\n')

    ###############################################################################################################################
    # Load Data from CoNLL task for SRL and the Transfer Data
    # Create alphabets from BOTH SLR and Process Bank
    ###############################################################################################################################
    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    chunk_alphabet, srl_alphabet, transfer_alphabet = srl_data.create_alphabets(alphabet_path, train_path,
                                                                 data_paths=[dev_path, test_path],
                                                                 transfer_train_path = transfer_train_path,
                                                                 transfer_data_paths=
                                                                 [transfer_dev_path, transfer_test_path], transfer=transfer,
                                                                 embedd_dict=embedd_dict,
                                                                 max_vocabulary_size=60000
                                                                 )

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("SRL Alphabet Size: %d" % srl_alphabet.size())
    logger.info("Transfer Alphabet Size: %d" % transfer_alphabet.size())


    logger.info("Reading Data into Variables")
    use_gpu = torch.cuda.is_available()

    data_train = srl_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                    chunk_alphabet, srl_alphabet, use_gpu=use_gpu)
    num_data = sum(data_train[1])
    num_labels = srl_alphabet.size()

    data_dev = srl_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet,
                                                  chunk_alphabet, srl_alphabet, use_gpu=use_gpu, volatile=True)
    data_test = srl_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   chunk_alphabet, srl_alphabet, use_gpu=use_gpu, volatile=True)
    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, srl_alphabet)


    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[srl_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in embedd_dict:
                embedding = embedd_dict[word]
            elif word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    char_dim = args.char_dim
    trig_dim = args.trig_dim
    window = 3
    num_layers = args.num_layers
    tag_space = args.tag_space
    initializer = nn.init.xavier_uniform

    if args.dropout == 'std':
        #print(tag_space, 'tag space', trig_dim, chunk_alphabet.size())

        network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(),
                                     char_dim, char_alphabet.size(), trig_dim, chunk_alphabet.size(),
                                     num_filters, window,
                                     mode, hidden_size, num_layers, num_labels,
                                     tag_space=tag_space, embedd_word=word_table, p_rnn=p, bigram=bigram)
    else:
        raise NotImplementedError

    if use_gpu:
        network.cuda()

    lr = learning_rate
    optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d, crf=%s" % (
        mode, num_layers, hidden_size, num_filters, tag_space, 'bigram' if bigram else 'unigram'))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, dropout: %.2f, unk replace: %.2f)" % (
        gamma, num_data, batch_size, p, unk_replace))

    num_batches = num_data / batch_size + 1
    dev_f1 = 0.0
    dev_acc = 0.0
    dev_precision = 0.0
    dev_recall = 0.0
    test_f1 = 0.0
    test_acc = 0.0
    test_precision = 0.0
    test_recall = 0.0
    best_epoch = 0

    #best_model_wts = None
    # Done till here
    #exit(0)
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, mode, args.dropout, lr, decay_rate, schedule))
        train_err = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            word, char, _, chunks, labels, masks, lengths = srl_data.get_batch_variable(data_train, batch_size,
                                                                                       unk_replace=unk_replace)

            optim.zero_grad()
            loss = network.loss(word, char, chunks, labels, mask=masks)
            #print(loss.volatile)
            #print(word.volatile, char.volatile, chunks.volatile, labels.volatile)

            loss.backward()
            optim.step()

            num_inst = word.size(0)
            train_err += loss.data[0] * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 100 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                    batch, num_batches, train_err / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

            #break
            # TODO: If keyboard interrupt, kill the process and just call transfer once before killing


        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

        # evaluate performance on dev data
        network.eval()
        tmp_filename = 'tmp/%s_dev%d' % (str(uid), epoch)
        writer.start(tmp_filename)

        for batch in srl_data.iterate_batch_variable(data_dev, batch_size):
            word, char, pos, chunk, labels, masks, lengths = batch
            preds, _ = network.decode(word, char, chunk, target=labels, mask=masks,
                                         leading_symbolic=srl_data.NUM_SYMBOLIC_TAGS)
            writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                         preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
        writer.close()
        acc, precision, recall, f1 = evaluate(tmp_filename)
        with open(res_path, 'a') as rp:
            rp.write('Epoch: '+str(epoch)+' Result: '+str(precision)+' '+str(recall)+' '+str(f1)+'\n')

        print('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))

        """# TODO: delete this
        best_model_wts = copy.deepcopy(network.state_dict())"""

        if dev_f1 < f1:
            dev_f1 = f1
            dev_acc = acc
            dev_precision = precision
            dev_recall = recall
            best_epoch = epoch
            best_model_wts = copy.deepcopy(network.state_dict())
            #network.save_state_dict(model_best_path)
            # evaluate on test data when better performance detected
            tmp_filename = 'tmp/%s_test%d' % (str(uid), epoch)
            writer.start(tmp_filename)

            for batch in srl_data.iterate_batch_variable(data_test, batch_size):
                word, char, pos, chunk, labels, masks, lengths = batch
                preds, _ = network.decode(word, char, chunk, target=labels, mask=masks,
                                          leading_symbolic=srl_data.NUM_SYMBOLIC_TAGS)
                writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                             preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
            writer.close()
            test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename)

        print("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
            dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
        print("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
            test_acc, test_precision, test_recall, test_f1, best_epoch))

        if epoch % schedule == 0:
            lr = learning_rate / (1.0 + epoch * decay_rate)
            optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

    #plot(str(uid),res_path)
    def transfer(network):
        print('In transfer')
        pb_alphabet = transfer_alphabet
        # Load data from Process Bank
        transfer_data_train = srl_data.read_data_to_variable(transfer_train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                      chunk_alphabet, pb_alphabet, use_gpu=use_gpu)
        transfer_data_dev = srl_data.read_data_to_variable(transfer_train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                       chunk_alphabet, pb_alphabet, use_gpu=use_gpu, volatile=True)
        transfer_data_test = srl_data.read_data_to_variable(transfer_test_path, word_alphabet, char_alphabet, pos_alphabet,
                                                       chunk_alphabet, pb_alphabet, use_gpu=use_gpu, volatile=True)

        num_data = sum(transfer_data_train[1])
        num_labels = pb_alphabet.size()
        writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, pb_alphabet)
        # Load previous best trained model
        network.load_state_dict(best_model_wts)
        # Change the last layer to incorporate new Softmax layer
        out_dim = network.od
        import torch.nn as nn
        from neuronlp2.nn.modules import ChainCRF
        network.dense_softmax = nn.Linear(out_dim, num_labels)
        network.crf = ChainCRF(out_dim, num_labels, bigram=bigram)

        if use_gpu:
            network = network.cuda()
        lr = learning_rate
        optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
        logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d, crf=%s" % (
            mode, num_layers, hidden_size, num_filters, tag_space, 'bigram' if bigram else 'unigram'))
        logger.info("training: l2: %f, (#training data: %d, batch: %d, dropout: %.2f, unk replace: %.2f)" % (
            gamma, num_data, batch_size, p, unk_replace))

        num_batches = num_data / batch_size + 1
        dev_f1 = 0.0
        dev_acc = 0.0
        dev_precision = 0.0
        dev_recall = 0.0
        test_f1 = 0.0
        test_acc = 0.0
        test_precision = 0.0
        test_recall = 0.0
        best_epoch = 0
        # Done till here

        for epoch in range(1, transfer_num_epochs + 1):
            print(transfer_num_epochs, 'total epochs to do')
            print('Transfer Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
                epoch, mode, args.dropout, lr, decay_rate, schedule))
            train_err = 0.
            train_total = 0.

            start_time = time.time()
            num_back = 0
            network.train()
            for batch in range(1, num_batches + 1):
                word, char, _, chunks, labels, masks, lengths = srl_data.get_batch_variable(transfer_data_train, batch_size,
                                                                                           unk_replace=unk_replace)

                optim.zero_grad()
                loss = network.loss(word, char, chunks, labels, mask=masks)
                loss.backward()
                optim.step()

                num_inst = word.size(0)
                train_err += loss.data[0] * num_inst
                train_total += num_inst
                time_ave = (time.time() - start_time) / batch
                time_left = (num_batches - batch) * time_ave
                # update log
                #print(train_err, 'train error')
                if batch % 10 == 0:
                    sys.stdout.write("\b" * num_back)
                    sys.stdout.write(" " * num_back)
                    sys.stdout.write("\b" * num_back)
                    log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                        batch, num_batches, train_err / train_total, time_left)
                    sys.stdout.write(log_info)
                    sys.stdout.flush()
                    num_back = len(log_info)


            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

            # evaluate performance on dev data
            network.eval()
            tmp_filename = 'tmp/%s_dev_transfer%d' % (str(uid), epoch)
            writer.start(tmp_filename)

            for batch in srl_data.iterate_batch_variable(transfer_data_dev, batch_size):
                word, char, pos, chunk, labels, masks, lengths = batch
                preds, _ = network.decode(word, char, chunk, target=labels, mask=masks,
                                             leading_symbolic=srl_data.NUM_SYMBOLIC_TAGS)
                writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                             preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
            writer.close()
            acc, precision, recall, f1 = evaluate(tmp_filename)
            with open(res_path, 'a') as rp:
                rp.write('Epoch: '+str(epoch)+' Result: '+str(precision)+' '+str(recall)+' '+str(f1)+'\n')

            print('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))

            if dev_f1 < f1:
                dev_f1 = f1
                dev_acc = acc
                dev_precision = precision
                dev_recall = recall
                best_epoch = epoch

                # evaluate on test data when better performance detected
                tmp_filename = 'tmp/%s_test_transfer%d' % (str(uid), epoch)
                writer.start(tmp_filename)

                for batch in srl_data.iterate_batch_variable(transfer_data_test, batch_size):
                    word, char, pos, chunk, labels, masks, lengths = batch
                    preds, _ = network.decode(word, char, chunk, target=labels, mask=masks,
                                              leading_symbolic=srl_data.NUM_SYMBOLIC_TAGS)
                    writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                                 preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
                writer.close()
                test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename)

            print("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
                dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
            print("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
                test_acc, test_precision, test_recall, test_f1, best_epoch))

            if epoch % schedule == 0:
                lr = learning_rate / (1.0 + epoch * decay_rate)
                optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

        #plot(str(uid),res_path)
        # Learn and Evaluate on that
    transfer(network)



if __name__ == '__main__':
    main()
