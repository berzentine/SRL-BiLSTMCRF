__author__ = 'Nidhi'

#Part 2
# Update the new model - transfer learning part
def transfer():

    # load and write the PB tags
    _, _,_, _, pb_alphabet = conll03_data.create_alphabets(pb_alphabet_path, transfer_train_path,
                                                                 data_paths=[transfer_dev_path, transfer_test_path],
                                                                 embedd_dict=embedd_dict,
                                                                 max_vocabulary_size=50000)
    # Load data from Process Bank
    transfer_data_train = conll03_data.read_data_to_variable(transfer_train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                  chunk_alphabet, pb_alphabet, use_gpu=use_gpu, volatile=True)
    transfer_data_test = conll03_data.read_data_to_variable(transfer_test_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   chunk_alphabet, pb_alphabet, use_gpu=use_gpu, volatile=True)

    num_data = sum(transfer_data_train[1])
    num_labels = pb_alphabet.size()
    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, pb_alphabet)

    network.load_state_dict(torch.load(model_best_path))
    # Load previous best trained model
    # Change the last layer to incorporate new Softmax layer
    model_ft = network(pretrained=True)
    out_dim = model_ft.od
    model_ft.dense_softmax = nn.Linear(out_dim, num_labels)
    model_ft.crf = ChainCRF(out_dim, num_labels, bigram=bigram)

    if use_gpu:
        model_ft = model_ft.cuda()
    lr = learning_rate
    optim = SGD(model_ft.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
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
    #exit(0)
    # Learn and Evaluate on that




    pass
