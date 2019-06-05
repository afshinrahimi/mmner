"""
Models for supervised meta-learning.
"""



import numpy as np
import tensorflow as tf

import random


UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

#DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))



class NERModel:
    """
    NER model copied from glample https://github.com/guillaumegenthial/sequence_tagging
    with very small modifications.
    """
    def __init__(self, config, var_given=False, vars=None):
        self.config = config
        self.lr_backup = config.lr
        self.logger = config.logger
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.var_given = var_given
        self.vars = vars
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        #self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings[list(self.config.embeddings)[0]],
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                if self.var_given and 'char_emb' in self.vars:
                    _char_embeddings = self.vars['char_emb']
                else:
                    # get char embeddings matrix
                    _char_embeddings = tf.get_variable(
                            name="_char_embeddings",
                            dtype=tf.float32,
                            shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                if self.var_given and 'char_lstmfw_kernel' in self.vars and 'char_lstmbw_kernel' in self.vars:
                    cell_fw = MyLSTMCell(self.config.hidden_size_char,
                                         state_is_tuple=True, var_given=True,
                                         kernel=self.vars['char_lstmfw_kernel'],
                                         bias=self.vars['char_lstmfw_bias'])
                    cell_bw = MyLSTMCell(self.config.hidden_size_char,
                                         state_is_tuple=True, var_given=True,
                                         kernel=self.vars['char_lstmbw_kernel'],
                                         bias=self.vars['char_lstmbw_bias'])
                else:
                    # bi lstm on chars
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                            state_is_tuple=True)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                            state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                self.chr = output
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            if self.var_given and 'lstmfw_kernel' in self.vars and 'lstmbw_kernel' in self.vars:
                cell_fw = MyLSTMCell(self.config.hidden_size_lstm, var_given=True,
                                     kernel=self.vars['lstmfw_kernel'], bias=self.vars['lstmfw_bias'])
                cell_bw = MyLSTMCell(self.config.hidden_size_lstm, var_given=True,
                                     kernel=self.vars['lstmbw_kernel'], bias=self.vars['lstmbw_bias'])
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
            self.h = output

        with tf.variable_scope("proj"):
            if self.var_given and 'proj_w' in self.vars and 'proj_b' in self.vars:
                W = self.vars['proj_w']
                b = self.vars['proj_b']
            else:
                W = tf.get_variable("W", dtype=tf.float32,
                        shape=[2*self.config.hidden_size_lstm, self.config.ntags])

                b = tf.get_variable("b", shape=[self.config.ntags],
                        dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])





    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            if self.var_given and 'transitions' in self.vars:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths, self.vars['transitions'])
            else:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                        self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

    def predict_batch(self, words, session, dropout=1.0):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=dropout)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = session.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = session.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)#, var_list=[v for v in tf.trainable_variables() if 'mixture' in v.name])



    def train(self, train, dev, session, model_state, lang_order=None, softmax=None, short_names=None,
              lang=None, verbose=False, best_score=0):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        nepoch_no_imprv = 0 # for early stopping
        score = 0
        best_params = model_state.export_variables()
        num_samples = len(train)
        lowres = num_samples <= 1000
        self.logger.info('lr:{}'.format(self.config.lr))
        custom_nepoch_no_imprv = self.config.lowres_nepoch_no_imprv if lowres else self.config.nepoch_no_imprv
        for epoch in range(self.config.nepochs):
            #added by afshin for efficiency
            #for i in range(10):
            #    self.run_epoch(train, dev, epoch, eval_dev=False)
            score = self.run_epoch(train=train, dev=dev, epoch=epoch, session=session, eval_dev=True if dev else False)
            self.config.lr *= self.config.lr_decay # decay learning rate


            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                #self.save_session()
                best_params = model_state.export_variables()
                best_score = score
                #self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= custom_nepoch_no_imprv:
                    if verbose:
                        self.logger.info("- early stopping {} epochs without "\
                                "improvement, total epochs {}".format(nepoch_no_imprv, epoch))
                    break
            if verbose:
                self.logger.info("Epoch {:} out of {:} score {:.2f}".format(epoch + 1,
                                                               self.config.nepochs, score))
        #reset config.lr
        self.config.lr = self.lr_backup
        return best_score, best_params




    def run_evaluate(self, test, session, isDev=False):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        eval_batch_size = max(self.config.batch_size, self.config.test_batch_size)
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, eval_batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words, session, dropout=self.config.dev_dropout if isDev else 1.0)
            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}

    def tag(self, test, session):
        eval_batch_size = max(self.config.batch_size, self.config.test_batch_size)
        all_predicted_labels = []
        all_sequence_lengths = []
        for words, labels in minibatches(test, eval_batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words, session, dropout=1.0)
            all_sequence_lengths.append(sequence_lengths)
            for lab_pred, length in zip(labels_pred,
                                             sequence_lengths):
                #lab      = lab[:length]
                lab_pred = lab_pred[:length]
                all_predicted_labels.append(lab_pred)



        return all_predicted_labels, all_sequence_lengths

    def evaluate(self, test, session, isDev=False):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        metrics = self.run_evaluate(test, session, isDev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        #self.logger.info('Test performance after training on lowres: ' + msg)
        return metrics

    def run_epoch(self, train, dev, epoch, session, eval_dev=True):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        #prog = Progbar(target=nbatches)
        train_losses = []
        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)
            _, train_loss = session.run(
                    [self.train_op, self.loss], feed_dict=fd)

            train_losses.append(train_loss)
            #prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            #if i % 10 == 0:
            #    self.file_writer.add_summary(summary, epoch*nbatches + i)
        if eval_dev:
            metrics = self.run_evaluate(dev, session, isDev=False)
            msg = " - ".join(["{} {:04.2f}".format(k, v)
                    for k, v in metrics.items()])
            #self.logger.info(msg)

            return metrics["f1"]
        return np.mean(train_losses)


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

def sentence_minibatches(sentences, minibatch_size):
    """
    Args:
        data: generator of sentence
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch= []
    for x in sentences:
        if len(x_batch) == minibatch_size:
            yield x_batch
            x_batch = []
    if len(x_batch) != 0:
        yield x_batch

def _mini_batches_conll(samples, batch_size, num_batches, replacement, shuffle=True):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    batch_count = 0
    while True:
        if shuffle:
            random.shuffle(samples)
        x_batch, y_batch = [], []
        for (x, y) in samples:
            if len(x_batch) == batch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
                batch_count += 1
                if batch_count == num_batches:
                    return

            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]

        if len(x_batch) != 0:
            yield x_batch, y_batch
            return







class NERModelMultiAnnotator(NERModel):
    '''
    A NERModel that accepts multiple datasets and trains on them.
    The sampling strategy is based on the confidence in each annotator
    '''
    def __init__(self, config, var_given=False, vars=None):
        super(NERModelMultiAnnotator, self).__init__(config, var_given, vars)


    def set_annotation_info(self, annotators, annotator_expertise, annotations):
        # name of annotators
        self.annotators = annotators
        # confidence in annotators a vector of (#n_annotator,) size
        self.annotator_expertise = annotator_expertise
        #annotations (labelled datasets) by annotators, a dicionary of {annotator_name:dataset}
        self.annotations = annotations

    def run_epoch_unsup(self, train, dev, epoch, session, eval_dev=True):
        """Performs one complete pass over the train set and evaluate on dev
        here the train set is a sample from predictions of source high resource
        models. The samplng is based on model expertise (self.annotator_expertise).

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        num_explore = self.config.num_unsup_epochs
        n_annotator_samples = self.config.test_batch_size
        for n_explore in range(num_explore):
            #select samples from annotations based on annotator expertise (a multinomial distribution)
            num_samples = np.random.multinomial(n_annotator_samples, self.annotator_expertise)
            #now sample from annotations based on num_samples
            all_samples = []
            for i, n in enumerate(np.nditer(num_samples)):
                n = int(n)
                if n == 0:
                    continue
                annotator = self.annotators[i]
                samples = self.annotations[annotator].sample(n)
                all_samples.extend(samples)
            #now sample something from the most expert annotator (gold data).
            num_samples_gold = self.config.num_unsup_gold
            if num_samples_gold:
                all_samples.extend(train.sample(num_samples_gold))
            np.random.shuffle(all_samples)
            #now iterate over it and optimise
            train_losses = []
            for i, (words, labels) in enumerate(
                    _mini_batches_conll(all_samples, batch_size=self.config.test_batch_size,
                                        num_batches=1, replacement=False, shuffle=True)):

                fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                        self.config.dropout)
                _, train_loss = session.run(
                        [self.train_op, self.loss], feed_dict=fd)

                train_losses.append(train_loss)

        if eval_dev:
            metrics = self.run_evaluate(dev, session, isDev=False)
            msg = " - ".join(["{} {:04.2f}".format(k, v)
                    for k, v in metrics.items()])
            #self.logger.info(msg)
            return metrics["f1"]

        return np.mean(train_losses)


    def train_unsup(self, train, dev, session, model_state, lang_order=None, softmax=None, short_names=None, lang=None, verbose=False):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset (note that dev might be actually the train dataset here).

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        score = 0
        best_params = None
        num_samples = len(train)
        lowres = num_samples <= 1000
        self.logger.info('lr:{}'.format(self.config.lr))
        custom_nepoch_no_imprv = self.config.lowres_nepoch_no_imprv if lowres else self.config.nepoch_no_imprv
        for epoch in range(self.config.nepochs):
            score = self.run_epoch_unsup(train=train, dev=dev, epoch=epoch, session=session, eval_dev=True if dev else False)
            self.config.lr *= self.config.lr_decay # decay learning rate


            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                #self.save_session()
                best_params = model_state.export_variables()
                best_score = score
                #self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= custom_nepoch_no_imprv:
                    if verbose:
                        self.logger.info("- early stopping {} epochs without "\
                                "improvement, total epochs {}".format(nepoch_no_imprv, epoch))
                    break
            if verbose:
                self.logger.info("Epoch {:} out of {:} score {:.2f}".format(epoch + 1,
                                                               self.config.nepochs, score))
        #reset config.lr
        self.config.lr = self.lr_backup
        return best_score, best_params
