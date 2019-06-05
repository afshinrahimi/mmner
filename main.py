import matplotlib as mlp
mlp.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pdb
import itertools
import pickle
import gzip
import os
import random
import pdb
import tensorflow as tf
import logging
import numpy as np
import json
import io
import os
import sys
import tarfile
from collections import defaultdict

from models import NERModel, NERModelMultiAnnotator
from config import  read_dataset_panx_individual, Config,read_dataset_panx_multiannotated, count_unks

from variables import VariableState

from utils import annotation_to_byteio, annotation_to_byteio_yuan


args = None
config = None
fig_dir = './figs/'
log_dir = './logs/'
hr_samples = 20000
lr_samples = 100
highres_datasets = {}
lowres_datasets = {}


def dump(obj):
    #log the configuration attributes
    for attr in dir(obj):
        if attr in ['vocab_chars', 'embeddings', 'filename_trimmed', 'processing_word',
                    'word_vocab', 'vocab_words', '__dict__', 'filename_words']:
            continue
        if attr.startswith('__'):
            continue
        logging.info("%s = %r" % (attr, getattr(obj, attr)))




def load_highres_datasets():
    global highres_datasets
    for langid in config.highres_langs:
        dataset = read_dataset_panx_individual(config, langid, max_iter=hr_samples)
        highres_datasets[langid] = dataset
        logging.info('loading {} with {} samples'.format(langid, hr_samples))


def load_lowres_datasets():
    global lowres_datasets
    for langid in config.lowres_langs:
        dataset = read_dataset_panx_individual(config, langid, max_iter=lr_samples)
        lowres_datasets[langid] = dataset
        logging.info('loading {} with {} samples'.format(langid, lr_samples))


def save_model_params(info, filename):
    logging.info('saving model in {}'.format(filename))
    with gzip.open(filename, 'wb') as fout:
        pickle.dump(info, fout)

def load_model_params(filename):
    logging.info('loading model from {}'.format(filename))
    with gzip.open(filename, 'rb') as fin:
        info = pickle.load(fin)
    return info

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)







def train_lang(lang, dataset, config, writer, init_vars=None):
    set_seed(args.seed)
    train_set, dev_set, test_set = dataset['train'], dataset['dev'], dataset['test']
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            if writer is None:
                writer = tf.summary.FileWriter(log_dir)

            merged, dev_acc_ph, dev_f1_ph, test_acc_ph, test_f1_ph = merge_summaries(lang)
            model = NERModel(config)
            sess.run(tf.global_variables_initializer())
            model_state = VariableState(sess, tf.trainable_variables())
            model_wembed_state = VariableState(sess, [v for v in tf.global_variables() if 'words/_word_embeddings:0' == v.name])
            model_wembed_state.import_variables([config.embeddings[lang]])
            if init_vars:
                logging.info('importing variables...')
                model_state.import_variables(init_vars)
            logging.info('training lang={}...'.format(lang))
            best_score, best_params = model.train(train=train_set, dev=dev_set, session=sess, model_state=model_state, verbose=args.verbose)
            model_state.import_variables(best_params + [config.embeddings[lang]])
            msgtest = model.evaluate(test_set, sess)
            test_acc = msgtest['acc']
            test_f1 = msgtest['f1']
            logging.info('test lang:{} acc:{} f1:{}'.format(lang, test_acc, test_f1))
            msgdev = model.evaluate(dev_set, sess, isDev=False)
            acc = msgdev['acc']
            f1 = msgdev['f1']
            score = {'dev':msgdev, 'test':msgtest}
            logging.info('dev lang:{} acc:{} f1:{}'.format(lang, acc, f1))
            summary = sess.run(merged,
                               feed_dict={dev_f1_ph: f1, dev_acc_ph: acc, test_acc_ph: test_acc, test_f1_ph: test_f1})
            writer.add_summary(summary)
            #params = model_state.export_variables()
            params = best_params
            param_names = {i: v.name for i, v in enumerate(tf.trainable_variables())}
    return param_names, params, score

def individual_model(args, writer):
    """
    Load data and train a model on it.
    """
    logging.info(config)
    logging.info('only running on {} langs....'.format('highres' if args.indhigh else 'lowres'))
    f1s = {}
    accs = {}
    scores = {}
    datasets = highres_datasets if args.indhigh else lowres_datasets

    for lang, dataset in datasets.items():
        param_names, params, msg = train_lang(lang, dataset, config, writer)
        info = {'param_names':param_names, 'params':params, 'score':msg, 'lang':lang}
        scores[lang] = msg
        if args.save:
            filename = os.path.join(config.params_dir, '{}lr{}drop{}'.format(lang, config.lr, config.dropout))
            save_model_params(info, filename)
    logging.info(scores)
    return scores




def merge_summaries(lang):
    dev_acc_ph = tf.placeholder(tf.float32, shape=())
    dev_f1_ph = tf.placeholder(tf.float32, shape=())
    test_acc_ph = tf.placeholder(tf.float32, shape=())
    test_f1_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('dev_acc_{}'.format(lang), dev_acc_ph)
    tf.summary.scalar('dev_f1_{}'.format(lang), dev_f1_ph)
    tf.summary.scalar('test_acc_{}'.format(lang), test_acc_ph)
    tf.summary.scalar('test_f1_{}'.format(lang), test_f1_ph)
    merged = tf.summary.merge_all()
    return merged, dev_acc_ph, dev_f1_ph, test_acc_ph, test_f1_ph









def lang_tag_lang(args, writer=None):
    infos = {}
    param_names = None
    filenames = [os.path.join(config.dir_model_highres, file) for file in os.listdir(config.dir_model_highres)]
    for filename in filenames:
        lang = filename.split('/')[-1][0:2]
        if lang not in config.highres_langs:
            logging.info('{} not in highres but the model exists: {}'.format(lang, filename))
            continue
        info = load_model_params(filename)
        infos[lang] = info
        param_names = info['param_names']
    f1s = {}
    scores = {}
    best_params = {}
    total_job_num = len(infos) * len(highres_datasets)
    job_num = 0
    lang_by_lang_scores = defaultdict(dict)
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            model = NERModel(config)
            sess.run(tf.global_variables_initializer())
            model_state = VariableState(sess, tf.trainable_variables())
            model_wembed_state = VariableState(sess, [v for v in tf.global_variables() if 'words/_word_embeddings:0' == v.name])
            #now make predictions for each language
            for lang_u, dataset in highres_datasets.items():
                model_wembed_state.import_variables([config.embeddings[lang_u]])
                tar = tarfile.open(os.path.join(config.dir_unlabeled, f'{lang_u}.tar.gz'), "w:gz")
                for lang, info in infos.items():
                    logging.info('running job {} out of {}'.format(job_num, total_job_num))
                    job_num += 1
                    if lang == lang_u:
                        continue
                    logging.info('predicting on {} using {} model'.format(lang_u, lang))
 
                    # import the high resource model with its embeddings
                    params = infos[lang]['params']
                    model_state.import_variables(params)
                    msgtrain = model.evaluate(lowres_datasets[lang_u]['train'], sess)
                    msgdev = model.evaluate(lowres_datasets[lang_u]['dev'], sess)
                    msgtest = model.evaluate(lowres_datasets[lang_u]['test'], sess)
                    msgtrainhigh = model.evaluate(highres_datasets[lang_u]['train'], sess)
                    train_acc, train_f1 = msgtrain['acc'], msgtrain['f1']
                    lang_by_lang_scores[lang_u][lang] = {'train':msgtrain, 'dev':msgdev, 'test':msgtest, 'trainhigh': msgtrainhigh}

                    #read words from the real file
                    for dataset_partition in ['train', 'dev', 'test']:
                        label_ids, all_sequence_lengths = model.tag(dataset[dataset_partition], sess)
                        all_words = dataset[dataset_partition].sentences
                        all_records = []
                        for words, lids in zip(all_words, label_ids):
                            #one sentence with its labels
                            labels = [config.tag_vocab[id] for id in lids]
                            record = [w + '\t' + l  for w, l in zip(words, labels)]
                            all_records.append(record)
                        unl_data, unl_tarinfo = annotation_to_byteio(all_records, f'{lang_u}_{lang}_{dataset_partition}')
                        tar.addfile(tarinfo=unl_tarinfo, fileobj=unl_data)
                tar.close()
    logging.info(str(lang_by_lang_scores))
    json.dump(lang_by_lang_scores, open(config.langtaglang_file, 'w'))


def export_annotation_to_uagg(args, writer=None):
    if os.path.exists(config.dir_bccannotations) and os.listdir(config.dir_bccannotations):
        logging.warn(f'directory {config.dir_bccannotations} exists and is not empty! Delete the contents if you do not need the files. exiting...')
        sys.exit(0)
    for lang in config.lowres_langs:
        logging.info(f'exporting {lang}')
        lang_datasets = read_dataset_panx_multiannotated(config, lang, max_iter=hr_samples)
        # export highresource model labels as annotators for further examination
        hrdataset = read_dataset_panx_individual(config, lang, max_iter=hr_samples)
        # hrdataset['train'].data[0] = [[([585, 591, 624, 625, 615], 42487), ... ([594, 624, 609, 615, 578, 601], 10836)], [6, 4, 5, 5]]
        # config.vocab_tags = {'B-LOC': 0, 'B-PER': 1, 'I-PER': 2, 'I-LOC': 3, 'B-ORG': 4, 'I-ORG': 5, 'O': 6}
        # config.vocab_words = {id:word}
        # create a tar file
        tag_id = {v: k for k, v in config.vocab_tags.items()}
        yuantar = tarfile.open(os.path.join(config.dir_bccannotations, f'{lang}.tar.gz'), "w:gz")

        def dataset_to_yuan_format(dataset, filename):
            annotations = []
            data = dataset.data
            sentences = dataset.sentences
            for sentid, sentencelabel in enumerate(data):
                wordids, labelid = sentencelabel
                sentence = sentences[sentid]
                wordsequentialids = [*range(len(wordids))]
                vocabids = [word[1] for word in wordids]
                words = sentence
                labels = [tag_id[id] for id in labelid]
                for idx in range(len(wordsequentialids)):
                    annotations.append(
                        [sentid, wordsequentialids[idx], vocabids[idx], labelid[idx], labels[idx], sentence[idx]])
            unl_data, unl_tarinfo = annotation_to_byteio_yuan(annotations, filename)
            yuantar.addfile(tarinfo=unl_tarinfo, fileobj=unl_data)

        dataset_to_yuan_format(hrdataset['train'], f'{lang}_train')
        dataset_to_yuan_format(hrdataset['dev'], f'{lang}_dev')
        dataset_to_yuan_format(hrdataset['test'], f'{lang}_test')
        for fname, dataset in lang_datasets.items():
            dataset_to_yuan_format(dataset, fname)
        yuantar.close()


def multiannotator(args, writer=None):
    #load annotator confidences for each language from json file
    lang_lang_scores = json.load(open(config.langtaglang_file, 'r'))
    model_created = False
    nermodel = None
    lang_scores = {}
    unsup_lang_scores = {}
    lr_unsup = config.unsuplr
    lr_sup = config.unsupsuplr
    nepochs_backup = config.nepochs
    test_batch_size_backup = config.test_batch_size
    scores = {}
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            for lang in config.lowres_langs:
                config.test_batch_size = test_batch_size_backup
                set_seed(args.seed)
                config.load(lang)

                #very long sentences OOM error
                if lang in ['hu', 'en', 'th', 'pl', 'sl']:
                    config.test_batch_size = 20


                if not model_created:
                    nermodel = NERModelMultiAnnotator(config)
                    merged, dev_acc_ph, dev_f1_ph, test_acc_ph, test_f1_ph = merge_summaries(lang)
                    model_created = True
                else:
                    nermodel.config = config

                #normalise annotator expertise
                annotators = lang_lang_scores[lang]
                #decide which partition of the multiannotated datasets to train on train versus test
                partition = 'test' if args.testannotations else 'train'
                annotator_expertise = {f'{lang}_{k}_{partition}': v['train']['f1'] for k, v in lang_lang_scores[lang].items() if k not in ['ja', 'ko', 'zh', 'th']}
                annotator_sorted = sorted(annotator_expertise)
                expertise = [annotator_expertise[ann] for ann in annotator_sorted] if not args.uniformexpertise else [1.0] * len(annotator_sorted)
                expertise = np.array(expertise)
                #only use topk annotators
                topk = config.unsuptopk
                #assert topk <= len(annotator_sorted)
                sorted_indices = np.argsort(expertise)
                zero_out_num = len(annotator_sorted) - topk
                if zero_out_num > 0:
                    zero_out_indices = sorted_indices[0:zero_out_num] if not args.uniformexpertise else np.random.choice(sorted_indices, size=zero_out_num, replace=False)
                    expertise[zero_out_indices] = 0
                expertise = expertise / np.sum(expertise)
                #load annotations (unlabelled data labelled by highresource models from diff. languages
                lang_datasets = read_dataset_panx_multiannotated(config, lang, max_iter=hr_samples)
                #load lr dataset
                lrdataset = read_dataset_panx_individual(config, lang, max_iter=lr_samples)

                nermodel.set_annotation_info(annotator_sorted, expertise, lang_datasets)
                sess.run(tf.global_variables_initializer())
                model_state = VariableState(sess, tf.trainable_variables())
                model_wembed_state = VariableState(sess, [v for v in tf.global_variables() if
                                                          'words/_word_embeddings:0' == v.name])
                model_wembed_state.import_variables([config.embeddings[lang]])
                config.lr = lr_unsup
                if args.lowtrainasdev:
                    #note we don't pass development set to unsup training, train set plays the role of dev set here because
                    #of scarsity of annotated data.
                    best_score, best_params = nermodel.train_unsup(lrdataset['train'], lrdataset['train'], sess, model_state, verbose=args.verbose)
                else:
                    dev_annotator_sorted = [ann.replace(partition, 'dev') for ann in annotator_sorted]
                    total_dev_samples = 1000
                    num_dev_samples = np.random.multinomial(total_dev_samples, expertise)
                    #now sample from annotations based on num_samples
                    dev_samples = []
                    for i, n in enumerate(np.nditer(num_dev_samples)):
                        n = int(n)
                        if n == 0:
                            continue
                    dev_annotator = dev_annotator_sorted[i]
                    samples = lang_datasets[dev_annotator].sample(n)
                    dev_samples.extend(samples)
                    best_score, best_params = nermodel.train_unsup(lrdataset['train'], dev_samples, sess, model_state, verbose=args.verbose)
                model_state.import_variables(best_params)
                msgtest = nermodel.evaluate(lrdataset['test'], sess)
                test_acc = msgtest['acc']
                test_f1 = msgtest['f1']
                logging.info('test lang:{} acc:{} f1:{}'.format(lang, test_acc, test_f1))
                msgdev = nermodel.evaluate(lrdataset['dev'], sess, isDev=False)
                acc = msgdev['acc']
                f1 = msgdev['f1']
                score = {'dev': msgdev, 'test': msgtest}
                logging.info('dev lang:{} acc:{} f1:{}'.format(lang, acc, f1))
                unsup_lang_scores[lang] = {'acc': test_acc, 'f1': test_f1}
                #after training on data labelled by other languages, train on little supervision from target language
                train_on_gold = True
                if train_on_gold:
                    #now train on gold train after training on bad annotators
                    config.lr = lr_sup
                    #could we not use dev data here for early stopping?
                    #e.g. only run for 5 epochs and set dev to None instead of lrdataset['dev']
                    if args.unsupsupnepochs:
                        logging.info(f'no dev for early stopping, stop after {args.unsupsupnepochs} iters.')
                        nermodel.config.nepochs = args.unsupsupnepochs
                        best_score, best_params = nermodel.train(train=lrdataset['train'], dev=None, session=sess,
                                                              model_state=model_state, verbose=args.verbose)
                        nermodel.config.nepochs = nepochs_backup
                    else:
                        #use devset and early stopping
                        best_score, best_params = nermodel.train(train=lrdataset['train'], dev=lrdataset['dev'], session=sess,
                                                                 model_state=model_state, verbose=args.verbose)
                    model_state.import_variables(best_params)
                    msgtest = nermodel.evaluate(lrdataset['test'], sess)
                    test_acc = msgtest['acc']
                    test_f1 = msgtest['f1']
                    logging.info('test lang:{} acc:{} f1:{}'.format(lang, test_acc, test_f1))
                    msgdev = nermodel.evaluate(lrdataset['dev'], sess, isDev=False)
                    acc = msgdev['acc']
                    f1 = msgdev['f1']
                    score = {'dev': msgdev, 'test': msgtest}
                    logging.info('dev lang:{} acc:{} f1:{}'.format(lang, acc, f1))
                lang_scores[lang] = {'acc': test_acc, 'f1': test_f1}
                logging.info('results after unsup pretrain')
                logging.info(f'unsup{args.seed}={unsup_lang_scores}')
                for lang, score in unsup_lang_scores.items():
                    logging.info('{}\t{}\t{}'.format(lang, int(score['acc']), int(score['f1'])))

                logging.info('fine-tune results after unsup pretrain')
                logging.info(lang_scores)
                for lang, score in lang_scores.items():
                    logging.info('{}\t{}\t{}'.format(lang, int(score['acc']), int(score['f1'])))
                scores[lang] = {'dev': msgdev, 'test': msgtest}

                if writer:
                    summary = sess.run(merged,
                                       feed_dict={dev_f1_ph: f1, dev_acc_ph: acc, test_acc_ph: test_acc,
                                                  test_f1_ph: test_f1})
                    writer.add_summary(summary)

    logging.info(f'sup{args.seed}={lang_scores}')
    return scores








def set_config(args, config, lang):
    lang_lr = {}
    lang_batch = {}
    #todo set best config for each lang and model
    config.lr = 0.01
    #target {'uk': 61.39817629179332, 'et': 53.639846743295024, 'kk': 62.06896551724138, 'gl': 57.391304347826086}) {'uk': '2 0.01', 'et': '1 0.01', 'kk': '1 0.01', 'gl': '2 0.01'}
    single_hp = {'uk': '2 0.01', 'et': '1 0.01', 'kk': '1 0.01', 'gl': '2 0.01'}
    if args.model == 'single':
        bs, lr = single_hp[lang].split()
        bs, lr = int(bs), float(lr)
        config.lr = lr
        config.batch_size = bs






def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',default='single', help='single, joint, mixture, averageparam')
    parser.add_argument('-s', '--save', default=0, type=int, help='0 not save, 1 save')
    parser.add_argument('-t', '--tune', default=0, type=int, help='0 not tune, 1 tune')
    parser.add_argument('-indhigh', default=0, type=int, help='0 individual model lowresource else highresource')
    parser.add_argument('-lr', default=0.0, type=float, help='learning rate')
    parser.add_argument('-drop', default=0.0, type=float, help='dropout')
    parser.add_argument('-devdrop', default=0.0, type=float, help='dropout for development 1.0 means no dropout')
    parser.add_argument('-batch', default=0, type=int, help='batch szie')
    parser.add_argument('-batchtest', default=0, type=int, help='test batch szie')
    parser.add_argument('-seed', default=7, type=int, help='random seed')
    parser.add_argument('-jointonce', default=0, type=int, help='0=joint train for each lang, 1=jointonce')
    parser.add_argument('-indvar', default=1, type=int, help='0 general mixture in average model, else per variable type')
    parser.add_argument('-attch', default=0, type=int, help='0 not attend chars in attentional models else attend char')
    parser.add_argument('-attw', default=1, type=int, help='0 not attend words in attentional models else attend word')
    parser.add_argument('-verbose', default=0, type=int, help='0 not verbose, 1 verbose')
    parser.add_argument('-self', default=1, type=int, help='0 do not add target model to hres langs, 1 add it (e.g. for mixture model)')
    parser.add_argument('-unsupsample', default=10000, type=int, help='the number of unlabelled samples for unsupervised model')
    parser.add_argument('-unsuptopk', default=0, type=int, help='the number of top annotators')
    parser.add_argument('-unsupgold', default=-1, type=int, help='the number of gold samples in each epoch')
    parser.add_argument('-unsupepochs', default=0, type=int, help='the number of unsup epochs')
    parser.add_argument('-unsuplr', default=0, type=float, help='lr for unsup multiannotator training')
    parser.add_argument('-unsupsuplr', default=0, type=float, help='lr for sup multiannotator training')
    parser.add_argument('-exportannot', default='', type=str, help='if not empty dump uagg annotations in this directory.')
    parser.add_argument('-unsupsupnepochs', default=0, type=int, help='if > 0, do not use early stopping by dev set, stop after unsupsupnepochs iters')
    parser.add_argument('-uniformexpertise', default=0, type=int, help='if >0, use uniform expertise instead of F1')
    parser.add_argument('-testannotations', default=0, type=int, help='if >0, use the multiannotated test set instead of train set in the multiannotator/sagg model.')
    parser.add_argument('-countunks', default=0, type=int, help='if >0, count unks and exit!')
    parser.add_argument('-lowtrainasdev', default=0, type=int, help='if >0, use 100 lowres training as dev set for multiannotator unsupervised phase, otherwise use tagged dev data.')
    parser.add_argument('-hlangs', nargs='*', default=[], help="high resource language codes, uses config if empty")
    parser.add_argument('-llangs', nargs='*', default=[], help="low resource language codes, uses config if empty")
    parser.add_argument('-dir_output', type=str, required=True, help='dir to dump output to.')
    parser.add_argument('-dir_input', type=str, required=True, help='directory to read words, tags, trimmed embs from.')
    parser.add_argument('-dir_ner', type=str, required=True, help='directory to read NER datasets from.')
    args = parser.parse_args(argv)
    #run high resource CUDA_VISIBLE_DEVICES=0 python mmain.py -m single -v 1 -s 1 -indhigh 1
    #run low resource CUDA_VISIBLE_DEVICES=0 python mmain.py -m single -v 1 -s 1 -indhigh 0 -lr 0.01 -batch 1
    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    logging.info(args)

    if args.hlangs:
        Config.highres_langs = args.hlangs
    if args.llangs:
        Config.lowres_langs = args.llangs
    if args.lr:
        Config.lr = args.lr
    if args.batch:
        Config.batch_size = args.batch
    if args.batchtest:
        Config.test_batch_size = args.batchtest
    if args.drop:
        Config.dropout = args.drop
    if args.devdrop:
        Config.dev_dropout = args.devdrop
    if args.unsupgold >= 0:
        Config.num_unsup_gold = args.unsupgold
    if args.unsupepochs:
        Config.num_unsup_epochs = args.unsupepochs
    if args.unsuplr:
        Config.unsuplr = args.unsuplr
    if args.unsupsuplr:
        Config.unsupsuplr = args.unsupsuplr
    if args.unsuptopk:
        Config.unsuptopk = args.unsuptopk

    config = Config(args, load=False if args.model in ['multiannotator'] else True)
    dump(config)

    if args.model not in ['multiannotator']:
        load_lowres_datasets()
        if args.model in ['single', 'langtaglang']:
            load_highres_datasets()
            if args.countunks:
                logging.info('Counting unks in NER datasets...')
                count_unks(highres_datasets, config)
                sys.exit(0)



        all_datasets  = {**highres_datasets, **lowres_datasets}

    set_seed(args.seed)


    model_names = {'single': individual_model, 'langtaglang': lang_tag_lang,
                   'multiannotator':multiannotator, 'uaggexport': export_annotation_to_uagg}

    model = model_names[args.model]



    writer = tf.summary.FileWriter(os.path.join(log_dir, args.model))
    model(args, writer)
