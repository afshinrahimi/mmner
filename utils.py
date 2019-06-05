from io import open
import logging
import pdb
import random
import os
import numpy as np
import shutil
from collections import Counter
import tarfile
import json
import random
from collections import defaultdict
import io
import gzip
import sys
import argparse

np.random.seed(7)
random.seed(7)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def check_num_records(filename):
    records = 0
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip() == '':
                records += 1
    logging.info('{} records in {}'.format(records, filename))
    return records

def check_num_records_by_content(filehandle, filename):
    records = 0
    for line in filehandle:
        line = line.decode('utf-8')
        if line.strip() == '':
            records += 1
    logging.info('{} records in {}'.format(records, filename))
    return records

def extract_records(input_file, output_file, num_records):
    records = 0
    all_records = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        record = []
        for line in fin:
            line = line.strip()
            if 'docstart' in line.lower():
                continue
            if line == '' and len(record) > 0:
                all_records.append(record)
                record = []
            elif line != '':
                record.append(line)
        all_records.append(record)
    selected_records = all_records[0:num_records]

    with open(output_file, 'w', encoding='utf-8') as fout:
        for i, record in enumerate(selected_records):
            for r in record:
                fout.write(r + '\n')
            fout.write('\n')

def create_train_dev_test_from_one_file(conll_file, lang, output_dir):
    records = 0
    all_records = []
    with open(conll_file, 'r', encoding='utf-8') as fin:
        record = []
        for line in fin:
            line = line.strip()
            if 'docstart' in line.lower():
                continue
            if line == '' and len(record) > 0:
                all_records.append(record)
                record = []
            elif line != '':
                fields = line.split()
                if len(fields) > 1:
                    line = fields[0] + '\t' + fields[-1]
                    record.append(line)
        if len(record) > 0:
            all_records.append(record)
    logging.info('{} sentences for {} detected.'.format(len(all_records), lang))
    train_100 = all_records[0:100]
    train_1000 = all_records[0:1000]
    train_10000 = all_records[0:10000]
    test_count = dev_count = min(len(all_records) - 10000 // 2, 10000)

    dev = all_records[10000:10000 + dev_count]
    test = all_records[10000 + dev_count: 10000 + dev_count + test_count]
    train_100_file = os.path.join(output_dir, '{}.train.{}'.format(lang, 100))
    train_1000_file = os.path.join(output_dir, '{}.train.{}'.format(lang, 1000))
    train_10000_file = os.path.join(output_dir, '{}.train.{}'.format(lang, 10000))
    dev_file = os.path.join(output_dir, '{}.dev'.format(lang))
    test_file = os.path.join(output_dir, '{}.test'.format(lang))
    file_records = [(train_100, train_100_file), (train_1000, train_1000_file), (train_10000, train_10000_file), (dev, dev_file), (test, test_file)]
    for file_record in file_records:
        with open(file_record[1], 'w', encoding='utf-8') as fout:
            for i, record in enumerate(file_record[0]):
                for r in record:
                    fout.write(r + '\n')
                fout.write('\n')
    logging.info('created train dev test splits for {} in {}'.format(lang, output_dir))



def add_lang_to_each_word(dir_name, name_index, lang_separator=':', embfile=None):
    if embfile:
        files = [embfile]
    else:
        files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    for file in files:
        if not embfile:
            if file[-5:] == 'multi' or 'cca-59' in str(file):
                continue

        lang = file.split('.')[name_index]
        print('lang:{}'.format(lang))
        filename = os.path.join(dir_name, file)
        lines = []
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.strip() != '':
                    line = lang + lang_separator + line
                lines.append(line)
        with open(filename + '.multi', 'w', encoding='utf-8') as fout:
            for line in lines:
                fout.write(line)



def create_ner_training_datasets():
    for lang in ['de', 'en', 'nl']:
        for i in [100, 1000, 10000]:
            extract_records(input_file='./datasets/ner/{}.train'.format(lang), output_file='./datasets/ner/{}.train.{}'.format(lang, i), num_records=i)
            check_num_records('./datasets/ner/{}.train.{}'.format(lang, i))

def collect_vocab_and_tags(dir_name, fname=None):
    '''
    :param dir_name:
    :return set of all the vocab (+$UNK$ and +$NUM$ in all the files in the directory (expect all the files to be in conll2003 format):
    '''
    tags = set()
    vocab = set()
    chars = set()

    files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))] if not fname else [os.path.join(dir_name, fname)]
    for file in files:
        #we just want the files where language id is added to the beginning of the words
        if file[-5:] != 'multi':
            continue
        filename = os.path.join(dir_name, file)
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line.strip() != '' and 'docstart' not in line.lower():
                    fields = line.split()
                    if len(fields) > 1:
                        word, tag = fields[0], fields[-1]
                    else:
                        logging.info('warning: error in line {} in file {}'.format(line, filename))
                    vocab.add(word.lower())
                    tags.add(tag)
                    chars.update(word)
    return vocab, tags, chars


def collect_vocab_and_tags_panx(panx_dir_name, fname=None):
    '''
    :param dir_name: contains tar.gz files each with train test dev conll files
    :return set of all the vocab (+$UNK$ and +$NUM$ in all the files in the directory (expect all the files to be in conll2003 format):
    important note: all vocab in wikiembs :https://fasttext.cc/docs/en/pretrained-vectors.html are lowercase while
    words in wikiemb+crawlemb: https://fasttext.cc/docs/en/crawl-vectors.html have upper case words so it will make a huge
    difference in terms of OOVs and how we match words in NER datasets to words in the pre-trained word embeddings.
    '''
    tags = set()
    vocab = set()
    chars = set()

    files = [f for f in os.listdir(panx_dir_name) if os.path.isfile(os.path.join(panx_dir_name, f))] if not fname else [fname]
    for file in files:
        targz_file = os.path.join(panx_dir_name, file)
        tar = tarfile.open(targz_file, "r:gz")
        for member in tar.getmembers():
            #don't collect the vocabulary of extra annotated data
            if member.name == 'extra':
                continue
            bio_file = tar.extractfile(member)
            for line in bio_file:
                line = line.decode('utf-8')
                if line.strip() != '' and 'docstart' not in line.lower():
                    fields = line.split()
                    if len(fields) > 1:
                        word, tag = fields[0], fields[-1]
                    else:
                        logging.info('warning: error in line {} in file {}'.format(line, targz_file))
                    vocab.add(word)
                    tags.add(tag)
                    chars.update(word)
        tar.close()
    return vocab, tags, chars

def collect_embedding_vocabs(dir_name, embfile=None):
    '''

    :param dir_name: directory where all the word embeddings with different languages are located
    expected that lang id- (e.g. en-) is added to the beginning of each word so that exact words in different
    languages are distinguishable
    :return: set of vocab
    '''
    vocab_emb = set()
    filename = os.path.join(dir_name, emb_file)

    with gzip.open(filename, 'rt', encoding='utf-8') if 'gz' in embfile else open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip() != '':
                word = line.split(' ')[0]
                vocab_emb.add(word)
    vocab_lang_distribution = [v.split(':')[0] for v in vocab_emb]
    vocab_lang_distribution = Counter(vocab_lang_distribution)
    logging.info('embedding file vocab stats: {}'.format(str(vocab_lang_distribution)))
    return vocab_emb

def write_list(items, output_file):
    with open(output_file, 'w', encoding='utf-8') as fout:
        for i, item in enumerate(sorted(items)):
            if i != len(items) - 1:
                fout.write('{}\n'.format(item))
            else:
                fout.write(item)
def read_list(filename):
    items = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for item in fin:
            items.append(item.strip())
    return items


def write_vocab_tags_chars_embs(vocab_ner, tags, chars, vocab_emb, output_dir):
    write_list(tags, os.path.join(output_dir, 'tags.txt'))
    write_list(chars, os.path.join(output_dir, 'chars.txt'))

    '''
    now we should add any vocab that might have uppercase in vocab_ner
    but is lowercase in vocab_emmb.
    note that lowercase=True should be turned to lowercase=False in mconfig file to reflect this.
    
    question: what about the cases when a lower case word is in NER but an uppercase is in emb?
    
    '''

    vocab = vocab_ner & vocab_emb
    for v in vocab_ner:
        vlow = v.lower()
        if v == vlow:
            continue
        if v not in vocab_emb and vlow in vocab_emb:
            vocab.add(vlow)

    vocab_lang_distribution = [v.split(':')[0] for v in vocab]
    vocab_lang_distribution = Counter(vocab_lang_distribution)
    logging.info(vocab_lang_distribution)
    with open(os.path.join(output_dir, 'stats'), 'w') as fout:
        fout.write(str(vocab_lang_distribution))

    vocab.add('$UNK$')
    vocab.add('$NUM$')
    write_list(vocab, os.path.join(output_dir, 'words.txt'))

def build_vocab(conll_dir, emb_dir, output_dir, emb_file=None, panx=False, fname=None):
    if panx:
        vocab, tags, chars = collect_vocab_and_tags_panx(conll_dir, fname=fname)
    else:
        vocab, tags, chars = collect_vocab_and_tags(conll_dir)
    vocab_emb = collect_embedding_vocabs(emb_dir, emb_file)
    write_vocab_tags_chars_embs(vocab, tags, chars, vocab_emb, output_dir)
    return vocab, tags, chars

def trim_embs(emb_dir, vocab_file, output_dir, dim, emb_file=None):
    vocab = read_list(vocab_file)
    vocabset = set(vocab)
    vocab_emb = set()
    embeddings = np.zeros([len(vocab), dim])

    filename = os.path.join(emb_dir, emb_file)
    with gzip.open(filename, 'rt', encoding='utf-8') if 'gz' in emb_file else open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split(' ')
            word = line[0]
            if word not in vocabset:
                continue

            embedding = [float(x) for x in line[1:]]
            word_idx = vocab.index(word)
            embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(os.path.join(output_dir, 'trimmed_embs.npz'), embeddings=embeddings)

def data_to_byteio(records, name):
    data = ''
    for i, record in enumerate(records):
        for r in record:
            data += r + '\n'
        data += '\n'
    encoded_data = data.encode('utf8')
    data_byteio = io.BytesIO(encoded_data)
    tarinfo = tarfile.TarInfo(name)
    tarinfo.size = len(encoded_data)
    return data_byteio, tarinfo

def annotation_to_byteio(records, name):
    data = ''
    for i, record in enumerate(records):
            data += '\n'.join([str(c) for c in record]) + '\n\n'

    encoded_data = data.encode('utf8')
    data_byteio = io.BytesIO(encoded_data)
    tarinfo = tarfile.TarInfo(name)
    tarinfo.size = len(encoded_data)
    return data_byteio, tarinfo

def annotation_to_byteio_yuan(records, name):
    data = ''
    for i, record in enumerate(records):
            data += '\t'.join([str(c) for c in record]) + '\n'

    encoded_data = data.encode('utf8')
    data_byteio = io.BytesIO(encoded_data)
    tarinfo = tarfile.TarInfo(name)
    tarinfo.size = len(encoded_data)
    return data_byteio, tarinfo

def write_panx_preprocessed(filehandle, filename, output_dir):
    langid = filename.split('.')[0]
    records = 0
    all_records = []
    record = []
    tag_records = defaultdict(set)
    last_label_in_record = None
    for line in filehandle:
        line = line.decode('utf-8')
        line = line.strip()
        if 'docstart' in line.lower():
            continue
        if line == '' and len(record) > 0:
            all_records.append(record)
            if last_label_in_record:
                tag_records[last_label_in_record].add(len(all_records) - 1)
            last_label_in_record = None
            record = []
        elif line != '':
            fields = line.split()
            if len(fields) > 1:
                line = langid + ':' + fields[0] + '\t' + fields[-1]
                if 'B-' in fields[-1]:
                    last_label_in_record = fields[-1]
                record.append(line)
    if len(record) > 0:
        all_records.append(record)

    #find the minimum count of B- tag, used for stratified sampling
    min_count_tag = min([len(v) for v in tag_records.values()])
    #now sample min_count_tag records from each tag
    new_records = []
    for _, records in tag_records.items():
        new_records.extend(random.sample(records, min_count_tag))
    #new records contains indices of items in all_records
    random.seed(0)
    np.random.seed(0)
    random.shuffle(new_records)
    all_records = [all_records[i] for i in new_records]


    total_records = len(all_records)
    if total_records > 30000:
        num_recs = 10000
    elif total_records > 3000:
        num_recs = 1000
    elif total_records > 300:
        num_recs = 100
    else:
        return

    num_train = min((total_records - 2 * num_recs) - (total_records - 2 * num_recs) % 1000, 20000)
    num_train = max((num_train // 5000) * 5000, num_recs)
    train_set = all_records[0:num_train]
    dev_set = all_records[num_train:num_train + num_recs]
    test_set = all_records[num_train + num_recs:num_train + 2 * num_recs]
    extra_set = all_records[num_train + 2 * num_recs:]
    print(langid, 'train', num_train, 'dev/test', num_recs)
    tar = tarfile.open(os.path.join(output_dir, filename), "w:gz")
    train_data, train_tarinfo = data_to_byteio(train_set, 'train')
    dev_data, dev_tarinfo = data_to_byteio(dev_set, 'dev')
    test_data, test_tarinfo = data_to_byteio(test_set, 'test')
    extra_data, extra_tarinfo = data_to_byteio(extra_set, 'extra')
    tar.addfile(tarinfo=train_tarinfo, fileobj=train_data)
    tar.addfile(tarinfo=dev_tarinfo, fileobj=dev_data)
    tar.addfile(tarinfo=test_tarinfo, fileobj=test_data)
    tar.addfile(tarinfo=extra_tarinfo, fileobj=extra_data)
    tar.close()

def get_cca59_languages(cca_59_file):
    lang_count = Counter()
    with gzip.open(cca_59_file, 'rt', encoding='utf-8') if 'gz' in cca_59_file else io.open(cca_59_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            lang = line.split(':')[0]
            lang_count[lang] += 1
    return lang_count

def panx_to_dataset(input_dir, output_dir, supported_langs=None, count=False):
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if count:
        file_record = {}
        for file in files:
            targz_file = os.path.join(input_dir, file)
            tar = tarfile.open(targz_file, "r:gz")
            for member in tar.getmembers():
                if '.bio' in member.name:
                    bio_file = tar.extractfile(member)
                    if bio_file is not None:
                        num_records = check_num_records_by_content(bio_file, file)
                        file_record[file] = num_records
        file_record = Counter(file_record)
        print(file_record.most_common())
        with open(os.path.join(output_dir, 'lang_stats.json'), 'w') as fout:
            json.dump(file_record, fout)
    for file in files:
        langid = file.split('.')[0]
        if supported_langs and langid not in supported_langs:
            continue
        targz_file_in = os.path.join(input_dir, file)
        targz_file_out = os.path.join(output_dir, file)
        tar = tarfile.open(targz_file_in, "r:gz")
        for member in tar.getmembers():
            if '.bio' in member.name:
                bio_file = tar.extractfile(member)
                if bio_file is not None:
                    write_panx_preprocessed(bio_file, file, output_dir)
        tar.close()

def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--embs_dir', type=str, required=True, help='directory containing the complete word embeddings')
    parser.add_argument('--ner_built_dir', type=str, required=True, help='directory where built output (trimmed embs + vocab) will be written into.')
    args = parser.parse_args(argv)
    return args



if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    logging.info(args)

    #facebook or allenai multilingual embeddings?
    embs_dir = args.embs_dir
    ner_built_dir = args.ner_built_dir
    #raw ner downloaded from panx
    wiki_ner_input_dir = './datasets/panx2_all/'
    #directory for converted raw ner data to train/test/dev stratified
    wiki_ner_output_dir = './datasets/panx_datasets' #'./datasets/panx2_supported45'

    wiki_embsupported_languages = ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et',
                                   'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'lt', 'lv', 'mk', 'ms', 'nl',
                                   'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sv', 'ta', 'tl',
                                   'tr', 'uk', 'vi']
    print(f"num langs: {len(wiki_embsupported_languages)}")


    create_ner_datasets = False
    if create_ner_datasets:
        #only run this once to create stratified train/dev/test splits from wikiann, and save them.
        panx_to_dataset(wiki_ner_input_dir, wiki_ner_output_dir, supported_langs=None)#wiki_embsupported_languages)
        sys.exit(0)
    all_chars = set()
    if not os.path.exists(ner_built_dir):
        os.mkdir(ner_built_dir)
    for lang in wiki_embsupported_languages:
        output_dir = os.path.join(ner_built_dir , f'builtdata_{lang}')
        emb_file = lang + '.multi.gz'
    #create_train_dev_test_from_one_file(conll_file='./datasets/originals/ner/de/wikiann-de.bio', lang='de', output_dir=ner_dir)
    #create_train_dev_test_from_one_file(conll_file='./datasets/originals/ner/en/wikiann-en.bio', lang='en',  output_dir=ner_dir)
    #create_train_dev_test_from_one_file(conll_file='./datasets/originals/ner/nl/wikiann-nl.bio', lang='nl',  output_dir=ner_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"directory exists:{output_dir}")
        os.mkdir(output_dir)
    #for panx langid is already added to conll terms
    #add_lang_to_each_word(ner_dir, name_index=0, lang_separator=':')

        #embeddings already have lang identifier
        vocab, tags, chars = build_vocab(wiki_ner_output_dir, os.path.join(embs_dir, lang), output_dir, emb_file=emb_file, panx=True, fname='{}.tar.gz'.format(lang))
        all_chars = all_chars | chars
        trim_embs(os.path.join(embs_dir, lang), os.path.join(output_dir, 'words.txt'), output_dir, 300, emb_file=emb_file)
    write_list(all_chars, os.path.join(ner_built_dir, 'chars.txt'))
    write_list(tags, os.path.join(ner_built_dir, 'tags.txt'))

