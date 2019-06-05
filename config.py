import numpy as np
import os
import random
import logging
import tarfile
import pdb
#from utils import collect_vocab_and_tags_panx, collect_embedding_vocabs, write_vocab_tags_chars_embs, trim_embs
from collections import defaultdict
# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)



class Config():
    def __init__(self, args, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.dir_output = args.dir_output
        self.dir_input = args.dir_input
        self.dir_wikiann = args.dir_ner
        self.dir_model = os.path.join(self.dir_output, "weights")
        self.dir_model_highres = os.path.join(self.dir_output, 'weights')
        self.dir_unlabeled = os.path.join(self.dir_output, 'multiannotations')
        self.dir_bccannotations = os.path.join(self.dir_output, 'bccannotations')
        self.langtaglang_file = os.path.join(self.dir_output, 'langtaglang.json')

        self.params_dir = self.dir_model
        self.filename_words = {}
        for lang in set(self.highres_langs + self.lowres_langs):
            self.filename_words[lang] = os.path.join(self.dir_input, f'builtdata_{lang}/words.txt')

        self.filename_tags = os.path.join(self.dir_input, 'tags.txt')
        self.filename_chars = os.path.join(self.dir_input, 'chars.txt')
        self.path_log = os.path.join(self.dir_output, "log.txt")
        self.filename_trimmed = {}
        for lang in set(self.highres_langs + self.lowres_langs):
            self.filename_trimmed[lang] = os.path.join(self.dir_input, f'builtdata_{lang}/trimmed_embs.npz')

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.mkdir(self.dir_output)
            os.mkdir(self.dir_model)
            os.mkdir(self.dir_unlabeled)
            os.mkdir(self.dir_bccannotations)



        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            for lang in set(self.highres_langs + self.lowres_langs):
                self.load(lang)


    def load(self, lang):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        logging.info('loading {}'.format(lang))
        # 1. vocabulary
        self.vocab_words[lang] = load_vocab(self.filename_words[lang])
        for lang, vocab_word in self.vocab_words.items():
            self.word_vocab[lang] = {v:k for k, v in vocab_word.items()}
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        self.tag_vocab = {v:k for k, v in self.vocab_tags.items()}
        self.nwords[lang]     = len(self.vocab_words[lang])
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word[lang] = get_processing_word(self.vocab_words[lang],
                self.vocab_chars, lowercase=False, chars=self.use_chars, trim_lang=True, lowerchars=True if self.lowres_langs==['de'] else False)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False, trim_lang=False)
            
        # 3. get pre-trained embeddings
        self.embeddings[lang] = (get_trimmed_glove_vectors(self.filename_trimmed[lang], self.capacity)
                if self.use_pretrained else None)
        #logging.info('loaded {} with vocab size {} and embedding size {}'.format(lang, self.nwords[lang], self.embeddings[lang].shape[0]))


    # general config
    highres_langs = ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et',
                                   'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'lt', 'lv', 'mk', 'ms', 'nl',
                                   'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sv', 'ta', 'tl',
                                   'tr', 'uk', 'vi']

    #highres_langs = ['en']
    #highres_langs = highres_langs[20:25]
    lowres_langs = ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et',
                                   'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'lt', 'lv', 'mk', 'ms', 'nl',
                                   'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sv', 'ta', 'tl',
                                   'tr', 'uk', 'vi']


    #filenames





    #lowres_langs = ['en']

    lang_script = {}
    # vocab (created from dataset with build_data.py)
    #filename_words = "./datasets/builtdata_cmu8/words.txt"
    vocab_words = {}
    word_vocab = {}
    nwords = {}
    embeddings = {}
    processing_word = {}
    capacity = 80000
    num_unsup_iter = 1000
    num_unsup_gold = 0
    num_unsup_epochs = 1000
    unsuplr = 5e-3
    unsupsuplr = 5e-4
    unsuptopk = 10
    # glove files
    #filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    #filename_trimmed = "./datasets/builtdata_cmu8/trimmed_embs_cmu.npz"
    use_pretrained = True


    # embeddings
    dim_word = 300
    dim_char = 100


    # dataset
    #filename_dev = "datasets/ner/uk.dev.multi"
    #filename_test = "datasets/ner/uk.test.multi"
    #filename_train = "datasets/ner/uk.train.100.multi"

    #filename_dev = filename_test = filename_train = "ner/data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    #nepoch for highres 20 for lowres 100
    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.5
    batch_size       = 1
    test_batch_size  = 100
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3
    lowres_nepoch_no_imprv = 5
    dev_dropout = 1.0

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU

    entropy_loss_weight = 0

def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True, trim_lang=False, lowerchars=False):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word if (not trim_lang or word in ['$NUM$', '$UNK$']) else word[3:].lower() if lowerchars else word[3:]:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            elif word.lower() in vocab_words:
                word = vocab_words[word.lower()]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct word missing: {}".format(word))

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f

def count_unks(datasets, config):
    print("lang sents words ne unk nerunk")
    for lang in datasets:
        sentences = datasets[lang]['train'].data
        unkid = config.vocab_words[lang][UNK]
        total_unk_count = 0
        total_word_count = 0
        total_ounk_count = 0
        total_ne = 0
        for sentid, sentencelabel in enumerate(sentences):
            sentence, labelids = sentencelabel
            vocabids = [word[1] for word in sentence]
            unk_indices = [i for i in range(len(vocabids)) if vocabids[i] == unkid]
            unk_labels = [labelids[i] for i in unk_indices]
            o_unks = [1 for l in unk_labels if l==5]
            num_ne = sum([1 for l in labelids if l!=5])
            num_o_unks = sum(o_unks)
            total_ounk_count += num_o_unks
            unk_count = sum([1 if vid==unkid else 0 for vid in vocabids])
            word_count = len(vocabids)
            total_unk_count += unk_count
            total_word_count += word_count
            total_ne += num_ne
        total_NER_unk_count = total_unk_count - total_ounk_count
        num_sentences = sentid
        print(f"{lang} {num_sentences+1} {total_word_count} {total_ne} {total_unk_count} {total_NER_unk_count}")



def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d

class CoNLLDatasetList(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
        self.data = []
        self.sentences = []
        self.nwords = 0
        self.nunks = 0
        self.loaddata()


    def __iter__(self):
        for i, item in enumerate(self.data):
            if self.max_iter and i > self.max_iter:
                return
            yield item[0], item[1]



    def loaddata(self):
        niter = 0
        if type(self.filename) == str:
            f = open(self.filename, 'r', encoding='utf-8')
        else:
            #an open file is passed on
            f = self.filename
        words, tags = [], []
        sentence = []
        for line in f:
            if type(self.filename) != str:
                line = line.decode('utf-8')
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    niter += 1
                    if self.max_iter is not None and niter > self.max_iter:
                        break
                    self.data.append([words, tags])
                    words, tags = [], []
                    self.sentences.append(sentence)
                    sentence = []
            else:
                #Afshin: change the delimiter from ' ' to default because tabs
                ls = line.split()
                word, tag = ls[0],ls[-1]
                self.nwords += 1
                sentence += [word]
                if self.processing_word is not None:
                    word = self.processing_word(word)
                if self.processing_tag is not None:
                    tag = self.processing_tag(tag)
                words += [word]
                tags += [tag]
        f.close()
    def __len__(self):
        return len(self.data)

    def sample(self, num_instances):
        return random.sample(self.data[0:self.max_iter], k=num_instances)




def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char

def get_trimmed_glove_vectors(filename, capacity=0):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            if capacity:
                assert capacity > data['embeddings'].shape[0], 'capacity less than vocab size'
                logging.info('load embeddings with shape {} from {}'.format(str(data['embeddings'].shape), filename))
                extra_rows = capacity - data['embeddings'].shape[0]
                if extra_rows > 0:
                    extra_embs = np.zeros((extra_rows, data['embeddings'].shape[1]), dtype=np.float32)
                return np.vstack((data['embeddings'], extra_embs))
            else:
                logging.info('load embeddings with shape {} from {}'.format(str(data['embeddings'].shape), filename))
                return data["embeddings"]

    except IOError:
        raise MyIOError(filename)

def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger





def read_dataset_panx_individual(config, langid, max_iter=None):
    """
    :param data_dir: where multilingual conll datasets are located (e.g. french dataset has fr- prefix)
    :param highres_langs: code for high resource languages (e.g. ['en', 'de', 'fr']
    :param lowres_lang: code for one low resource language e.g. 'uk'
    :return: datasets for all languages in a dictionary where key is langid and value is the dataset records
    """
    data_dir = config.dir_wikiann
    targz_file = os.path.join(data_dir, "{}.tar.gz".format(langid))
    tar = tarfile.open(targz_file, "r:gz")
    data_set = {}
    #for lowres_lang we need to load train dev test
    targz_file = os.path.join(data_dir, "{}.tar.gz".format(langid))
    tar = tarfile.open(targz_file, "r:gz")
    for member in tar.getmembers():
        #don't read extra data
        if member.name == 'extra':
            continue
        file_handle = tar.extractfile(member)
        dataset = CoNLLDatasetList(file_handle, config.processing_word[langid],
                         config.processing_tag, max_iter if member.name in ['train', 'dev'] and max_iter else config.max_iter)
        data_set[member.name] = dataset

    return data_set

def read_dataset_panx_multiannotated(config, langid, max_iter=None):
    """
    :param data_dir: where multilingual conll datasets are located (e.g. french dataset has fr- prefix)
    :param highres_langs: code for high resource languages (e.g. ['en', 'de', 'fr']
    :param lowres_lang: code for one low resource language e.g. 'uk'
    :return: datasets for all languages in a dictionary where key is langid and value is the dataset records
    """
    data_dir = config.dir_unlabeled
    targz_file = os.path.join(data_dir, f'{langid}.tar.gz')
    tar = tarfile.open(targz_file, "r:gz")
    data_set = {}
    #for lowres_lang we need to load train dev test
    for member in tar.getmembers():
        logging.info('loading {} tagged by {} max_iter {}'.format(langid, member.name, max_iter))
        file_handle = tar.extractfile(member)
        dataset = CoNLLDatasetList(file_handle, config.processing_word[langid],
                         config.processing_tag, max_iter if max_iter else config.max_iter)
        data_set[member.name] = dataset

    return data_set