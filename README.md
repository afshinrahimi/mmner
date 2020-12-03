# mmner

Massively Multilingual Transfer for NER https://arxiv.org/abs/1902.00193

The paper was accepted as a long paper in ACL2019.

You can download the datasets from from https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN

If you want to automatically download the dataset using wget/curl use this link instead: https://www.dropbox.com/s/12h3qqog6q4bjve/panx_dataset.tar

If you use the datasets please cite:
Pan, Xiaoman, et al. "Cross-lingual name tagging and linking for 282 languages." 
Proceedings of the 55th Annual Meeting of the Association 
for Computational Linguistics (Volume 1: Long Papers). Vol. 1. 2017.


We have partitioned the original datasets into train/test/dev sets for benchmarking our multilingual transfer models:
Rahimi, Afshin, Yuan Li, and Trevor Cohn. "Massively Multilingual Transfer for NER." arXiv preprint arXiv:1902.00193 (2019).
```
@inproceedings{rahimi-etal-2019-massively,
    title = "Massively Multilingual Transfer for {NER}",
    author = "Rahimi, Afshin  and
      Li, Yuan  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1015",
    pages = "151--164",
}
```


#### Download fastext wiki embeddings
```{r, engine='sh', code_block_name}
mkdir ./monolingual

for l in af ar bg bn bs ca cs da de el en es et fa fi fr he hi hr hu id it lt lv mk ms nl no pl pt ro ru sk sl sq sv ta tl tr uk vi
do
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$l.vec -P ./monolingual
done
```

#### Create cross-lingual embeddings by mapping all languages into english space
```{r, engine='sh', code_block_name}
git clone https://github.com/facebookresearch/MUSE.git
cd MUSE
chmod +x ./get_evaluation.sh
./get_evaluation.sh

for l in af am ar bg bn bs ca cs da de el en es et fa fi fr he hi hr hu id it lt lv mk ms nl no pl pt ro ru si sk sl so sq sv sw ta tl tr uk uz vi yo 
do
    python supervised.py --tgt_lang en --src_lang $l --tgt_emb ./monolingual/wiki.en.vec --src_emb data/webcrawltxt/wiki.$l.vec --n_refinement 5 --dico_train identical_char
done
#you can actually only save the mapping, and use it to build the cross-lingual embeddings later.
#the cross-lingual embeddings will be save in MUSE/dumped
```

##### Add langid the the beginning of each line, change filename, and compress files
```{r, engine='sh', code_block_name}
cd dumped
for l in af am ar bg bn bs ca cs da de el en es et fa fi fr he hi hr hu id it lt lv mk ms nl no pl pt ro ru si sk sl so sq sv sw ta tl tr uk uz vi yo 
do
    sed -i -e 's/^/$l:/' $l.txt
    mv $l.txt $l.multi
    gzip $l.multi
done
#now we have files like af.multi.gz where each word begins with langid:, e.g. en:good -0.1 1.2 2.5 ... 
cd ../..
```


##### Download panx_datasets 
from https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN
and save it in current directory so that you have ./panx_datasets/en.tar.gz. the panx_datasets address is hardcoded in utils.py.



##### Create builtdata 
using cross-lingual word embeddings and NER datasets (this will create a term-, char- and tag-vocab for datasets, and limits wordembs to vocab in ner dataset). Set the parameters in the code (which languages to build data for, etc.)
```{r, engine='sh', code_block_name}
python utils.py --embs_dir ./MUSE/dumped --ner_built_dir ./datasets/wiki/identchar
```

##### Run single lowres models
```{r, engine='sh', code_block_name}
python main.py -m single -v 1 -s 0 -batch 1 -lr 0.01 -seed 1 -dir_input datasets/wiki/identchar/ -dir_output experiments/conll/wiki/identchar
```



##### Run single highres model 
Note the indhigh should be 1 and batch size and learning rate are different from lowres settings.
-s 1 saves the individual highres models in a directory, which will later be used for annotating data in other languages (langtaglang).
```{r, engine='sh', code_block_name}

####for wikian datasets
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m single -indhigh 1 -v 1 -s 1 -batch 100 -lr 0.001 -seed 1  -dir_input datasets/wiki/identchar/ -dir_output experiments/wiki/identchar
```


##### Use each highres model (previously saved) to annotate data in another language
(requires hlangs llangs and batchsize=20 for conll datasets)
```{r, engine='sh', code_block_name}
CUDA_VISIBLE_DEVICES=0 python main.py -m langtaglang -dir_input datasets/wiki/identchar/ -dir_output experiments/wiki/identchar
```


##### Export the model predictions for the BEA (=uagg) model 
Should be executed after langtaglang. this doesn't involve any defferential computation, and just changes the format of langtaglang data (it will be saved in dir_output/bccannotations)
```{r, engine='sh', code_block_name}
python main.py  -m uaggexport -dir_input datasets/wiki/identchar/ -dir_output experiments/wiki/identchar
```
##### If you just want to try BEA, instead of the above export, download the exported model predictions from
https://www.amazon.com/clouddrive/share/1qJHGHszDwyQkbKHTA3kxd3oRqMjSj4vdx1YMrCHsa1

##### Run BEA for Wikiann
Example data and code can be found in the '[bea_code](/bea_code)' folder. First run '[create_data_wiki.ipynb](/bea_code/create_data_wiki.ipynb)' to read in raw data (tar files, generated by the previous step) and output csv files. Then run '[run_bea_wiki.ipynb](/bea_code/run_bea_wiki.ipynb)' to obtain results in different settings.

##### Run RaRe for Wikiann
first pretrain on top k (here 10) annotations (sorted by f1 results in taglangtag.json) and then fine-tune on lowres gold data from the target language.
to take an average f1 run the model for n times using n different seeds
```{r, engine='sh', code_block_name}
CUDA_VISIBLE_DEVICES=0 python main.py -m multiannotator -v 1 -unsuptopk 10 -unsupepochs 1000 -unsupgold 0 -unsuplr 5e-3 -unsupsuplr 5e-4 -unsupsupnepochs 5 -seed 1 -dir_input datasets/wiki/identchar/ -dir_output experiments/wiki/identchar
```




#### CoNLL
CoNLL is fairly similar to wikian, only lowercase all characters for German transfer.
Also remove German for transfer to other languages.

##### Run highres models
```{r, engine='sh', code_block_name}
CUDA_VISIBLE_DEVICES=5 nohup python main.py -m single -indhigh 1 -v 1 -s 1 -batch 20 -lr 0.001 -seed 1 -hlangs es en nl de -llangs es en nl de -dir_input datasets/conll/wiki/identchar/ -dir_output experiments/conll/wiki/identcharlower
```

##### Direct transfer (langtaglang)
```{r, engine='sh', code_block_name}
#for german characters should be lowercased both in source and target (an extra if in config.py)
CUDA_VISIBLE_DEVICES=0 python mmain.py -m langtaglang -dir_input datasets/wiki/identchar/ -dir_output experiments/wiki/identchar
```

##### Export data for bea model (bea=uagg)
```{r, engine='sh', code_block_name}
python main.py -m uaggexport -llangs en es de nl -hlangs en es de nl -dir_input datasets/conll/wiki/identchar/ -dir_output experiments/conll/wiki/identcharlower
```

##### Run BEA for CoNLL
Example data and code can be found in the '[bea_code](/bea_code)' folder. First run '[create_data_conll.ipynb](/bea_code/create_data_conll.ipynb)' to read in raw data (tar files, generated by the previous step) and output csv files. Then run '[run_bea_conll.ipynb](/bea_code/run_bea_conll.ipynb)' to obtain results in different settings.

##### Run RaRe for CoNLL
```{r, engine='sh', code_block_name}
python mmain.py -m multiannotator -v 1 -unsuptopk 3 -unsupepochs 1000 -unsupgold 0 -unsuplr 1e-3 -unsupsuplr 5e-4 -unsupsupnepochs 5 -testannotations 0  -s 0   -seed 0
```
