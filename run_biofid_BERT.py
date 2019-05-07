from typing import List

import torch

from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, BertEmbeddings

# print(torch.version.cuda)
# print("CUDA:", torch.cuda.current_device())
device = torch.device('cuda:1')

corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(
    'resources/corpora/biofid_tax_28.03/',
    {0: 'text', 1: 'pos', 2: 'lemma', 3: 'ner'},
    train_file='train.biofid.conll',
    test_file='test.biofid.conll',
    dev_file='dev.biofid.conll',
    max_sequence_length=512)

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # GloVe embeddings
    # WordEmbeddings('de'),
    # WordEmbeddings('/resources/public/stoeckel/emb/Leipzig40MT.gensim'),

    # ELMoEmbeddings(options_file='/resources/public/stoeckel/emb/ELMo_Leipzig100k/options_post.json',
    #                weight_file='/resources/public/stoeckel/emb/ELMo_Leipzig100k/ELMo_Leipzig100k.hdf5'),

    BertEmbeddings('bert-base-multilingual-cased'),

    # contextual string embeddings, forward
    PooledFlairEmbeddings('german-forward', pooling='min'),

    # contextual string embeddings, backward
    PooledFlairEmbeddings('german-backward', pooling='min'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/biofid-BERT-tax',
              max_epochs=150)
