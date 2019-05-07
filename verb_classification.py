# https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md#training-a-text-classification-model

from flair.data import TaggedCorpus, Sentence
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


# The line should have the following format:
# __label__<class_name> <text>
# If you have a multi class task, you can have as many labels as you want at the beginning of the line, e.g.,
# __label__<class_name_1> __label__<class_name_2> <text>
# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_classification_corpus("path",
                                                                     "train",
                                                                     "test",
                                                                     "dev",
                                                                     use_tokenizer=False)

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [
    WordEmbeddings('/resources/public/stoeckel/emb/Leipzig40MT.gensim'),
    FlairEmbeddings('german-forward'),
    FlairEmbeddings('german-backward'),
]

# 4. initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                                   hidden_size=512,
                                                                   reproject_words=True,
                                                                   reproject_words_dimension=256,
                                                                   )

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('resources/taggers/verb_sense',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter

plotter = Plotter()
plotter.plot_training_curves('resources/taggers/verb_sense/loss.tsv')
plotter.plot_weights('resources/taggers/verb_sense/weights.txt')

classifier = TextClassifier.load_from_file('resources/taggers/verb_sense/final-model.pt')

# create example sentence
sentence = Sentence('Es läuft heute sehr gut !')

# predict tags and print
classifier.predict(sentence)

print(sentence.labels)
