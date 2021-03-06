# Tutorial 4: BERT, ELMo, and Flair Embeddings

Next to standard WordEmbeddings and CharacterEmbeddings, we also provide classes for BERT, ELMo and Flair embeddings. These embeddings enable you to train truly state-of-the-art NLP models.

This tutorial explains how to use these embeddings. We assume that you're familiar with the [base types](/resources/docs/TUTORIAL_1_BASICS.md) of this library as well as [standard word embeddings](/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md), in particular the `StackedEmbeddings` class.

## Embeddings

All word embedding classes inherit from the `TokenEmbeddings` class and implement the `embed()` method which you need to
call to embed your text. This means that for most users of Flair, the complexity of different embeddings remains
hidden behind this interface. Simply instantiate the embedding class you require and call `embed()` to embed your text.

All embeddings produced with our methods are Pytorch vectors, so they can be immediately used for training and
fine-tuning.

## Flair Embeddings

Contextual string embeddings are [powerful embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing)
 that capture latent syntactic-semantic information that goes beyond
standard word embeddings. Key differences are: (1) they are trained without any explicit notion of words and
thus fundamentally model words as sequences of characters. And (2) they are **contextualized** by their
surrounding text, meaning that the *same word will have different embeddings depending on its
contextual use*.

With Flair, you can use these embeddings simply by instantiating the appropriate embedding class, same as standard word embeddings:

```python
from flair.embeddings import FlairEmbeddings

# init embedding
flair_embedding_forward = FlairEmbeddings('news-forward')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
flair_embedding_forward.embed(sentence)
```

You choose which embeddings you load by passing the appropriate string to the constructor of the `FlairEmbeddings` class. 
Currently, the following contextual string embeddings are provided (note: replace '*X*' with either '*forward*' or '*backward*'):
 
| ID | Language | Embedding | 
| -------------     | ------------- | ------------- |
| 'multi-X'    | English, German, French, Italian, Dutch, Polish | Mix of corpora (Web, Wikipedia, Subtitles, News) |
| 'multi-X-fast'    | English, German, French, Italian, Dutch, Polish | Mix of corpora (Web, Wikipedia, Subtitles, News), CPU-friendly |
| 'news-X'    | English | Trained with 1 billion word corpus |
| 'news-X-fast'    | English | Trained with 1 billion word corpus, CPU-friendly |
| 'mix-X'     | English | Trained with mixed corpus (Web, Wikipedia, Subtitles) |
| 'ar-X'     | Arabic | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'bg-X'  | Bulgarian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'bg-X-fast'  | Bulgarian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia or SETimes) |
| 'cs-X'     | Czech | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'cs-v0-X'    | Czech | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version) |
| 'de-X'  | German  | Trained with mixed corpus (Web, Wikipedia, Subtitles) |
| 'de-historic-ha-X'  | German (historical) | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Historical German trained over *Hamburger Anzeiger* |
| 'de-historic-wz-X'  | German (historical) | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Historical German trained over *Wiener Zeitung* |
| 'es-X'    | Spanish | Added by [@iamyihwa](https://github.com/zalandoresearch/flair/issues/80): Trained with Wikipedia |
| 'es-X-fast'    | Spanish | Added by [@iamyihwa](https://github.com/zalandoresearch/flair/issues/80): Trained with Wikipediam CPU-friendly |
| 'eu-X'    | Basque | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'eu-v0-X'    | Basque | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version) |
| 'fa-X'     | Persian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'fi-X'     | Finnish | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'fr-X'    | French | Added by [@mhham](https://github.com/mhham): Trained with French Wikipedia |
| 'he-X'     | Hebrew | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'hi-X'     | Hindi | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'hr-X'     | Croatian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'id-X'     | Indonesian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'it-X'     | Italian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'ja-X'    | Japanese | Added by [@frtacoa](https://github.com/zalandoresearch/flair/issues/527): Trained with 439M words of Japanese Web crawls (2048 hidden states, 2 layers)|
| 'nl-X'     | Dutch | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'nl-v0-X'    | Dutch | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): LM embeddings (earlier version) |
| 'no-X'     | Norwegian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'pl-X'  | Polish  | Added by [@borchmann](https://github.com/applicaai/poleval-2018): Trained with web crawls (Polish part of CommonCrawl) |
| 'pl-opus-X'     | Polish | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'pt-X'    | Portuguese | Added by [@ericlief](https://github.com/ericlief/language_models): LM embeddings |
| 'sl-X'     | Slovenian | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'sl-v0-X'  | Slovenian  | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia and OpenSubtitles2018) |
| 'sv-X'    | Swedish | Added by [@stefan-it](https://github.com/zalandoresearch/flair/issues/614): Trained with Wikipedia/OPUS |
| 'sv-v0-X'    | Swedish | Added by [@stefan-it](https://github.com/stefan-it/flair-lms): Trained with various sources (Europarl, Wikipedia or OpenSubtitles2018) |
| 'pubmed-X'    | English | Added by [@jessepeng](https://github.com/zalandoresearch/flair/pull/519): Trained with 5% of PubMed abstracts until 2015 (1150 hidden states, 3 layers)|


So, if you want to load embeddings from the German forward LM model, instantiate the method as follows:

```python
flair_de_forward = FlairEmbeddings('de-forward')
```

And if you want to load embeddings from the Bulgarian backward LM model, instantiate the method as follows:

```python
flair_bg_backward = FlairEmbeddings('bg-backward')
```

## Recommended Flair Usage

We recommend combining both forward and backward Flair embeddings. Depending on the task, we also recommend adding standard word embeddings into the mix. So, our recommended `StackedEmbedding` for most English tasks is: 


```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'), 
                                        FlairEmbeddings('news-forward'), 
                                        FlairEmbeddings('news-backward'),
                                       ])
```

That's it! Now just use this embedding like all the other embeddings, i.e. call the `embed()` method over your sentences.

```python
sentence = Sentence('The grass is green .')

# just embed a sentence using the StackedEmbedding as you would with any single embedding.
stacked_embeddings.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```
Words are now embedded using a concatenation of three different embeddings. This combination often gives state-of-the-art accuracy.


## BERT Embeddings

[BERT embeddings](https://arxiv.org/pdf/1810.04805.pdf) were developed by Devlin et al. (2018) and are a different kind
of powerful word embedding based on a bidirectional transformer architecture.
We are using the implementation of [huggingface](https://github.com/huggingface/pytorch-pretrained-BERT) in Flair.
The embeddings itself are wrapped into our simple embedding interface, so that they can be used like any other
embedding.

```python
from flair.embeddings import BertEmbeddings

# init embedding
embedding = BertEmbeddings()

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

You can load any of the pre-trained BERT models by providing the model string during initialization:

| ID | Language | Embedding |
| -------------     | ------------- | ------------- |
| 'bert-base-uncased' | English | 12-layer, 768-hidden, 12-heads, 110M parameters |
| 'bert-large-uncased'   | English | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| 'bert-base-cased'    | English | 12-layer, 768-hidden, 12-heads , 110M parameters |
| 'bert-large-cased'   | English | 24-layer, 1024-hidden, 16-heads, 340M parameters |
| 'bert-base-multilingual-cased'     | 104 languages | 12-layer, 768-hidden, 12-heads, 110M parameters |
| 'bert-base-chinese'    | Chinese Simplified and Traditional | 12-layer, 768-hidden, 12-heads, 110M parameters |


## ELMo Embeddings

[ELMo embeddings](http://www.aclweb.org/anthology/N18-1202) were presented by Peters et al. in 2018. They are using
a bidirectional recurrent neural network to predict the next word in a text.
We are using the implementation of [AllenNLP](https://allennlp.org/elmo). As this implementation comes with a lot of
sub-dependencies, which we don't want to include in Flair, you need to first install the library via
`pip install allennlp` before you can use it in Flair.
Using the embeddings is as simple as using any other embedding type:

```python
from flair.embeddings import ELMoEmbeddings

# init embedding
embedding = ELMoEmbeddings()

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
embedding.embed(sentence)
```

AllenNLP provides the following pre-trained models. To use any of the following models inside Flair
simple specify the embedding id when initializing the `ELMoEmbeddings`.

| ID | Language | Embedding |
| ------------- | ------------- | ------------- |
| 'small' | English | 1024-hidden, 1 layer, 14.6M parameters |
| 'medium'   | English | 2048-hidden, 1 layer, 28.0M parameters |
| 'original'    | English | 4096-hidden, 2 layers, 93.6M parameters |
| 'pt'   | Portuguese | |
| 'pubmed' | English biomedical data | [more information](https://allennlp.org/elmo) |


## Combining BERT and Flair

You can very easily mix and match Flair, ELMo, BERT and classic word embeddings. All you need to do is instantiate each embedding you wish to combine and use them in a StackedEmbedding. 

For instance, let's say we want to combine the multilingual Flair and BERT embeddings to train a hyper-powerful multilingual downstream task model. 

First, instantiate the embeddings you wish to combine: 

```python
from flair.embeddings import FlairEmbeddings, BertEmbeddings

# init Flair embeddings
flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')

# init multilingual BERT
bert_embedding = BertEmbeddings('bert-base-multilingual-cased')
```

Now instantiate the `StackedEmbeddings` class and pass it a list containing these three embeddings.

```python
from flair.embeddings import StackedEmbeddings

# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[flair_forward_embedding, flair_backward_embedding, bert_embedding])
```

That's it! Now just use this embedding like all the other embeddings, i.e. call the `embed()` method over your sentences.

```python
sentence = Sentence('The grass is green .')

# just embed a sentence using the StackedEmbedding as you would with any single embedding.
stacked_embeddings.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
```

Words are now embedded using a concatenation of three different embeddings. This means that the resulting embedding
vector is still a single PyTorch vector.


## Next 

You can now either look into [document embeddings](/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md) to embed entire text 
passages with one vector for tasks such as text classification, or go directly to the tutorial about 
[loading your corpus](/resources/docs/TUTORIAL_6_CORPUS.md), which is a pre-requirement for
[training your own models](/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md).

