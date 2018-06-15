# hSBM_Topicmodel

A tutorial for topic-modeling with hierarchical stochastic blockmodels using graph-tool.


## Data

The corpus is saved in corpus.txt, where each line is a separate doc with words separated by whitespace.
Optionally, we can provide a file with titles for the documents in titles.txt

## Setup

#### Install graph-tool

We use the [graph-tool](https://graph-tool.skewed.de/) package for finding topical structure in the word-document networks.
- see the [installation-instructions](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions), where you will find packages for linux, etc.
- an alternative for linux is to install via a conda-environment, see [here](https://gitlab.com/ostrokach-forge/graph-tool)

#### Get Jupyter notebook

In order to execture the tutorial-notebook, install [jupyter](http://jupyter.org/), e.g.

`pip install jupyter`

#### Get hSBM-TopicModel repository

In order to do topic modeling with stochastic block models we need to get the code from the repositroy:

`git clone https://github.com/martingerlach/hSBM_Topicmodel.git`

## Run the code

Start jupyter notebooks

`jupter notebook`

then select the 'TopSBM-tutorial'-notebook.

It will guide you through the different steps to do topic modeling with stochastic block models:

- How to construct the word-document network from a corpus of text

- How to fit the stochastic block model to the word-document network

- How to extract the topics from the fitted model, e.g.
	- the most important words for each topic
	- the clustering of documents
	- the topic mixtures for each document

- How to visualize the topical structure, in particular the hierarchy of topics


