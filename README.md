# hSBM_Topicmodel

A tutorial for topic-modeling with hierarchical stochastic blockmodels using graph-tool.


## Data

The corpus is saved in corpus.txt, where each line is a separate doc with words separated by whitespace.
Optionally, we can provide a file with titles for the documents in titles.txt

## Setup

#### Installing graph-tool

We use the [graph-tool](https://graph-tool.skewed.de/) package for finding topical structure in the word-document networks. 
See the [installation-instructions](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions), where you will find packages for linux, etc.

Another option, which worked for me is to use conda virtual environments:

- Install anaconda, see e.g. [here](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

- Create a conda-environment called "gt-test" and install graph-tool. The instructions follow the approach outlined [here](https://gitlab.com/ostrokach-forge/graph-tool)

`conda create -n gt-test -c ostrokach-forge -c conda-forge -c defaults --override-channels "python=3.6" graph-tool`

- in order to enable plotting functionality functions we also have to install this:

`conda install -n gt-test -c pkgw-forge gtk3`

- activate the new environment

`source activate gt-test`

- in order to use jupyter notebook, we have to install jupyter into the environment and link the environment as a separate kernel

`conda install jupyter`

`python -m ipykernel install --user --name gt-test --display-name "gt-test"`

#### Download the hSBM-TopicModel repo

`git clone https://github.com/martingerlach/hSBM_Topicmodel.git`

## Run the code

Start jupyter notebooks

`jupter notebook`

then select the 'hsbm-topicmodel-tutorial'-notebook.


