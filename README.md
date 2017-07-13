hSBM_Topicmodel

Example script to do topic-modeling with graph-tool's hierarchical stochastic blockmodel (hsbm).

Simply run 
python hsbm_tm.py
(--help for possible arguments)

## Data

The corpus is saved in corpus.txt, where each line is a separate doc with words separated by whitespace.
Optionally, we can provide a file with titles for the documents in titles.txt

## OUTPUT

- nodes: node-indices
node-index 	counts	kind	name
with counts = doc-length (docs) or total #of occurrences (words)
kind = 0 (for doc-node) or 1 (for word-node)
name = title (for doc), word-type for word-nodes
- 
- graph.xml.gz (a graph-tool graph-object)
- state.pkl (the pickled inferred state from graph-tool)


Requirements:
- python
- numpy
- graph-tool 2.22, available here: https://graph-tool.skewed.de/


