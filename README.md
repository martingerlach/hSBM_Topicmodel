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
- nodes_blocks_level_<level>
	- for each node, we write the block it was assigned to
	- node-index \t block-index
- corpus_composition_level_<level>_block_<block>_<kind>
	- all nodes belonging to each block (block-index) on a given level in the hierarchy (indicate whether words or documents)
- doc_topic_dist_level_<level>
	- for each level, we make a list of all documents and count the number of word-tokens assigned to each word-block; this can be interpreted as the topic distribution
	- doc-index \t number of non-empty word-blocks \t block-index:number of tokens associated to word-block
- draw_hierarchy.pdf
	- layout of inferred graph (documents on the left and words on the right)
- graph.xml.gz (a graph-tool graph-object)
- state.pkl (the pickled inferred state from graph-tool)



Requirements:
- python
- numpy
- graph-tool 2.22, available here: https://graph-tool.skewed.de/


