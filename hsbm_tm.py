from __future__ import print_function
import numpy as np
import os,sys,argparse
from collections import Counter,defaultdict
import pickle
import graph_tool.all as gt


class hsbm(object):

    def __init__(self,args):
        '''
        Initialize hsbm-instance
        - create a folder where to save results: self.args.output
        - make a bipartite word-doc graph from the corpus. save as self.graph
        - do the hsbm inference. save the state as self.inference
        '''
        self.args = args
        self.out_path = self.args.output

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        ## get the graph-object
        self.graph = self.make_graph()
        ## do the hsbm-inference
        self.state = self.inference(self.graph)

    def make_graph(self):
        """Transform a corpus into a bipartite word-document network with D docs and V word-(types)
            and N word-tokens.
           The corpus is expected to be in the form that each line corresponds to a doc
           and word-tokens in each doc a re separated by whitespace

           Returns a graph-tool graph-instance with nodes and edges:
           - nodes 0,...,D-1 correspond to the document nodes 
           - nodes D,...,V+D-1 correspond to the word-nodes
           - we have N edges, where an edge corresponds to a word-token 
           (occurrence of a word-type in a document)

           We define vertex-properties
           - 'name': 
                - for docs the document titles (supplied in a separate file from args.titles)
                  if n.a. we index from 0,...,D-1
                - for words it is the string of the word-type
            - 'kind': doc-nodes:0; word-nodes:1

           We save the graph as graph.xml.gz
           We save a file called 'nodes' which contains <node-index> <name>

        """
        ## if no graph exists, make a graph
        if self.args.graph==None:

            ## load corpus
            filename_corpus = self.args.corpus
            list_texts = []
            with open(filename_corpus) as f:
                x = f.readlines()
            list_texts = [h.split() for h in x]
            D = len(list_texts)

            ## load titles
            try:
                filename_titles = self.args.titles
                list_titles = []
                with open(filename_titles) as f:
                    x = f.readlines()
                list_titles = [h.split()[0] for h in x]
            except IOError:
                list_titles = [str(h) for h in range(D)]



            ## make a graph
            ## create a graph
            g = gt.Graph(directed=False)
            ## define node properties
            ## name: docs - title, words - 'word'
            ## kind: docs - 0, words - 1
            ## cat: category of the doc-node, words - not defined
            name = g.vp["name"] = g.new_vertex_property("string")
            kind = g.vp["kind"] = g.new_vertex_property("bool")  

            docs_add = defaultdict(lambda: g.add_vertex())
            words_add = defaultdict(lambda: g.add_vertex())

            ## add all documents first
            for i_d in range(D):
                title = list_titles[i_d]
                d=docs_add[title]

            ## add all documents and words as nodes
            ## add all tokens as links
            for i_d in range(D):
                title = list_titles[i_d]
                text = list_texts[i_d]
                
                d=docs_add[title]
                name[d] = title
                kind[d] = 0
                c=Counter(text)
                for word,count in c.items():
                    w=words_add[word]
                    name[w] = word
                    kind[w] = 1
                    for n in range(count):
                        g.add_edge(d,w)
            ## save graph
            g.save(os.path.join(self.out_path,'graph.xml.gz'))
        ## if we already have a graph, simply load it
        else:
            g = gt.load_graph(self.args.graph)
        self.make_index(g)
        return g

    def make_index(self,g):
        ## write an index with the names of alle nodes
        ###write a file with node-index, count (length/occurrences for doc/word), kind (0/1 for doc/word) and name
        with open(os.path.join(self.out_path,'nodes'),'w') as f:
            for v in g.vertices():
                index = str(v)
                kind = g.vp['kind'][v]
                name = g.vp['name'][v]
                n_count = v.out_degree()
                f.write('%s \t %s \t %s \t %s \n'%(index,n_count,kind,name))

    def inference(self,g):
        """We load a graph-tool graph-object and fit an hsbm:
           - hierarchical
           - nonoverlapping
           - degree-corrected
           We get a state-object which is a 'NestedBlockState'.
           We save as 'state.pkl'
        """
        if self.args.state == None:
            state=gt.minimize_nested_blockmodel_dl(g,deg_corr=True,overlap=False)
            ## save state
            with open(os.path.join(self.out_path,'state.pkl'),'wb') as f:
                pickle.dump(state,f)
        ## if the state already exists, we just load
        else:
            with open(self.args.state,'rb') as f:
                state = pickle.load(f)
        return state

    def group_membership(self,g,state):
        """Given the graph  and the inferred state, we get the group-membership
            of each node in each level of the hierarchy.
            Note that we have a non-overlapping hsbm, so group-membership is 'hard'.
            IN:
            - g: graph-tool graph-object
            - state: inferred block-state
            OUT:
            We write a file for each hierarchy-level: 'blocks_level_<level>'
            <node-index> <group-number>
        """
        L = len(state.levels)

        # ## block membership of each node on each level
        for l in range(L):
            with open(os.path.join(self.out_path,'nodes_blocks_level_%s'%(l)),'w') as f:
                state_proj_l = state.project_level(l)
                blocks_proj_l = state_proj_l.get_blocks()
                for i_b,b in enumerate(blocks_proj_l.a):
                    f.write('%s \t %s \n'%(i_b,b)) 

    def draw_hierarchy(self,state,pedges):
        """We draw the hierarchy of the fitted hsbm.
        IN:
        - state, inferred hiearchical block state from hsbm
        - pedges, float, fraction of edges to keep in plot
        OUT:
        - draw_hierarchy.pdf showing a visualization of the graph and the group-structure
        """
        gt.draw_hierarchy(state,layout='bipartite',\
                  output=os.path.join(self.out_path,"draw_hierarchy.pdf"),\
                  subsample_edges=int(pedges*self.graph.num_edges()),\
                  hshortcuts=1, hide=0,\
                  )

    def block_composition(self):
        '''
        On each level in hierarchy,
        for each block we collect all nodes and write a file
        "nodes_blocks_level_<level>" .
        -for word-blocks we order words according to # of occurrences.
        -for doc-blocks we order docs according to doc-length.

        We load the following files:
        - "nodes"-file created in make_index
        - "nodes_blocks_level_<level>"-file created in group_membership
        '''
        ## get indices and counts only of the word-nodes
        with open(os.path.join(self.out_path,'nodes')) as f:
            x = f.readlines()
        inds = [int(h.split()[0]) for h in x]
        counts = [int(h.split()[1]) for h in x]
        kind = [int(h.split()[2]) for h in x]
        name = [h.split()[3] for h in x]

        ## get the group-memberhsip at each given level
        L = len(self.state.levels)
        for l in range(L-1):
            with open(os.path.join(self.out_path,'nodes_blocks_level_%s'%(l))) as f:
                x = f.readlines()
            blocks = np.array([int(h.split()[1]) for h in x])
            ## these are the blocks
            set_blocks = np.sort(list(set(blocks)))
            for b in set_blocks:
                ## find all words from that block
                inds_sel = np.where(blocks==b)[0]
                name_sel = np.array(name)[inds_sel]

                counts_sel = np.array(counts)[inds_sel]
                kind_sel = np.array(kind)[inds_sel][0]
                ## re-order according to number of occurrences
                ind_sort = np.argsort(counts_sel)[::-1]
                name_sel = name_sel[ind_sort]
                counts_sel = counts_sel[ind_sort]
                if kind_sel == 0:
                    fname = 'composition_level_%s_block_%s_docs'%(l,b)
                elif kind_sel == 1:
                    fname = 'composition_level_%s_block_%s_words'%(l,b)
                else:
                    print('check group memberhsip')
                with open(os.path.join(self.out_path,fname),'w') as f:
                    for n in name_sel:
                        f.write('%s \n'%(n))

    def topic_doc_dist(self,g,state):
        '''
        For each doc (assigned to a given node-block),
        we count the block-membership of all word-tokens appearing in the doc.
        IN:
        - g
        - state
        OUT:
        - we create a file "doc_topicdist_level_<level>"
        doc-index   K_d     b:t
        where K_d is the number of tuples with b:t with t>0 
        where b is the block-index and t is the number tokens assigned to block b in doc d
        '''
        ## loop over all docs
        L = len(self.state.levels)
        for l in range(L-1):
            state_proj_l = state.project_level(l)
            blocks_proj_l = state_proj_l.get_blocks()
            with open(os.path.join(self.out_path,'doc_topicdist_level_%s'%(l)),'w') as f:
                for v in g.vertices():
                    if g.vp['kind'][v]==0:
                    ## loop over all docs
                        list_b_doc = []
                        for v_nbr in v.out_neighbours():
                            list_b_doc += [blocks_proj_l[v_nbr]]
                        c_b_doc = Counter(list_b_doc)
                        set_b_doc = sorted(list(c_b_doc.keys()))
                        len_b_doc = len(set_b_doc)
                        f.write('%s \t %s'%(int(v),len_b_doc))
                        for b in set_b_doc:
                            f.write('\t %s:%s'%(b,c_b_doc[b]))
                        f.write('\n')

                    else:
                    ## stop the loop
                        break
                    # break


        # print(set_blocks)

if __name__=='__main__':

    parser = argparse.ArgumentParser("Using graph-tool's hSBM for topic modeling")
    parser.add_argument("-o", "--output",help="Where to save output and results",default='result/',type=str)
    parser.add_argument("-g", "--graph",help="Filename to existing graph (if None, rebuild from corpus.txt)",default=None,type=str)
    parser.add_argument("-s", "--state",help="Filename to existing inferred graph (if None, do inference again)",default=None,type=str)
    parser.add_argument("-c", "--corpus",help="Filename to corpus text",default='corpus.txt',type=str)
    parser.add_argument("-t", "--titles",help="Filename to titles of texts (will be added as node-properties - if file does not exist: indexed with integers starting from 0)",default='titles.txt',type=str)
    parser.add_argument("-d", "--draw",help="Whether to draw the hierarchy of the inferred network (default=0/no)",default=0,type=int)
    parser.add_argument("-p", "--pedges",help="Fraction of edges to include when drawing the hierarchy (only has an effect if --draw==1",default=0.01,type=float)
    parser.add_argument("-r", "--remove",help="Delete output-folder (get rid of old result), default=0",default=0,type=int)

    args = parser.parse_args()

    ## do the hsbm on a corpus
    tm = hsbm(args=args)
    ## get the group-membership for each node on each level of the hierarchy
    tm.group_membership(tm.graph,tm.state)
    ## block composition
    tm.block_composition()
    ## doc_top
    tm.topic_doc_dist(tm.graph,tm.state)
    ## draw the hierarchy
    if args.draw==1:
        tm.draw_hierarchy(tm.state,args.pedges)