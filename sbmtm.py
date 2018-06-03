from __future__ import print_function
import numpy as np
import os,sys,argparse
from collections import Counter,defaultdict
import pickle
import graph_tool.all as gt

def make_graph(list_texts, documents = None):
    """Transform a corpus into a bipartite word-document network with D docs and V word-(types)
        and N word-tokens.
       The corpus is in the following format:
       - or a list of documents where each doc is a list of tokens 

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


    """

    D = len(list_texts)

    ## if there are no document titles, we assign integers 0,...,D-1
    ## otherwise we use supplied titles
    if documents == None:
        list_titles = [str(h) for h in range(D)]
    else:
        list_titles = documents

    ## make a graph
    ## create a graph
    g = gt.Graph(directed=False)
    ## define node properties
    ## name: docs - title, words - 'word'
    ## kind: docs - 0, words - 1
    name = g.vp["name"] = g.new_vertex_property("string")
    kind = g.vp["kind"] = g.new_vertex_property("int")  

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

    return g


def sbm_fit(g,overlap=False,hierarchical=True):
    '''
    Infer the block structure of the bipartite word-document network.
    Default: a hierarchical, nonoverlapping blockmodel.
    IN:
    - g, graph, see make_graph
    OUT:
    - ???
    '''
    ## vertex property map to ensure that words and documents are not clustered together
    clabel = g.vp['kind']

    ## the inference
    state=gt.minimize_nested_blockmodel_dl(g,deg_corr=True,overlap=False,\
                                       state_args={'clabel':clabel,'pclabel':clabel})
    
    ## collect all the results in a dictionary.
    result = {}
    result['state'] = state
    result['mdl'] = state.entropy()


    ## keep the list of words and documents
    words = [ g.vp['name'][v] for v in  g.vertices() if g.vp['kind'][v]==1   ]
    docs = [ g.vp['name'][v] for v in  g.vertices() if g.vp['kind'][v]==0   ]
    result['words'] = words
    result['docs'] = docs

    V = get_V(g)
    D = get_D(g)
    N = get_N(g)
    result['V'] = V
    result['D'] = D
    result['N'] = N

    ## get the group membership statistics
    L = len(state.levels)
    result['L'] = L

    dict_stats_groups = {}
    ## for each level in the hierchy we make a dictionary with the node-group statistics
    ## e.g. group-membership
    ## we omit the highest level in the hierarchy as there will be no more distinction between
    ## word- and doc-groups
    for l in range(L-1):
        dict_stats_groups_l = get_groups(state,g,l=l)
        dict_stats_groups[l] = dict_stats_groups_l
    result['stats_groups'] = dict_stats_groups
    return result


## get group-topic statistics
def get_groups(state,g,l=0):
    '''
    extract statistics on group membership of nodes form the inferred state.
    return dictionary
    - B_d, int, number of doc-groups
    - B_w, int, number of word-groups
    - p_tw_w, array B_w x V; word-group-membership:
         prob that word-node w belongs to word-group tw: P(tw | w) 
    - p_td_d, array B_d x D; doc-group membership:
         prob that doc-node d belongs to doc-group td: P(td | d)
    - p_w_tw, array V x B_w; topic distribution:
         prob of word w given topic tw P(w | tw)
    - p_tw_d, array B_w x d; doc-topic mixtures:
         prob of word-group tw in doc d P(tw | d)
    '''
    V = get_V(g)
    D = get_D(g)
    N = get_N(g)
    ## if we have a nested state: project to a level, otherwise take state as is
    if isinstance(state,(gt.BlockState,gt.OverlapBlockState)):
        state_l = state.copy(overlap=True)
    else:
        state_l = state.project_level(l).copy(overlap=True)

    state_l_edges = state_l.get_edge_blocks() ## labeled half-edges

    ## count labeled half-edges, group-memberships
    B = state_l.B
    n_wb = np.zeros((V,B)) ## number of half-edges incident on word-node w and labeled as word-group tw
    n_db = np.zeros((D,B)) ## number of half-edges incident on document-node d and labeled as document-group td
    n_dbw = np.zeros((D,B)) ## number of half-edges incident on document-node d and labeled as word-group td

    for e in g.edges():
        z1,z2 = state_l_edges[e]
        v1 = e.source()
        v2 = e.target()
        n_db[int(v1),z1] += 1
        n_dbw[int(v1),z2] += 1
        n_wb[int(v2)-D,z2] += 1

    p_w = np.sum(n_wb,axis=1)/float(np.sum(n_wb))

    ind_d = np.where(np.sum(n_db,axis=0)>0)[0]
    Bd = len(ind_d)
    n_db = n_db[:,ind_d]

    ind_w = np.where(np.sum(n_wb,axis=0)>0)[0]
    Bw = len(ind_w)
    n_wb = n_wb[:,ind_w]

    ind_w2 = np.where(np.sum(n_dbw,axis=0)>0)[0]
    n_dbw = n_dbw[:,ind_w2]

    ## group-membership distributions
    # group membership of each word-node P(t_w | w)
    p_tw_w = (n_wb/np.sum(n_wb,axis=1)[:,np.newaxis]).T

    # group membership of each doc-node P(t_d | d)
    p_td_d = (n_db/np.sum(n_db,axis=1)[:,np.newaxis]).T

    ## topic-distribution for words P(w | t_w)
    p_w_tw = n_wb/np.sum(n_wb,axis=0)[np.newaxis,:]

    ## Mixture of word-groups into documetns P(t_w | d)
    p_tw_d = (n_dbw/np.sum(n_dbw,axis=0)[np.newaxis,:]).T


    result = {}
    result['Bd'] = Bd
    result['Bw'] = Bw
    result['p_tw_w'] = p_tw_w
    result['p_td_d'] = p_td_d
    result['p_w_tw'] = p_w_tw 
    result['p_tw_d'] = p_tw_d

    return result


### functions handling the results-dictionary

def sbm_plot(dict_result, pedges = 0.05,filename_save = None):
    '''
    Plot the graph and group structure.
    optional: 
    - pedges, float; subsample fraction of edges to plot (faster, less memory)
    - filename_save, str; where to save the plot. if None, will not be saved
    '''
    N = dict_result['N']
    state = dict_result['state']
    _ = gt.draw_hierarchy(state,layout='bipartite',\
              output=filename_save,\
              subsample_edges=int(pedges*N),\
              hshortcuts=1, hide=0,\
              )

def get_Bwd(dict_result,l):
    '''
    Return the number of document and word groups on a given level
    '''
    dict_groups = dict_result['stats_groups'][l]
    return dict_groups['Bd'], dict_groups['Bw']

def most_common_words(dict_result,l,n=10):
    '''
    get the n most common words for each word-group in level l.
    return tuples (word,P(w|tw))
    '''
    dict_groups = dict_result['stats_groups'][l]
    Bw = dict_groups['Bw']
    p_w_tw = dict_groups['p_w_tw']

    words = dict_result['words']

    ## loop over all word-groups
    dict_group_words = {}
    for tw in range(Bw):
        p_w_ = p_w_tw[:,tw]
        ind_w_ = np.argsort(p_w_)[::-1]
        list_words_tw = []
        for i in ind_w_[:n]:
            if p_w_[i] > 0:
                list_words_tw+=[(words[i],p_w_[i])]
            else:
                break
        dict_group_words[tw] = list_words_tw
    return dict_group_words

def most_common_docs(dict_result,l,n=10):
    '''
    Return n 'most common' documents for each doc-group td,
    i.e. they have the largest value in p(td | d)
    '''
    dict_groups = dict_result['stats_groups'][l]
    Bd = dict_groups['Bd']
    p_td_d = dict_groups['p_td_d']

    docs = dict_result['docs']

    ## loop over all word-groups
    dict_group_docs = {}
    for td in range(Bd):
        p_d_ = p_td_d[td,:]
        ind_d_ = np.argsort(p_d_)[::-1]
        list_docs_td = []
        for i in ind_d_[:n]:
            if p_d_[i] > 0:
                list_docs_td+=[(docs[i],p_d_[i])]
            else:
                break
        dict_group_docs[td] = list_docs_td
    return dict_group_docs

def sbm_topicdist_w(dict_result,l):
    dict_groups = dict_result['stats_groups'][l]
    p_w_tw = dict_groups['p_w_tw']
    return p_w_tw

def sbm_topicdist_d(dict_result,l):
    dict_groups = dict_result['stats_groups'][l]
    p_tw_d = dict_groups['p_tw_d']
    return p_tw_d

def sbm_group_membership(dict_result,l):
    dict_groups = dict_result['stats_groups'][l]
    p_tw_w = dict_groups['p_tw_w']
    p_td_d = dict_groups['p_td_d']
    return p_td_d,p_tw_w




### helper functions

def get_V(g):
    '''
    return number of word-nodes == types
    '''
    return int(np.sum(g.vp['kind'].a==1)) # no. of types
def get_D(g):
    '''
    return number of doc-nodes == number of documents
    '''
    return int(np.sum(g.vp['kind'].a==0)) # no. of types
def get_N(g):
    '''
    return number of edges == tokens
    '''
    return int(g.num_edges()) # no. of types


### functions for reading corpora
def read_corpus(filename):
    '''We assume that each line is a document and all words are separated by blank space.
        we return a list of documents, where each document is a list of strings (the word-tokens)
    '''
    with open(filename,'r') as f:
        x=f.readlines()
    texts = [h.split() for h in x]
    return texts