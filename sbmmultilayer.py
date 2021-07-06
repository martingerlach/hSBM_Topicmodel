# -*- coding: utf-8 -*-
#!/usr/bin/python3
'''
Description: 2-layer hierarchical SBM based on doc-word network and hyperlink network.

Author: Chris Hyland and Yuanming Tao
'''


import os,sys
import graph_tool.all as gt
import numpy as np
import pandas as pd
import pickle
from collections import Counter,defaultdict

class sbmmultilayer:
    """
    Topic modelling using Hierarchical Multilayer Stochastic Block Model. The
    model is an implementatino of a 2-layer multilayer SBM where the first layer
    is a bipartite network between documents and word-types based off the TopSBM
    formulation. The second layer is a hyperlink network between the documents.

    Parameters
    ----------
    random_seed : int, default = None
        Controls randomization used in topSBM
    n_init : int
         Number of random initialisations to perform in order to avoid a local
         minimum of MDL. The minimum MDL solution is chosen.

    Attributes
    ----------
    g : graph_tool.Graph
        Multilayered network
    words: list
        Word nodes.
    documents: list
        Document nodes.
    state:
        Inferred state from graph_tool.
    groups: dict
        Results of group membership from inference.
        Key is an integer, indicating the level of grouping (starting from 0).
        Value is a dict of information about the grouping which contains:
    mdl: float
        The minimum description length of inferred state.
    n_levels: int
        Number of levels in hierarchy of the inferred state.
    """
    def __init__(self, random_seed = None, n_init = 1):
        self.random_seed = random_seed
        self.n_init = n_init

        self.g = None
        self.words = []
        self.documents = []
        self.state = None
        self.groups = {}
        self.mdl = np.nan
        self.n_levels = np.nan


    def make_graph(self, list_texts, list_titles, list_hyperlinks):
        """
        Load a corpus and generate the multilayered network where one layer
        is the multigraph word-document bipartite network and another is the document
        hyperlink network.

        Document node will be given be the number 0 and word nodes will be
        given the number 1.

        Parameters
        ----------
        list_texts : type
            Description of parameter `list_texts`.
        list_titles : type
            Description of parameter `list_titles`.
        list_hyperlinks : type
            Description of parameter `list_hyperlinks`.

        Returns
        -------
        type
            Description of returned object.

        """
        # Number of documents
        D = len(list_texts)

        # Initialize a graph to store multilayer graph
        g = gt.Graph(directed=True)

        #### Define node properties ####
        # Documents - 'title', words - 'word'
        name = g.vp["name"] = g.new_vp("string")
        # Documents nodes (0), word nodes (1)
        kind = g.vp["kind"] = g.new_vp("int")
        # Specify Vertex Layers: word node: [0]; doc node: [0, 1]
        vlayers = g.vp["vlayers"] = g.new_vp("vector<int>")

        #### Define edge properties ####
        # Edge multiplicity
        edgeCount = g.ep["edgeCount"] = g.new_ep("int")

        # Need to specify edgetype to indicate which layer an edge is in
        # Hyperlink edge (1) and doc-word edge (0)
        edgeType = g.ep["edgeType"] = g.new_ep("int")

        # Create dictionary of vertices with key-value pair {name: Vertex}
        doc_vertices = defaultdict(lambda: g.add_vertex())
        word_vertices = defaultdict(lambda: g.add_vertex())

        # Initialise document nodes based on name of wikipedia article
        for title in list_titles:
            d = doc_vertices[title]
            vlayers[d] = [0,1]

        #### Construct hyperlink graph ####
        # Construct hyperlinks between articles
        for pair in list_hyperlinks:
            # Retrieve source and target nodes and add hyperlink edge
            s = doc_vertices[pair[0]]
            t = doc_vertices[pair[1]]
            e = g.add_edge(s, t)
            edgeCount[e] = 1
            edgeType[e] = 1 # Indicates the edge is hyperlink

        #### Construct bipartite word-doc graph ####
        # Create edges between documents and words
        for doc_id in range(D):
            title = list_titles[doc_id]
            text = list_texts[doc_id]
            d = doc_vertices[title]
            name[d] = title
            kind[d] = 0 # label 0 is document node
            c = Counter(text) # {word: # of ocurrences}
            for word, count in c.items():
                w = word_vertices[word]
                name[w] = word
                kind[w] = 1 # word node
                vlayers[w] = [0]
                e = g.add_edge(d, w) # add link between document and word node
                edgeCount[e] = count # assign weighting to edge based on number of occurrences
                edgeType[e] = 0 # to indicate the edge is word occurrence

        # Initialise words and documents network to model.
        self.g = g
        self.words = [ g.vp['name'][v] for v in  g.vertices() if g.vp['kind'][v]==1   ]
        self.documents = [ g.vp['name'][v] for v in  g.vertices() if g.vp['kind'][v]==0   ]


    def fit(self):
        """
        Fits the hSBM to the undirected, layered multigraph, where the graph in the doc-word layer is bipartite.
        This uses the independent layer multilayer network where we have a degree-corrected SBM.
        """
        # We need to impose constraints on vertices and edges to keep track which layer are they in.
        state_args = {}
        # Vertices with different label values will not be clustered in the same group
        state_args["pclabel"] = self.g.vp["kind"]
        # Split the network in discrete layers based on edgetype. 0 is for word-doc graph and 1 is for hyperlink graph.
        state_args["ec"] = self.g.ep["edgeType"]
        # Independent layers version of the model (instead of 'edge covariates')
        state_args["layers"] = True
        # Edge multiplicities based on occurrences.
        state_args["eweight"] = self.g.ep.edgeCount

        self.g.save("foo.gt.gz")
        # Specify parameters for community detection inference
        gt.seed_rng(self.random_seed)
        mdl = np.inf
        # Fit n_init random initializations to avoid local optimum of MDL.
        for _ in range(self.n_init):
            # Enables the use of LayeredBlockState. Use a degree-corrected layered SBM.
            state_temp = gt.minimize_nested_blockmodel_dl(self.g, state_args=dict(base_type=gt.LayeredBlockState,
                                                                                  **state_args))
            mdl_temp = state_temp.entropy()
            if mdl_temp < mdl:
                # We have found a new optimum
                mdl = mdl_temp
                state = state_temp.copy()

        self.state = state
        self.mdl = state.entropy()

        n_levels  = len(self.state.levels)
        # Figure out group levels
        if n_levels == 2:
            # Bipartite network
            self.groups = { 0: self.get_groupStats(l=0) }
            self.n_levels = len(self.groups)
        # Omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
        else:
            self.groups = { level: self.get_groupStats(l=level) for level in range(n_levels - 2) }
            self.n_levels = len(self.groups)


    def get_groupStats(self, l=0):
        '''
        Description:
        -----------
            Extract statistics on group membership of nodes form the inferred state.
        Returns:  dict
        -----------
            - B_d, int, number of doc-groups
            - B_w, int, number of word-groups

            - p_td_d, array (B_d, D);
                      doc-group membership:
                      # group membership of each doc-node, matrix of ones and zeros, shape B_d x D
                      prob that doc-node d belongs to doc-group td: P(td | d)

            - p_tw_w, array (B_w, V);
                      word-group membership:
                      # group membership of each word-node, matrix of ones or zeros, shape B_w x V
                      prob that word-node w belongs to word-group tw: P(tw | w)

            - p_tw_d, array (B_w, D);
                      doc-topic mixtures:
                      ## Mixture of word-groups into documents P(t_w | d), shape B_w x D
                      prob of word-group tw in doc d P(tw | d)

            - p_w_tw, array (V, B_w);
                      per-topic word distribution, shape V x B_w
                      prob of word w given topic tw P(w | tw)
        '''

        V = self.get_V() # number of word-type nodes
        D = self.get_D() # number of document nodes
        N = self.get_N() # number of word-tokens (edges excluding hyperlinks)

        g = self.g
        state = self.state

        # Retrieve the number of blocks
        # Project the partition at level l onto the lowest level and return the corresponding state.
        state_l = state.project_level(l).agg_state.copy(overlap=True)
        B = state_l.get_B() # number of blocks

        # Returns an edge property map which contains the block labels pairs for each edge.
        # Note that in the text network, one endpoint will be in doc blocks and other endpoint
        # will be in word type block
        state_l_edges = state_l.get_edge_blocks()

        # Count labeled half-edges, total sum is # of edges
        # Number of half-edges incident on word-node w and labeled as word-group tw
        n_wb = np.zeros((V,B)) # will be reduced to (V, B_w)

        # Number of half-edges incident on document-node d and labeled as document-group td
        n_db = np.zeros((D,B)) # will be reduced to (D, B_d)

        # Number of half-edges incident on document-node d and labeled as word-group tw
        n_dbw = np.zeros((D,B))  # will be reduced to (D, B_w)

        # Count labeled half-edges, total sum is # of edges
        for e in g.edges():
            # We only care about edges in text network
            if g.ep.edgeType[e] == 0:
                # z1 will have values from 1, 2, ..., B_d; document-group i.e document block that doc node is in
                # z2 will have values from B_d + 1, B_d + 2,  ..., B_d + B_w; word-group i.e word block that word type node is in
                z1, z2 = state_l_edges[e]
                # v1 ranges from 0, 1, 2, ..., D - 1
                # v2 ranges from D, ..., (D + V) - 1 (V # of word types)
                v1 = int(e.source()) # document node index
                v2 = int(e.target()) # word type node index
                n_wb[v2-D,z2] += 1 # word type v2 is in topic z2
                n_db[v1,z1] += 1 # document v1 is in doc cluster z1
                n_dbw[v1,z2] += 1 # document v1 has a word in topic z2

        # Retrieve the corresponding submatrices
        n_db = n_db[:, np.any(n_db, axis=0)] # (D, B_d)
        n_wb = n_wb[:, np.any(n_wb, axis=0)] # (V, B_w)
        n_dbw = n_dbw[:, np.any(n_dbw, axis=0)] # (D, B_d)

        B_d = n_db.shape[1]  # number of document groups
        B_w = n_wb.shape[1] # number of word groups (topics)

        # Group membership of each word-type node in topic, matrix of ones or zeros, shape B_w x V
        # This tells us the probability of topic over word type
        p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

        # Group membership of each doc-node, matrix of ones of zeros, shape B_d x D
        p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

        # Mixture of word-groups into documents P(t_w | d), shape B_d x D
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

        # Per-topic word distribution, shape V x B_w
        p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]

        result = {}
        result['Bd'] = B_d # Number of document groups
        result['Bw'] = B_w # Number of word groups
        result['p_tw_w'] = p_tw_w # Group membership of word nodes
        result['p_td_d'] = p_td_d # Group membership of document nodes
        result['p_tw_d'] = p_tw_d # Topic proportions over documents
        result['p_w_tw'] = p_w_tw # Topic distribution over words
        return result


    def get_topics(self, l=0, n=10):
        '''
        Get the n most common words for each word-group in level l.
        Return tuples (word,P(w|tw))
        '''
        dict_groups = self.groups[l]
        Bw = dict_groups['Bw'] # number of word-groups
        p_w_tw = dict_groups['p_w_tw'] # topic proportions over documents
        words = self.words
        # Loop over all word-groups
        dict_group_words = {}
        for tw in range(Bw):
            p_w_ = p_w_tw[:, tw]
            ind_w_ = np.argsort(p_w_)[::-1]
            list_words_tw = []
            for i in ind_w_[:n]:
                if p_w_[i] > 0:
                    list_words_tw+=[(words[i],p_w_[i])]
                else:
                    break
            dict_group_words[tw] = list_words_tw
        return dict_group_words


    def get_topicProportion(self, doc_index, l=0):
        '''
        Get the topic proportion for a particular document
        '''
        dict_groups =  self.groups[l]
        p_tw_d = dict_groups['p_tw_d'] # Topic proportions over all documents
        list_topics_tw = []
        for tw, p_tw in enumerate(p_tw_d[:,doc_index]):
            list_topics_tw += [(tw,p_tw)]
        return list_topics_tw


    def get_docclusters(self,l=0,n=10):
        '''
        Get n 'most common' documents from each document cluster.
        Most common refers to largest contribution in group membership vector.
        For the non-overlapping case, each document belongs to one and only one group with prob 1.
        '''
        dict_groups = self.groups[l]
        Bd = dict_groups['Bd'] # number of doc-groups
        p_td_d = dict_groups['p_td_d'] # p_td_d, array B_d x D; doc-group membership: prob that doc-node d belongs to doc-group td: P(td | d)

        docs = self.documents
        # Loop over all word-groups to retrieve clusters.
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


    def clusters_query(self,doc_index,l=0):
        '''
        Get all documents in the same group as the query-document.
        Note: Works only for non-overlapping model.
        '''
        dict_groups = self.groups[l]
        Bd = dict_groups['Bd']
        p_td_d = dict_groups['p_td_d']

        documents = self.documents
        ## loop over all word-groups
        dict_group_docs = {}
        td = np.argmax(p_td_d[:,doc_index])

        list_doc_index_sel = np.where(p_td_d[td,:]==1)[0]

        list_doc_query = []

        for doc_index_sel in list_doc_index_sel:
            if doc_index != doc_index_sel:
                list_doc_query += [(doc_index_sel,documents[doc_index_sel])]

        return list_doc_query

################################################################################
#### Helper methods ####
################################################################################

# Helper methods for nodes
    def get_D(self):
        '''Return number of doc-nodes == number of documents'''
        return int(np.sum(self.g.vp['kind'].a==0))

    def get_V(self):
        '''Return number of word-nodes == types'''
        return int(np.sum(self.g.vp['kind'].a==1))

################################################################################
# Helper methods for edges
    def get_N(self):
        '''Return number of edges == tokens'''
        return int(np.sum([self.g.ep.edgeCount[e] for e in self.g.edges() if self.g.ep['edgeType'][e]== 0 ]))

    def get_Hl(self):
        '''Return number of hyperlinks'''
        return int(np.sum([self.g.ep.edgeCount[e] for e in self.g.edges() if self.g.ep['edgeType'][e]== 1 ]))

################################################################################
# Auxillary methods for graph
    def save_graph(self, filename = 'graph.gt.gz'):
        '''
        Save the word-document network generated by make_graph() as filename.
        Allows for loading the graph without calling make_graph().
        '''
        self.g.save(filename)


    def load_graph(self,filename = 'graph.gt.gz'):
        '''
        Load a word-document network generated by make_graph() and saved with save_graph().
        '''
        self.g = gt.load_graph(filename)
        self.words = [ self.g.vp["name"][v] for v in  self.g.vertices() if self.g.vp['kind'][v]==1   ]
        self.documents = [ self.g.vp['name'][v] for v in  self.g.vertices() if self.g.vp['kind'][v]==0   ]

    def save_inferred_state(self, filename = "state"):
        '''Save the inferred state to pickle file.'''
        state = self.state
        with open('%s.pkl'%filename, 'wb') as f:
            pickle.dump(state, f)

    def load_inferred_state(self, filename = "state"):
        '''Load the saved pickle file for inferred state.'''
        with open('%s.pkl'%filename, 'rb') as f:
            state = pickle.load(f)
        self.state = state
        self.mdl = state.entropy()
        n_levels  = len(state.levels)
        # Only trivial bipartite structure
        if n_levels == 2:
            self.groups = { 0: get_groupStats(l=0) }
            # Omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
        else:
            self.groups = { level: self.get_groupStats(l=level) for level in range(n_levels - 2) }
            self.n_levels = len(self.groups)

################################################################################
    def plot(self, filename = None,nedges = 1000):
        '''
        Plot the network and group structure by default.
        optional:
        - filename, str; where to save the plot. if None, will not be saved
        - nedges, int; subsample  to plot (faster, less memory)
        '''
        g = self.g
        self.state.draw(output=filename,
                                 subsample_edges = nedges,
                                 layout = "bipartite")
