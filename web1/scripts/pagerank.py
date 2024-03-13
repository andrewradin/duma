from builtins import range
import networkx as nx
import numpy as np
import scipy


def mat_without_nodes(mat, to_rem):
    """
    This returns the matrix with the specified rows/cols removed.
    This isn't super efficient, csr or csc sparse format will only be fast at
    slicing one of columns or rows, but not both.  But still faster than
    the alternative of not doing this.
    """
    row_mask = np.ones(mat.shape[0], dtype=bool)
    row_mask[to_rem] = False
    col_mask = np.ones(mat.shape[1], dtype=bool)
    col_mask[to_rem] = False
    return mat[row_mask][:,col_mask]

def run_page_rank(ref_node, g, ppi_g, M, prot_prefix, restart_prob, iterations):
    target_nodes = set([n
                         for n in nx.all_neighbors(g, ref_node) 
                         if n.startswith(prot_prefix)
                        ])
    person_dict = {n:g[ref_node][n]['weight']
                   for n in ppi_g
                   if n in target_nodes
                  }
    # basically if there are no targets, don't bother
    if float(sum(person_dict.values())) == 0.0:
        protrank_dict = dict.fromkeys(ppi_g.nodes(), 0.0)
    else:
### This takes basically all of the time for the entire method
# scipy speeds things up ~7x, so the above should be reassessed
# note that the very small scores are slightly different with this different method, but only when looking 5-6 digits past the decimal point
        protrank_dict = pagerank_scipy(list(ppi_g), M,
                alpha = float(restart_prob), # Chance of randomly restarting, 0.85 is default
                personalization = person_dict, # dictionary with a key for every graph node and nonzero personalization value for each node
                max_iter = iterations, # Maximum number of iterations in power method eigenvalue solver. default 100
                tol = 1e-6, # Error tolerance used to check convergence in power method solver. default 1e-6
                weight = 'weight', # Edge data key to use as weight. If None weights are set to 1
               )
    return protrank_dict



# We're using networkx's pagerank implementation, but the conversion from
# networkx graph to scipy sparse matrix is very slow in our use-case (90+%
# of the runtime of the original function).
# To address this, the code below takes in a scipy matrix instead of a
# networkx graph.
# ******* CODE BELOW FROM NETWORKX ********
def pagerank_scipy(nodelist, M, alpha=0.85, personalization=None,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None):
    """Returns the PageRank of the nodes in the graph.
    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.
    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.
    alpha : float, optional
      Damping parameter for PageRank, default=0.85.
    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specfiied, a nodes personalization value will be zero.
      By default, a uniform distribution is used.
    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.
    tol : float, optional
      Error tolerance used to check convergence in power method solver.
    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.
    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.
    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value
    Examples
    --------
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = nx.pagerank_scipy(G, alpha=0.9)
    Notes
    -----
    The eigenvector calculation uses power iteration with a SciPy
    sparse matrix representation.
    This implementation works with Multi(Di)Graphs. For multigraphs the
    weight between two nodes is set to be the sum of all edge weights
    between those nodes.
    See Also
    --------
    pagerank, pagerank_numpy, google_matrix
    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.
    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
    """
    import scipy.sparse

    assert len(nodelist) == M.shape[0]
    N = len(nodelist)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # initial vector
    x = scipy.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        p = scipy.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling.get(n, 0) for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)
# *********** END NETWORKX PASTE ***************
