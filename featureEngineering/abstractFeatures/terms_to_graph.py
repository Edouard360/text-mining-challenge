import itertools

import igraph


def terms_to_graph(terms, w):
    # This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    # Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'

    w = min(w, len(terms))

    from_to = {}

    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = []

    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))

    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i - w + 1):(i + 1)]

        # edges to try
        candidate_edges = []
        for p in range(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            # if not self-edge
            if try_edge[1] != try_edge[0]:

                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(sorted(set(terms)))

    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())

    # set edge and vertice weights
    g.es['weight'] = list(from_to.values())  # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=list(from_to.values()))  # weighted degree

    return (g)


def compute_node_centrality(graph):
    # degree
    degrees = graph.degree()
    degrees = [round(float(degree) / (len(graph.vs)), 5) for degree in degrees]

    # weighted degree
    w_degrees = [vertex["weight"] for vertex in graph.vs]
    w_degrees = [round(value, 5) for value in w_degrees]

    # closeness
    closeness = graph.closeness(normalized=True)
    closeness = [round(value, 5) for value in closeness]

    # weighted closeness
    w_closeness = graph.closeness(normalized=True, weights=graph.es["weight"])
    w_closeness = [round(value, 5) for value in w_closeness]

    return (zip(graph.vs["name"], degrees, w_degrees, closeness, w_closeness))
