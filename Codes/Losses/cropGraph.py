import numpy as np
import networkx as nx
from networkx.classes.graphviews import subgraph_view

def nodeInside(pos,crop):
    # pos i an np-array containing the dimensions of a single point in k dimensions
    # crop is a tuple of slice objects, defining a crop of k-dimensional space
    # the function returns True if pos lies inside crop, using pytorch index arithmetics
    # (i.e., location 29.5 is outside of array of size 30, but location 0 is inside)
    
    assert len(pos)==len(crop)
    for p,l in zip(pos,crop):
        if p<l.start or p>=l.stop:
            return False
    return True
    
def cropGraph(G,crop):
    # G is a nx.Graph, whose nodes have attributes called "pos";
    # each of these attributes is an np.array
    # and determines a position of the node
    # (the coordinates go in the standard order: 0th, 1st, 2nd, etc)
    # G.nodes[n]["pos"]==np.array([1,2,3]) means node n is at position 1,2,3
    #
    # crop is a tuple of slice objects
    #
    # this function returns another graph H,
    # which contains the nodes of G that lie inside the crop,
    # and, for each edge of G that crosses crop boundary,
    # a node lying on the crop boundary,
    # and connected to the end of the edge that is inside the crop;
    # these "boundary nodes" have an attribute called "fixedDim",
    # set to the index of the dimension perpendicular to the crop boundary that they traverse
    H=G.copy()
    nodes2delete=[]
    boundaryNodes=[]
    maxind=0
    for n in G.nodes:
        maxind=max(maxind,n)
        p=np.array(G.nodes[n]["pos"])
        if not nodeInside(p,crop):
            nodes2delete.append(n)
        else:
            # for each edge that goes outside of the crop, establish a new node
            # at the point where the edge crosses the crop boundary
            for m in G[n]:
                q=np.array(G.nodes[m]["pos"])
                if not nodeInside(q,crop):
                    # find the position at which the edge cuts the crop boundary
                    a=1.0
                    dim=0
                    for pp,qq,l,ind in zip(p,q,crop,range(len(crop))):
                        b=2.0
                        if qq<l.start:
                            b=(l.start-pp)/(qq-pp)
                        elif qq>l.stop-1:
                            b=(l.stop-1 -pp)/(qq-pp)
                        if b<a:
                            a=b
                            dim=ind
                    inters=a*q+(1-a)*p
                    boundaryNodes.append((inters,n,dim)) 
            H.nodes[n]["pos"] = p  - np.array([c.start for c in crop])
    H.remove_nodes_from(nodes2delete)
    
    newnode=maxind
    for position,ind,dim in boundaryNodes:
        newnode=newnode+1
        H.add_node(newnode,pos=position  - np.array([c.start for c in crop]),fixedDim=dim)
        H.add_edge(newnode,ind)
        
    return H

def cropGraph_dontCutEdges(G,crop):
    # returns a crop (view) of G that contains all nodes within "crop"
    # and their neighbors
    # G is a networkx graph
    # crop is an iterable of slice objects
    nodes2keep=set()
    for n in G.nodes:
        p=G.nodes[n]["pos"]
        if nodeInside(p,crop):
            nodes2keep.add(n)
            for m in G[n]:
                nodes2keep.add(m)
    H=subgraph_view(G,filter_node=lambda n: n in nodes2keep)

    return H
