import torch as th
import networkx as nx
from itertools import product
from .cropGraph import nodeInside,cropGraph,cropGraph_dontCutEdges
import torch.autograd as autograd


def getCropCoords(shp,cropsz,margin):
    # shp is the size of a tensor to be divided into crops of size at most cropsz
    # it is an iterable of ints
    # cropsz is maximum size of a crop, an iterable of ints
    # margin is the size of margins added to the sides of the crop
    # margin can be an iterable of ints or an int
    # it is used for cropping graphs, rather than tensors, as the crops with margins will go outside shp
    #
    # returns a list of lists of slice objects,
    # each list of slice objects defines a crop of img
    # of size at most cropsz 
    
    # first infer crop start and stop indices
    croplims=[]
    if not hasattr(margin,'__len__'): margin=[margin]*len(shp)
    for sz,cs,m in zip(shp,cropsz,margin):
        croplims.append([slice(a-m,min(a+cs,sz)+m) for a in range(0,sz,cs)])
    
    # then generate all crops as a product of indices across dimensions
    clims=[list(a) for a in product(*croplims)]
    
    return clims

# todo: change to sparse matrix
def nodeEdgePosMat(g,nodes,n2i):
    # returns a matrix that lets us compose the edge position matrix from the node position matrix 
    #    by matrix multiplication
    # g is an nx graph
    # nodes is a tensor of node positions, size k X d, where k is the number of nodes and d indexes dimensions
    # n2i is a mapping from nodes of graph g to indexes of the first dimension of tensor nodes
    # returns a tensor t of size k x n x 2 where n is the number of edges of graph g
    #       t[i][j][0]=1 if node i is the "left"  node of edge j, and t[i][j][0]=0 otherwise
    #       t[i][j][1]=1 if node i is the "right" node of edge j, and t[i][j][1]=0 otherwise
    
    k=nodes.shape[0]
    n=len(g.edges)
    if n > 0:
        t=th.zeros((k,n,2),dtype=nodes.dtype)
        for i,(u,v) in zip(range(n),g.edges):
            t[n2i[u]][i][0]=1.0
            t[n2i[v]][i][1]=1.0
    else:
        t=th.zeros((k,1,2),dtype=nodes.dtype)
        t[n2i[list(g.nodes)[0]]][0][0]=1.0
        t[n2i[list(g.nodes)[0]]][0][1]=1.0
        
    return t.to(nodes.device)

def closest_points_on_edges(edges,points):
    # edges is a tensor of size n x 2 x d
    # where n is the number of edges and d is the dimension of the space, 2 or 3
    # points is a tensor of size k x d
    # returns
    # alpha, a tensor of size n x k, such that, for
    #     a=alpha[i][j], s=edges[i][0][:], t=edges[i][1][:], p=points[j][:],
    #     q=(1-a)*s+a*t is the point on line section (s,t) that is closest to p among all points on (s,t)
    # closestPointOnEdge, a tensor of size n x k x d, closestPointOnEdge[i][j][:] contains q (see above)
    # dist2, a tensor of size n x k, containing the Euclidean distance between p and q
    # grad, a tensor of size n x k x 2 x d
    #       grad[i][j][0][:] is the gradient of the distance between edge i and node j 
    #                     with respect to the first node of the edge
    #       grad[i][j][1][:] is the gradient of this distance wr to the second node
    
    eps=1e-10
    
    s=edges[:,0,None,:]
    t=edges[:,1,None,:]
    pts=points[None,:,:]

    # the formulas follow from an analytical solution to a quadratic equation.
    # in two dimensions, where (s,t) forms the edge and p is the point,
    # the optimal alpha
    #    alpha=[(-px+sx)(sx-tx)+(-py+sy)(sy-ty)]/[(sx-tx)^2+(sy-ty)^2],
    # 

    stdif=s-t
    alpha_den=th.sum(stdif**2,dim=-1) # shape n x 1
    # handle nans for alpha_den=0
    alpha_den.masked_fill_(alpha_den<eps,1.0)
    alpha_num=th.sum((-pts+s)*stdif,dim=-1) # shape n x k
    alpha=alpha_num/alpha_den
    alpha.clamp_(0.0,1.0)

    # q=s*(1-alpha)+t*alpha
    closestPointOnEdge=s-stdif*alpha[:,:,None]  # shape n x k x d
    diff=closestPointOnEdge-pts # size n x k x d
    d=th.norm(diff,dim=-1) # size n x k
    
    dist,inds=th.min(d,dim=0)
    
    # dist is size k
    # inds is size k
    # diff is size n x k x d
    # d is size n x k
    # alpha is size n x k
    return dist,inds,diff,d,alpha

def closest_points_on_edges_grad(inds,diff,d,alpha):
    # for q=\alpha t + (1-\alpha) s,
    # gradient wrt t is (q-p)   \alpha /(dist2)
    # and,     wrt s,   (q-p)(1-\alpha)/(dist2)
    gr=diff/d[:,:,None] # size n x k x d
    gr.masked_fill_(gr.isnan(),0.0) 
    ag=gr*alpha[:,:,None]
    grad=th.stack([gr-ag,ag],dim=-2) # size n x k x 2 x d
    
    # zero the gradient for non-min distances
    mask=th.ones_like(alpha,dtype=th.bool)
    mask.scatter_(0,inds[None,:],0)
    grad.masked_fill_(mask[:,:,None,None],0.0)
    
    return grad

def renderGraphAsDistFun_wGrad(g,nodes,n2i,crop):
    # g is an nx graph
    # nodes is a tensor of size k x d, where k is the number of nodes and d number of dimensions
    # n2i is a mapping from nodes to the first dimension of tensor "nodes"
    # crop is an iterable of slice objects
    # it is used to generate the locations of all voxels/pixels within the crop
    
    # get the matrix of edges n x 2 x d
    ne=nodeEdgePosMat(g,nodes,n2i)
    e=th.einsum('kd,kne->ned',[nodes,ne])
    
    # get the matrix of points k x d
    pts=[list(range(c.start,c.stop)) for c in crop]
    pts=th.tensor(list(product(*pts)),dtype=nodes.dtype,device=nodes.device)
    
    # compute distances
    dist,inds,diff,d,alpha=closest_points_on_edges(e,pts)
    distedgegrad=closest_points_on_edges_grad(inds,diff,d,alpha)
    
    dngrad=th.einsum('kne,nmed->kmd',[ne,distedgegrad])
    
    nnodes=nodes.shape[0]
    dims=nodes.shape[-1]
    dist    =dist    .reshape([c.stop-c.start for c in crop])
    dngrad  =dngrad  .reshape([nnodes]+[c.stop-c.start for c in crop]+[dims])
        
    return dist,dngrad

def renderGraphAsDistFun_noGrad(g,nodes,n2i,crop):
    # g is an nx graph; we use its adjacency matrix to create the node-edge mapping matrix
    # nodes is a tensor of size k x d, where k is the number of nodes and d number of dimensions
    # n2i is a mapping from nodes to the first dimension of tensor "nodes"
    # crop is an iterable of slice objects
    # it is used to generate the locations of all voxels/pixels within the crop
    
    # get the matrix of edges n x 2 x d
    ne=nodeEdgePosMat(g,nodes,n2i)
    e=th.einsum('kd,kne->ned',[nodes,ne])
    
    # get the matrix of points k x d
    pts=[list(range(c.start,c.stop)) for c in crop]
    pts=th.tensor(list(product(*pts)),dtype=nodes.dtype,device=nodes.device)
    
    # compute distances
    dist,inds,diff,d,alpha=closest_points_on_edges(e,pts)
    dist=dist.reshape([c.stop-c.start for c in crop])
        
    return dist

def renderDistBig_wGrad(gr,nodes,n2i,size,cropsz,dmax,maxedgelen=None):
    dist=th.zeros(tuple(size),dtype=nodes.dtype,device=nodes.device)
    grad=th.zeros(tuple([nodes.shape[0]]+list(size)+[nodes.shape[-1]]),dtype=nodes.dtype,device=nodes.device)
    if maxedgelen is None:
        margin=dmax
    else:
        margin=(dmax+0.5*maxedgelen)/(2.0**0.5)
    gcropinds=getCropCoords(size,cropsz,margin)
    gcrops=[cropGraph_dontCutEdges(gr,ci) for ci in gcropinds]
    dcropinds=getCropCoords(size,cropsz,0)
    for g,di in zip(gcrops,dcropinds):
        if len(g.nodes)==0:
            dist[di]=dmax
        else:
            dist[di],grad[[slice(None)]+di+[slice(None)]]=renderGraphAsDistFun_wGrad(g,nodes,n2i,di)
            
    dist.clamp_(max=dmax)
    m=dist>dmax
    grad.masked_fill_(m.unsqueeze(0).unsqueeze(-1),0.0)
    return dist,grad

def renderDistBig(gr,nodes,n2i,size,cropsz,dmax,maxedgelen=None):
    # gr is an nx graph
    # nodes is a tensor of size k x d
    # size is an iterable of ints, the size of the distance volume
    # n2i is a mapping between the nodes of graph 'gr' and indices of tensor 'nodes'
    # cropsz is an iterable of ints, defining crop size to use when rendering the distance map
    # dmax is the value at which the distance will be capped
    # maxedgelen is the maximum expected edge length in the graph
    # (This is used to compute margins when cropping the graph)
    dist=th.zeros(tuple(size),dtype=nodes.dtype,device=nodes.device)
    # edges that do not have end points inside the crop 
    # can affect the distance map within the crop;
    # knowing maximum edge length, and the distance cap,
    # we can compute additional margin for croppigng the graph 
    if maxedgelen is None:
        margin=dmax
    else:
        margin=(dmax+0.5*maxedgelen)/(2.0**0.5)
    gcropinds=getCropCoords(size,cropsz,margin)
    gcrops=[cropGraph_dontCutEdges(gr,ci) for ci in gcropinds]
    dcropinds=getCropCoords(size,cropsz,0)
    for g,di in zip(gcrops,dcropinds):
        if len(g.nodes)==0:
            dist[di]=dmax
        else:
#             try:
            dist[di]=renderGraphAsDistFun_noGrad(g,nodes,n2i,di)
#             except:
#                 dist[di]=dmax
            
    dist.clamp_(max=dmax)
    return dist

def cmptExtEnergyEuclDist_wGrad(distim,gr,nodes,n2i,size,cropsz,dmax,maxedgelen=None):
    if maxedgelen is None:
        margin=dmax
    else:
        margin=(dmax+0.5*maxedgelen)/(2.0**0.5)
    gcropinds=getCropCoords(size,cropsz,margin)
    gcrops=[cropGraph_dontCutEdges(gr,ci) for ci in gcropinds]
    dcropinds=getCropCoords(size,cropsz,0)
    exte=0.0
    grad_tot=th.zeros_like(nodes)
    for g,di in zip(gcrops,dcropinds):
        distim_c=distim[di]
        if len(g.nodes)==0:
            exte+=th.norm(distim_c-dmax)
        else:
            dim_pred,grad=renderGraphAsDistFun_wGrad(g,nodes,n2i,di)
            dim_pred.clamp_(max=dmax)
            m=dim_pred>dmax
            grad.masked_fill_(m.unsqueeze(0).unsqueeze(-1),0.0)
            diff=dim_pred-distim_c
            diff_=diff.unsqueeze(0).unsqueeze(-1)
            grad=th.nansum((grad*diff_),dim=tuple(range(1,diff.dim()+1)))
            exte+=th.norm(diff)
            grad_tot+=grad
            
    return exte,grad_tot

def euclDist_inCrops_(distim,gr,nodes,n2i,cropsz,dmax,maxedgelen=None):
    # compute Euclidean distance between distim and a distance map generated
    # from the graph gr, and the gradients with respect to node positions,
    # and to distim
    # distim - input distance map; a 2D or 3D tensor;
    # gr - a networkx graph; each node has an attribute "pos", a 2- or 3-vector
    # nodes - a tensor of size k x d, where k is the number of nodes of gr
    #         and d is 2 or 3; the tensor contains the positions of nodes of gr.
    # n2i - mapping from nodes of gr to indices of tensor node
    # cropsz - an iterable of int of length d; the size of crops to use
    # dmax - the clamping value for the distance map
    # maxedgelen - the maximum length of edge expected in the graph;
    if maxedgelen is None:
        margin=dmax
    else:
        margin=(dmax+0.5*maxedgelen)/(2.0**0.5)
    size=distim.shape
    dcropinds=getCropCoords(size,cropsz,0)
    gcropinds=getCropCoords(size,cropsz,margin)
    gcrops=[cropGraph_dontCutEdges(gr,ci) for ci in gcropinds]
    exte=0.0
    grad_tot=th.zeros_like(nodes)
    distim_g=th.zeros_like(distim)
    for g,di in zip(gcrops,dcropinds):
        distim_c=distim[di]
        distim_g_c=distim_g[di]
        if len(g.nodes)==0:
            exte+=th.norm(distim_c-dmax)
        else:
            dim_pred,grad=renderGraphAsDistFun_wGrad(g,nodes,n2i,di)
            dim_pred.clamp_(max=dmax)
            m=dim_pred>dmax
            grad.masked_fill_(m.unsqueeze(0).unsqueeze(-1),0.0)
            diff=dim_pred-distim_c
            distim_g_c.add_(diff,alpha=-2.0)
            #distim_g[di].add(diff,alpha=-2.0)
            diff_=diff.unsqueeze(0).unsqueeze(-1)
            grad=th.nansum((2*grad*diff_),dim=tuple(range(1,diff.dim()+1)))
            exte+=th.norm(diff)
            grad_tot.add_(grad)
            
    return exte,grad_tot,distim_g

class euclDist_cropByCrop(autograd.Function):
    @staticmethod
    def forward(ctx, distim,gr,nodes,n2i,cropsz,dmax,maxedgelen=None):
        e,g_nodes,g_distim=euclDist_inCrops_(distim,gr,nodes,n2i,cropsz,dmax,maxedgelen)
        ctx.g_nodes=g_nodes
        ctx.g_distim=g_distim
        return e

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.g_distim,None,ctx.g_nodes,None,None,None,None

euclDist_inCrops=euclDist_cropByCrop.apply
