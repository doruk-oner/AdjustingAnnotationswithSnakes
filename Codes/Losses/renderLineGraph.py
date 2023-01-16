import numpy as np
import torch as th
import networkx as nx

def drawLine(lbl,begPoint,endPoint):
    # endPoint and begPoint should be np.arrays
    # lbl is an np.array to which the line is rendered
    d=endPoint-begPoint
    mi=np.argmax(np.fabs(d))
    if d[mi]==0: # beginning and end points the same
        lbl[tuple(begPoint.astype(np.int))]=1
    else:
        coef=d/d[mi] # a vector that points from the current to the next pixel
        sz=np.array(lbl.shape) # an array holding a shape not an array of shape
        numsteps=int(abs(d[mi]))+1
        step=int(d[mi]/abs(d[mi])) # +-1
        for t in range(0,numsteps):
            pos=begPoint+coef*t*step
            if np.all(pos<sz) and np.all(pos>=0):
                lbl[tuple(np.round(pos).astype(np.int))]=1
            else:
                print("warning: reqested point",pos,"but the volume size is",sz)
    return lbl

def renderGraph(t,g,offset=0,maxv=95):
    # t is an np.array
    # g is an nx Graph
    # offset is either an int or an np array of ints
    for e in g.edges:
        drawLine(t,np.clip(g.nodes[e[0]]["pos"]+offset,0,maxv),np.clip(g.nodes[e[1]]["pos"]+offset,0,maxv))
    return t


