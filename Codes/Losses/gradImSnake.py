import numpy as np
from functools import reduce
import torch as th

from .snake import Snake

# Gaussian gradient for snakes

def makeGaussEdgeFltr(stdev,d):
    # make a Gaussian-derivative-based edge filter
    # filter size is determined automatically based on stdev
    # the filter is ready to be used with pytorch conv 
    # input params:
    #   stdev - the standard deviation of the Gaussian
    #   d - number of dimensions
    # output:
    #   fltr, a np.array of size d X 1 X k X k,
    #         where k is an odd number close to 4*stdev
    #         fltr[i] contains a filter sensitive to gradients
    #         along the i-th dimension

    fsz=round(2*stdev)*2+1 # filter size - make the it odd

    n=np.arange(0,fsz).astype(np.float)-(fsz-1)/2.0
    s2=stdev*stdev
    v=np.exp(-n**2/(2*s2)) # a Gaussian
    g=n/s2*v # negative Gaussian derivative

    # create filter sensitive to edges along dim0
    # by outer product of vectors
    shps = np.eye(d,dtype=np.int)*(fsz-1)+1
    reshaped = [x.reshape(y) for x,y in zip([g]+[v]*(d-1), shps)]
    fltr=reduce(np.multiply,reshaped)
    fltr=fltr/np.sum(np.abs(fltr))
    
    # add the out_channel, in_channel initial dimensions
    fltr_=fltr[np.newaxis,np.newaxis]
    # transpose the filter to be sensitive to edges in all directions 
    fltr_multidir=np.concatenate([np.moveaxis(fltr_,2,k) for k in range(2,2+d)],axis=0)
    
    return fltr_multidir

def cmptGradIm(img,fltr):
    # convolves img with fltr, with replication padding
    # fltr is assumed to be of odd size
    # img  is either 2D: batch X channel X height X width
    #             or 3D: batch X channel X height X width X depth
    #      it is a torch tensor
    # fltr is either 2D: 2 X 1 X k X k
    #             or 3D: 3 X 1 X k X k X k
    #      it is a torch tensor
    
    if img.dim()==4:
        img_p=th.nn.ReplicationPad2d(fltr.shape[2]//2).forward(img)
        return th.nn.functional.conv2d(img_p,fltr)
    if img.dim()==5:
        img_p=th.nn.ReplicationPad3d(fltr.shape[2]//2).forward(img)
        return th.nn.functional.conv3d(img_p,fltr)
    else:
        raise ValueError("img should have 4 or 5 dimensions")

def cmptExtGrad(snakepos,eGradIm):
    # returns the values of eGradIm at positions snakepos
    # snakepos  is a k X d matrix, where snakepos[j,:] represents a d-dimensional position of the j-th node of the snake
    # eGradIm   is a tensor containing the energy gradient image, either of size
    #           3 X d X h X w, for 3D, or of size
    #           2     X h X w, for 2D snakes
    # returns a tensor of the same size as snakepos,
    # containing the values of eGradIm at coordinates specified by snakepos
    
    # scale snake coordinates to match the hilarious requirements of grid_sample
    # we use the obsolote convention, where align_corners=True
    scale=th.tensor(eGradIm.shape[1:]).reshape((1,-1)).type_as(snakepos)-1.0
    sp=2*snakepos/scale-1.0
    
    if eGradIm.shape[0]==3:
        # invert the coordinate order to match other hilarious specs of grid_sample
        spi=th.einsum('km,md->kd',[sp,th.tensor([[0,0,1],[0,1,0],[1,0,0]]).type_as(sp).to(sp.device)])
        egrad=th.nn.functional.grid_sample(eGradIm[None],spi[None,None,None],
                                           align_corners=True)
        egrad=egrad.permute(0,2,3,4,1)
    if eGradIm.shape[0]==2:
        # invert the coordinate order to match other hilarious specs of grid_sample
        spi=th.einsum('kl,ld->kd',[sp,th.tensor([[0,1],[1,0]]).type_as(sp).to(sp.device)])
        egrad=th.nn.functional.grid_sample(eGradIm[None],spi[None,None],
                                           align_corners=True)
        egrad=egrad.permute(0,2,3,1)
        
    return egrad.reshape_as(snakepos)

class GradImSnake(Snake):
    # a snake with external energy gradients sampled from a "gradient image"
    
    def __init__(self,graph,crop,stepsz,alpha,beta,ndims,gimg):
        # gimg is the "gradient image"
        # a tensor of size ndim X h X w for 2D snake, or ndim X d X h X w, for 3D
        # gimg[i,h,w] contains the gradient of the external energy
        # with respect to the i-th coordinate of a control point located at (h,w)
        super(GradImSnake,self).__init__(graph,crop,stepsz,alpha,beta,ndims)
        self.gimg=gimg
    
    def cuda(self):
        super(GradImSnake,self).cuda()
        if self.gimg is not None:
            self.gimg=self.gimg.cuda()
        
    def step(self):
        # external gradient for each control point is extracted from gimg
        return super(GradImSnake,self).step(cmptExtGrad(self.s,self.gimg))
    
    def optim(self,niter):
        # update the snake niter times
        if len(self.s) > 0:
            for i in range(niter):
                self.step()
        return self.s

