from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n=8

y1=np.empty([n,n])
y2=np.empty([n,n])

if rank == 0:
    x=np.linspace(-1,1,n)
    x1,x2=np.meshgrid(x,x)
    a1=x1/np.sqrt(x1*x1+x2*x2)
    a2=x2/np.sqrt(x1*x1+x2*x2)
else:
    x1=np.empty([n,n])
    x2=np.empty([n,n])
    a1=np.empty([n,n])
    a2=np.empty([n,n])

comm.Bcast([x1,MPI.DOUBLE])
comm.Bcast([x2,MPI.DOUBLE])
comm.Bcast([a1,MPI.DOUBLE])
comm.Bcast([a2,MPI.DOUBLE])


comm.Barrier()

mysize=n/comm.size

# each core should work out a slice of size (mysize,n)
myoffset=comm.rank*mysize

y1[myoffset:myoffset+mysize,:]=x1[myoffset:myoffset+mysize,:]#-a1[myoffset:myoffset+mysize,:]
y2[myoffset:myoffset+mysize,:]=x2[myoffset:myoffset+mysize,:]#-a2[myoffset:myoffset+mysize,:]

'''
if rank == 0:
    y1[myoffset:myoffset+mysize,:]=x1[myoffset:myoffset+mysize,:]#-a1[myoffset,myoffset+mysize]
    y2[myoffset:myoffset+mysize,:]=x2[myoffset:myoffset+mysize,:]#-a2[myoffset,myoffset+mysize]
    plt.plot(y1[myoffset:myoffset+mysize,:],y2[myoffset:myoffset+mysize,:],'o')
    plt.xlim([-1.05,1.05])
    plt.ylim([-1.05,1.05])
    plt.show()
'''

comm.Barrier()
yy1=np.array(comm.gather(y1,root=0))
yy2=np.array(comm.gather(y2,root=0))
print type(yy1),rank

if rank == 0:
    fig,axis=plt.subplots(1,1,figsize=(10,10))
    axis.plot(yy1[0,:,:],yy2[0,:,:],'o',color='red')
    axis.plot(yy1[1,:,:],yy2[1,:,:],'o',color='blue')
    axis.plot(yy1[2,:,:],yy2[2,:,:],'o',color='green')
    axis.plot(yy1[3,:,:],yy2[3,:,:],'o',color='orange')
    axis.set_xlim([-1.05,1.05])
    axis.set_ylim([-1.05,1.05])
    plt.show()


if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
    print 'This is rank ',rank
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    
