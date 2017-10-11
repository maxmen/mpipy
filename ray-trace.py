'''
A code for ray-tracing from a lens to a source plane, implementing the lens equation

y=x-a(x)

Run with:

mpiexec -n 4 python ray-trace.py -m mpi4py

(to use 4 cores)
'''


from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# number of rays to be shot through the lens plane
n=512

# initialize arrays (ray grid, deflection angles of toy model)
if rank == 0:
    x=np.linspace(-1,1,n)
    x1,x2=np.meshgrid(x,x)
    a1=x1/np.sqrt(x1*x1+x2*x2)
    a2=x2/np.sqrt(x1*x1+x2*x2)
    y1=np.empty([n,n])
    y2=np.empty([n,n])
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

#slice the x,a arrays
xx1=x1[myoffset:myoffset+mysize,:]
xx2=x2[myoffset:myoffset+mysize,:]
aa1=a1[myoffset:myoffset+mysize,:]
aa2=a2[myoffset:myoffset+mysize,:]

# lens equation to map the points on the lens plane to the source plane
yy1=xx1-aa1
yy2=xx2-aa2
plt.imshow(yy1)
plt.title('rank'+str(comm.rank))
plt.show()
comm.Barrier()

# now rank 0 gathers the results from the other nodes
yy1=np.array(comm.gather(yy1,root=0))
yy2=np.array(comm.gather(yy2,root=0))

# copy the results from all nodes into arrays y1, y2 and plot results
if rank == 0:
    for i in range(comm.size):
        y1[i*mysize:(i+1)*mysize,:]=yy1[i,:,:]
        y2[i*mysize:(i+1)*mysize,:]=yy2[i,:,:]

    fig,axis=plt.subplots(1,1,figsize=(10,10))
    axis.plot(y1,y2,'o',color='red')
    axis.plot(y1,y2,'o',color='blue')
    axis.set_xlim([-1.05,1.05])
    axis.set_ylim([-1.05,1.05])
    plt.show()

    
