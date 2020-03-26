
import random

from Trucks import Truck
from PSOModel import *


#----------------------PSO参数设置---------------------------------使用离散PSO


class DISCRETEPSO():

    def __init__(self,pN,dim,max_iter,truckset,objectfun):
        #set parameters
        self._popSize = numOfParticles = pN
        self._dimensions = dimensions = dim
        self._generations  = max_iter #?200
        self.w = 1 #0.72984 #0.8
        self.c1 = 2
        self.c2 = 2
        self.r1= 0.6
        self.r2=0.3
        self.truckset = truckset
        self.objectiveFun=objectfun

        # Swarm Initialization
        self._swarm=SwarmModel(objectfun)
        self._swarm.initpaticleparas(self.w,self.c1,self.c2,self.r1,self.r2)
        self._swarm.initSwarm( truckset,numOfParticles, dimensions)

#----------------------迭代运算----------------------------------
    def iterator(self):
        for t in range(self._generations):
            x=self._swarm.updateSwarm()
            self._swarm.drawpic()
            print(x)




#-------------------画图--------------------
# plt.figure(1)
# plt.title("Figure1")
# plt.xlabel("iterators", size=14)
# plt.ylabel("fitness", size=14)
# t = np.array([t for t in range(0,100)])
# fitness = np.array(fitness)
# plt.plot(t,fitness, color='b',linewidth=3)
# plt.show()