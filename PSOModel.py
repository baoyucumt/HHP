#===============================================================================
# @author: Yu Bao rewrited
# @organization: CUMT, School of Computer Science, 2017
#
#
# This package contains representations of the following models:
#  'Particle'            - an atomic element
#  'Swarm'               - a set of particles
#  'Neighbourhood'       - particles topology
#  'KnapsackSolution'    - representation for solution of the problem
#  'TSPSolution'         - representation for solution of the problem
#===============================================================================



#===============================================================================
#                             GENERIC MODELS
#===============================================================================
from ObjectiveFunction import OBJFUN
import numpy as np
import scipy.spatial as spp
import matplotlib.pyplot as plt

#---- Particle representation
class ParticleModel:
    _position       = None
    _velocity       = None
    _bestPosition   = None
    _nbBestPosition = None
    _fitness        = None
    _bestfitness    = None
    _objfun         = None
    _typerange      = None
    _nbbestfitness  = None
    _bestfitness = None

    def __init__(self,objectfun):
        self._position       = None  #最佳组合
        self._velocity       = None  #变换速度
        self._bestPosition   = None  #历史最佳
        self._nbBestPosition = None  #全局最佳
        self._fitness        = None  #适应度
        self._bestfitness    =None   #历史最佳的适应度
        self._nbbestfitness    = None  #global最佳的适应度
        self._objfun=OBJFUN(objectfun)    #目标函数
        self.dim=None


    def initparas(self,w,c1,c2,r1,r2):
        self.w=w
        self.c1=c1
        self.c2=c2
        self.r1=r1
        self.r2=r2

    def initParticle(self, dimensions,truckset):
        self.dim=dimensions
        self._typerange=len(truckset)
        self._objfun.setParas(truckset,len(truckset))
        # Create position array
        self._position = np.random.randint(0,self._typerange, size = dimensions)

        if self._position.max()>self._typerange:     #生成数字是否错误？
            print("bigger than 2")
            quit()
        # Create Velocity array
        self._velocity = np.random.randint(0,self._typerange, size = dimensions)
        # Save best Position so far as current Position
        self._bestPosition = self._position
        self.updateFitness()

    def updateFitness(self):
        # Get Differences of vector   ??????
        #hdist = spp.distance.hamming(self._position, self._solution)  #计算海明距离
        hdist=self._objfun.getobjectfunvalue(self._position)
        #print(hdist)
        self._fitness=hdist
        # Save it as best position if its better than previous best
        if self._bestfitness is None:
            self._bestfitness=hdist
        if hdist < self._bestfitness:
            self._bestPosition = np.copy(self._position)
            self._bestfitness = hdist


    def updatePosition(self):
        # VELOCITY NEEDS TO BE CONSTRICTED WITH VMAX
        # Get random coefficients e1 & e2
        # c = 2.5
        self.r1 = np.random.rand()
        self.r2 = np.random.rand()
        vmax = 6
        # Apply equation to each component of the velocity, add it to corresponding position component
        for i, velocity in enumerate(self._velocity):
            # velocity = 0.72984 * (velocity + c * e1 * (model._bestPosition[i] - model._position[i]) + c * e2 * (model._nbBestPosition[i] - model._position[i]))
            #velocity = self.w*velocity + self.c1 * self.r1 * (self._bestPosition[i] - self._position[i]) + \
            #          self.c2 * self.r2 * (self._nbBestPosition[i] - self._position[i])
            velocity = self.w*velocity + self.c1 * self.r1 * (self._bestPosition[i] - self._position[i]) + \
                    self.c2 * self.r2 * (self._bestPosition[i] - self._position[i])

            if abs(velocity) > vmax and abs(velocity) is velocity:
                velocity = vmax
            elif abs(velocity) > vmax:
                velocity = -vmax
            velocity = self.sigmoid(velocity)
            #  print "vel:", velocity
            for j in range(0,self._typerange):
                if self._nbbestfitness-self._bestfitness>(self._bestfitness/self.dim):
                    if np.random.rand(1) < (j + 1) * velocity / self._typerange:  # 离散量
                        self._position[i] = (j+1)%self._typerange
                else:
                    if np.random.rand(1)<(j+1)*velocity/self._typerange:  #离散量
                        self._position[i] = j

            # if np.random.rand(1) < velocity-1:          #这是给二进制用的
            #     self._position[i] = 1
            # else:
            #     self._position[i] = 0
            if self._position.max() > self._typerange:  # 生成数字是否错误？
                print("bigger than 2，22222")
                quit()

    def setgbest(self,gbestpositiion, gbestfit):
        self._nbBestPosition=np.copy(gbestpositiion)
        self._nbbestfitness=gbestfit

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-(x)))

# ---- Swarm representation
class SwarmModel:
    _particles = None
    _neighbourhoods = None
    _bestPosition = None
    _bestPositionFitness = None

    def __init__(self,objectfun):
        self._particles = []
        self._neighbourhoods = None
        self._bestPosition = None
        self._bestPositionFitness = None
        self._objectfun=objectfun

    def initpaticleparas(self,w,c1,c2,r1,r2):
        self.w=w
        self.c1=c1
        self.c2=c2
        self.r1=r1
        self.r2=r2

    def initSwarm(self, truckset, nParticles = 1, dimensions = 1):
        # Create Swarm
        for i in range(nParticles):
            newParticle = ParticleModel(self._objectfun)
            newParticle.initParticle(dimensions,truckset)
            newParticle.initparas(self.w,self.c1,self.c2,self.r1,self.r2)
            self._particles.append(newParticle)

        #self._neighbourhoods = self._neighbourhoodController.initNeighbourhoods(self)
        self.updateSwarmBestPosition()

    def updateSwarmBestPosition(self):
        # Find swarm best position and save it in swarm
        for nb in self._particles:  #self._neighbourhoods:
            #self._neighbourhoods.updateNeighbourhoodBestPosition(nb)
            if self._bestPositionFitness is None or nb._bestfitness < self._bestPositionFitness:
                self._bestPositionFitness = nb._bestfitness
                self._bestPosition = np.copy(nb._bestPosition)
        for curParticle in self._particles:
            curParticle.setgbest(self._bestPosition,self._bestPositionFitness)

    # Update all particles in the swarm
    def updateSwarm(self):
        for curParticle in self._particles:
            print(curParticle._position)
            curParticle.updatePosition()
            print(curParticle._position)
            print(curParticle._fitness)
            curParticle.updateFitness()

        self.updateSwarmBestPosition()
        return self._bestPositionFitness

    def drawpic(self):
        plt.figure(1)
        plt.title("Figure1")
        plt.grid(True)

        plt.ion()
        try:
            i=0
            for curParticle in self._particles:
                plt.scatter(i,curParticle._bestfitness)
                i+=1
            plt.pause(0.01)
            #plt.clf()
        except Exception as err:
            print(err)


# ---- Neighbourhood representation,用来找全局，在图上
class NeighbourhoodModel:
    _particles = []
    _bestPosition = None
    _bestPositionFitness = None

    def __init__(self, particles):
        self._particles = particles
        self._bestPosition = None
        self._bestPositionFitness = None

    def initNeighbourhoods(self, swarm, topology="gbest"):
        if topology is "gbest":
            return [NeighbourhoodModel(swarm._particles)]
        elif topology is "lbest":
            neighbourhoods = []
            for idx, curParticle in enumerate(swarm._particles):
                previousParticle = None
                nextParticle = None
                if idx is 0:
                    # Previous is last, next is next
                    nextParticle = swarm._particles[idx + 1]
                    previousParticle = swarm._particles[len(swarm._particles) - 1]
                elif idx is len(swarm._particles) - 1:
                    # Previous is previous, next is first
                    nextParticle = swarm._particles[0]
                    previousParticle = swarm._particles[idx - 1]
                else:
                    # Previous is previous, next is next
                    nextParticle = swarm._particles[idx + 1]
                    previousParticle = swarm._particles[idx - 1]
                neighbourhoods.append(NeighbourhoodModel([previousParticle, curParticle, nextParticle]))
            return neighbourhoods
