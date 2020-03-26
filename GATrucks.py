from ObjectiveFunction import FitNESSFUN as fitfun
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from Trucks import TrucksSet
import multiprocessing
import time

class GAGene:

    '''
    individual of genetic algorithm
    '''

    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):   #create on chrome
        '''
        generate a random chromsome for genetic algorithm
        用来调节车辆类型
        '''
        len = self.vardim
        rnd = np.random.randint(2,size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):           #将元素置于bound内
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self,trucksset,shoovelNum,pathNum,cycles):
        '''
        calculate the fitness of the chromsome
        '''
        #self.fitness = obj.GrieFunc(
        #    self.vardim, self.chrom, self.bound)
        fitness=fitfun("MTWTAVE")   #为一队车一轮平均时间之和
        fitness.setParas(trucksset,shoovelNum,pathNum,cycles)
        self.fitness=fitness.getobjectfunvalue()


class GeneticAlgorithm:

    '''
    The class for genetic algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN,params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables用来限制车辆类型
        MAXGEN: termination condition,循环次数
        param: algorithm required parameters,
        it is a list which is consisting of crossover rate, mutation rate, alpha(交叉
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []    #记录type
        self.truckspopulation=[] #记录trucks
        self.idpopulation=[]     #记录truck id
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.trucksset=[]
        self.params=params

    def setTrucksPara(self,Truckdata,pathNum,shovelNum,cycles):
        self.truckdata = Truckdata
        self.pathNum=pathNum
        self.shovelNum=shovelNum
        self.cycles=cycles
        return

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = GAGene(self.vardim, self.bound)
            ind.generate()
            trucksset=TrucksSet(ind.chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
            #gene is  a set of trucks
            self.population.append(ind)
            self.idpopulation.append(self.generateIdbyTruckset(trucksset))
            self.truckspopulation.append(trucksset)

    def generateIdbyTruckset(self,Truckset):
        #针对车辆ID序列进行交换,实际上先分配车型，然后在匹配id
        idset=[]
        for i in range(0, len(Truckset)):
            idset.append(Truckset[i].Id)
        return idset

    def evaluate(self):
        '''
        evaluation of the population fitnesses，对应相同的trucksset
        '''
        for i in range(0, self.sizepop):     #分别计算fitness
            #self.population[i].calculateFitness()
            if len(self.truckspopulation[i]) < 10:
                print("ok")
            self.population[i].calculateFitness(self.truckspopulation[i], \
                                self.shovelNum,self.pathNum,self.cycles)   #直接计算fitness，转成type矩阵计算
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        '''
        evolution process of genetic algorithm
        '''
        st = time.time()
        self.t = 0
        self.initialize()
        self.evaluate()
        #best = np.max(self.fitness)
        best = np.min(self.fitness)
        #bestIndex = np.argmax(self.fitness)   #find max value's index
        bestIndex = np.argmin(self.fitness)   #find minmum value's index
        self.best = copy.deepcopy(self.population[bestIndex])
        self.bestQ=copy.deepcopy(self.truckspopulation[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.bestold =self.best.fitness  #同第一代比较
        #self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness  #跟踪存储结果
        #self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        self.trace[self.t, 0] = (self.bestold - self.best.fitness) / self.best.fitness  #跟踪存储结果
        self.trace[self.t, 1] = (self.bestold - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.selectionOperation()   #
            self.crossoverOperation()   #
            self.mutationOperation()    #
            self.evaluate()
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            if best < self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
                self.bestQ = copy.deepcopy(self.truckspopulation[bestIndex])
            self.avefitness = np.mean(self.fitness)
            #self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            #self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            self.trace[self.t, 0] = (self.bestold - self.best.fitness) / self.best.fitness  # 跟踪存储结果
            self.trace[self.t, 1] = (self.bestold - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        et=time.time()
        print("time consumed: " ,et-st)
        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        print("fitness is ",self.bestold," and ",self.best.fitness)
        self.printResult()

    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        newpopid=[]
        newpoptrucks=[]
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))

        sum1 = 0.      #累加 self.fitness[i] / totalFitness,形成区段，然后概率选择
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]

        for i in range(0, self.sizepop):      #按照随机选择，用fitness约束概率
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):   #选择要替换的pop
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
            newpoptrucks.append(self.truckspopulation[idx])
            newpopid.append(self.generateIdbyTruckset(self.truckspopulation[idx]))
        self.population = newpop
        self.idpopulation=newpopid
        self.truckspopulation=newpoptrucks

    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        newpoptrucks = []
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)     #随机选择两个交叉
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()
            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):    #newpop[i]就是population[i]
                    newpop[i].chrom[j] = round(newpop[i].chrom[j] * self.params[2] + \
                        (1 - self.params[2]) * newpop[i + 1].chrom[j])
                    newpop[i + 1].chrom[j] = round(newpop[i + 1].chrom[j] * self.params[2] + \
                        (1 - self.params[2]) * newpop[i].chrom[j])
            trucksset1 = TrucksSet(newpop[i].chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
            newpoptrucks.append(trucksset1)
            trucksset2 = TrucksSet(newpop[i+1].chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
            newpoptrucks.append(trucksset2)
        self.truckspopulation=newpoptrucks    #id暂时不变
        self.population = newpop

    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        newpoptrucks = []
        for i in range(0, self.sizepop):     #突变次数
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = round(newpop[i].chrom[
                        mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos])
                        * (1 - random.random() ** (1 - self.t / self.MAXGEN)))
                else:
                    newpop[i].chrom[mutatePos] = round(newpop[i].chrom[
                        mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) *
                        (1 - random.random() ** (1 - self.t / self.MAXGEN)))
            trucksset1 = TrucksSet(newpop[i].chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
            newpoptrucks.append(trucksset1)
        self.population = newpop
        self.truckspopulation = newpoptrucks  # id暂时不变

    def printResult(self):
        '''
        plot the result of the genetic algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Genetic algorithm for function optimization")
        plt.legend()
        plt.show()

class GeneticAlgorithmByT:

    '''
    The class for genetic algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params,numofThread):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables用来限制车辆类型
        MAXGEN: termination condition,循环次数
        param: algorithm required parameters,
        it is a list which is consisting of crossover rate, mutation rate, alpha(交叉
        '''
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = [x for x in range(numofThread+1)]  # 记录type
        self.truckspopulation = [x for x in range(numofThread+1)]  # 记录trucks
        self.idpopulation = [x for x in range(numofThread+1)]  # 记录truck id
        self.fitness = [x for x in range(numofThread+1)]
        self.trace = np.zeros((self.MAXGEN+1, 2))
        for i in range(0,numofThread+1):   #最后一个是记录最终合起来的结果
            print(i)
            self.population[i] = []  # 记录type
            self.truckspopulation[i] = []  # 记录trucks
            self.idpopulation[i] = []  # 记录truck id
            self.fitness[i] = np.zeros((self.sizepop, 1))
            #self.trace[i] = np.zeros((self.MAXGEN, 2))
        self.trucksset = []
        self.params = params
        self.numofT=numofThread

    def setTrucksPara(self, Truckdata, pathNum, shovelNum, cycles):
        self.truckdata = Truckdata
        self.pathNum = pathNum
        self.shovelNum = shovelNum
        self.cycles = cycles
        return

    def initialize(self):
        '''
        initialize the population
        '''
        for j in range(0,self.numofT):
            for i in range(0, self.sizepop):
                ind = GAGene(self.vardim, self.bound)
                ind.generate()
                trucksset = TrucksSet(ind.chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
                # gene is  a set of trucks
                self.population[j].append(ind)
                self.idpopulation[j].append(self.generateIdbyTruckset(trucksset))
                self.truckspopulation[j].append(trucksset)

    def generateIdbyTruckset(self, Truckset):
        # 针对车辆ID序列进行交换,实际上先分配车型，然后在匹配id
        idset = []
        for i in range(0, len(Truckset)):
            idset.append(Truckset[i].Id)
        return idset

    def evaluate(self,indexofT):
        '''
        evaluation of the population fitnesses，对应相同的trucksset
        '''
        for i in range(0, self.sizepop):  # 分别计算fitness
            # self.population[i].calculateFitness()
            if len(self.truckspopulation[indexofT][i]) < 10:
                print("ok")
            self.population[indexofT][i].calculateFitness(self.truckspopulation[indexofT][i], \
                                                self.shovelNum, self.pathNum, self.cycles)  # 直接计算fitness，转成type矩阵计算
            self.fitness[indexofT][i] = self.population[indexofT][i].fitness

    def solve(self):
        '''
        evolution process of genetic algorithm
        '''
        st = time.time()
        self.t = 0
        self.initialize()
        for i in range(0, self.numofT):
            self.evaluate(i)
        best=[x for x in range(self.numofT + 1)]
        bestIndex = [x for x in range(self.numofT + 1)]
        self.best = [x for x in range(self.numofT + 1)]
        self.bestQ = [x for x in range(self.numofT + 1)]
        self.avefitness = [x for x in range(self.numofT + 1)]
        self.bestold = [x for x in range(self.numofT + 1)]  # 同第一代比较
        for i in range(0, self.numofT ):  # 最后一个是记录最终合起来的结果
            best[i] = np.min(self.fitness[i])
            bestIndex[i] = np.argmin(self.fitness[i])  # find minmum value's index
            self.best[i] = copy.deepcopy(self.population[i][bestIndex[i]])
            self.bestQ[i] = copy.deepcopy(self.truckspopulation[i][bestIndex[i]])
            self.avefitness[i] = np.mean(self.fitness)
            self.bestold[i] = self.best[i].fitness  # 同第一代比较
        self.trace[self.t, 0] = (self.bestold[0] - self.best[0].fitness) / self.best[0].fitness  # 跟踪存储结果
        self.trace[self.t, 1] = (self.bestold[0] - self.avefitness[0]) / self.avefitness[0]
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        self.submaxgen=self.MAXGEN//self.numofT
        t=[x for x in range(self.numofT)]
        for i in range(0,self.numofT):
            t[i] = multiprocessing.Process(target=self.run(i), args=())
            t[i].start()
        for i in range(0, self.numofT):
            t[i].join()
        et = time.time()
        print("time consumed: ", et - st)
        sorted(self.trace,key=lambda trace:trace[1])
        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print("Optimal solution is:")
        self.best[self.numofT]=self.best[0]
        for i in range(1, self.numofT):
            if self.best[i].fitness<self.best[self.numofT].fitness:
                self.best[self.numofT] = self.best[i]
            print("the %d fitness is :",i,self.best[i].fitness)
            print("the old fitness is :", i, self.bestold[i])
        print(self.best[self.numofT].chrom)
        print("fitness is ", self.bestold[self.numofT], " and ", self.best[self.numofT].fitness)
        self.printResult()

    def run(self,baseIndex):
        t=baseIndex*self.submaxgen
        while (t < (baseIndex+1)*self.submaxgen):
            t += 1
            self.selectionOperation(baseIndex)  #
            self.crossoverOperation(baseIndex)  #
            self.mutationOperation(baseIndex)  #
            self.evaluate(baseIndex)
            best = np.min(self.fitness[baseIndex])
            bestIndex = np.argmin(self.fitness[baseIndex])
            if best < self.best[baseIndex].fitness:
                self.best[baseIndex] = copy.deepcopy(self.population[baseIndex][bestIndex])
                self.bestQ[baseIndex] = copy.deepcopy(self.truckspopulation[baseIndex][bestIndex])
            self.avefitness[baseIndex] = np.mean(self.fitness[baseIndex])
            self.trace[t, 0] = (self.bestold[baseIndex] - self.best[baseIndex].fitness) / self.best[baseIndex].fitness  # 跟踪存储结果
            self.trace[t, 1] = (self.bestold[baseIndex] - self.avefitness[baseIndex]) / self.avefitness[baseIndex]
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                t, self.trace[t, 0], self.trace[t, 1]))

    def selectionOperation(self,indexofT):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        newpopid = []
        newpoptrucks = []
        totalFitness = np.sum(self.fitness[indexofT])
        accuFitness = np.zeros((self.sizepop, 1))

        sum1 = 0.  # 累加 self.fitness[i] / totalFitness,形成区段，然后概率选择
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[indexofT][i] / totalFitness
            sum1 = accuFitness[i]

        for i in range(0, self.sizepop):  # 按照随机选择，用fitness约束概率
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):  # 选择要替换的pop
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[indexofT][idx])
            newpoptrucks.append(self.truckspopulation[indexofT][idx])
            newpopid.append(self.generateIdbyTruckset(self.truckspopulation[indexofT][idx]))
        self.population[indexofT] = newpop
        self.idpopulation[indexofT] = newpopid
        self.truckspopulation[indexofT] = newpoptrucks

    def crossoverOperation(self,indexofT):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        newpoptrucks = []
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)  # 随机选择两个交叉
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[indexofT][idx1]))
            newpop.append(copy.deepcopy(self.population[indexofT][idx2]))
            r = random.random()
            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):  # newpop[i]就是population[i]
                    newpop[i].chrom[j] = round(newpop[i].chrom[j] * self.params[2] + \
                                               (1 - self.params[2]) * newpop[i + 1].chrom[j])
                    newpop[i + 1].chrom[j] = round(newpop[i + 1].chrom[j] * self.params[2] + \
                                                   (1 - self.params[2]) * newpop[i].chrom[j])
            trucksset1 = TrucksSet(newpop[i].chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
            newpoptrucks.append(trucksset1)
            trucksset2 = TrucksSet(newpop[i + 1].chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
            newpoptrucks.append(trucksset2)
        self.truckspopulation[indexofT] = newpoptrucks  # id暂时不变
        self.population[indexofT] = newpop

    def mutationOperation(self,indexofT):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        newpoptrucks = []
        for i in range(0, self.sizepop):  # 突变次数
            newpop.append(copy.deepcopy(self.population[indexofT][i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = round(newpop[i].chrom[
                                                           mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[
                        0, mutatePos])
                                                       * (1 - random.random() ** (1 - self.t / self.MAXGEN)))
                else:
                    newpop[i].chrom[mutatePos] = round(newpop[i].chrom[
                                                           mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[
                        mutatePos]) *
                                                       (1 - random.random() ** (1 - self.t / self.MAXGEN)))
            trucksset1 = TrucksSet(newpop[i].chrom, self.truckdata, self.pathNum, self.shovelNum).trucksset
            newpoptrucks.append(trucksset1)
        self.population[indexofT] = newpop
        self.truckspopulation[indexofT] = newpoptrucks  # id暂时不变

    def printResult(self):
        '''
        plot the result of the genetic algorithm
        '''
        x = np.arange(0, self.MAXGEN+1)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Genetic algorithm for function optimization")
        plt.legend()
        plt.show()