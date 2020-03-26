#===============================================================================
# @author: Yu Bao
# @organization: CUMT 2017
#
#
# This package contains different objective function of some problems definitions
#  'MTWT'             - minmized time of waiting time

#===============================================================================


#---- Required imports
from MTWT import MTWT

import numpy as np
import pylab as pyl

#---- Generic PSO Problem
class OBJFUN:

    _objectfunction  = None

    def __init__(self,object):
        self._objfun=object

    def setParas(self,trucksset,shovelnum):  #设置将要用到的参数
        self._trucksset=trucksset
        self._shovelnum=shovelnum

    # ---------------------目标函数MTWT函数-----------------------------
    #qmat是车辆队列
    def getobjectfunvalue(self,onepaticle):  #一个粒子onepaticle
        qmat=self.createqmat(onepaticle)
        cycles=20
        if self._objfun is "MTWT":
            self._objectfunction=MTWT(self._trucksset,self._shovelnum)
            sum = self._objectfunction.compute_AllQueuewaitingtime(qmat, len(qmat), cycles)

            return sum/cycles
        elif self._objfun  is "":
            self._objectfunction = None

    def createqmat(self, onepaticle):
        qmat=np.array(onepaticle).reshape(len(self._trucksset),int(len(onepaticle)/len(self._trucksset)))
        qmat = qmat + 1
        return qmat

'''
this is for GA
'''
class FitNESSFUN:
    _objectfunction = None

    def __init__(self, object):
        self._objfun = object

    def setParas(self, trucksset, shovelnum,pathNum,cycles):  # 设置将要用到的参数
        self._trucksset = trucksset
        self._shovelnum = shovelnum
        self._pathNum=pathNum
        self._cycles = cycles

    # ---------------------目标函数MTWT函数-----------------------------
    # qmat是车辆队列
    def getobjectfunvalue(self):  # 一个粒子onepaticle
        qmat = self.createqmat()
        #cycles = 20
        if self._objfun is "MTWT":
            self._objectfunction =MTWT(self._trucksset,self._shovelnum,self._pathNum) # MTWT(self._trucksset, self._shovelnum)
            sum = self._objectfunction.compute_AllQueuewaitingtime(qmat, len(qmat),self._cycles)
            return sum
        elif self._objfun is "MTWTAVE":
            self._objectfunction = MTWT(self._trucksset, self._shovelnum,
                                        self._pathNum)  # MTWT(self._trucksset, self._shovelnum)
            sum = self._objectfunction.compute_AllQueuewaitingtime(qmat, len(qmat), self._cycles)
            return sum / self._cycles
        elif self._objfun is "":
            self._objectfunction = None

    def createqmat(self):
        #qmat = np.array(oneChrom).reshape(len(self._trucksset), int(len(oneChrom) / len(self._trucksset)))
        quelen=len(self._trucksset)+len(self._trucksset)%self._shovelnum
        qmat = np.array(self._trucksset).reshape(self._shovelnum, (int)(quelen / self._shovelnum))
        return qmat