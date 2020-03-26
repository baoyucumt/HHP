'''
this method is the first easy one.
'''
from Trucks import Truck
import numpy as np
from TruckQueue import *
import time


#最小化卡车等待时间
class MTWT:
    def __init__(self,trucksset,shoovelNum=1,pathnum=1):
        self.trucksset=trucksset                  #卡车初始队列
        self.shoovelNum=shoovelNum
        self.sumofwaitingTime=0                   #等待时间，目标
        self.cur_tru_wt=0                         #current_truck_waitingtime
        self.truckQueue=[shoovelNum,]               #等待队列
        self.runQueue=[]                            #在path上的运行队列
        self.pathQueue=[pathnum]                    #不同path上的队列
        self.errorflag=0

    #计算整个队列需要等待的时间，queueMat车类型的队列矩阵，shovels类型一致位置不动，cycles表示整个队列循环次数，只有1条路的情况
    def compute_AllQueuewaitingtime(self,queueMat,shovelnum,cycles=1):
        #运行到有车辆开始返回
        tq = TrucksWaitingQueue(self.trucksset, queueMat)    #创建车辆等待队列
        rq = TrucksRunningQueue(self.trucksset, self.shoovelNum, 1)  #运行队列，1条路
        x=[]
        while len(x)==0:     #运行到第一辆车返回
            outqueue=tq.dequeue()
            if(len(outqueue)==0):   #不能出队，第一辆车没返回，铲车空
                print("有等待的铲车了")
                return 0
            x=rq.dispatchtruck(outqueue)
            #print("x value and len",x, len(x))
        '''
        #计算等待时间，waitingtime为总体等待时间
        '''
        waitingtime=0   #获取第一辆车出运行期的等待时间
        if x[0][1]<=0:
            waitingtime+=abs(x[0][1])
        else:
            print("等待时间计算错误！")
        truckcount=0           #对车辆进行计数
        while truckcount<=queueMat.size*cycles:    #不管车辆是否是原车辆，直接计数目，总数对了就算一轮
            for i in range(0,len(x)):
                if(len(x)>=3):
                    print("x is ",x[i][0].Id)
                mm=tq.enqueue(x[i][0])    #入队返回
                #self.trucksset[x[i][1]]
                waitingtime=waitingtime+mm+abs(x[i][1])    #总体等待时间
            outqueue=tq.dequeue()       #新出发车辆
            if (len(outqueue)==0) and tq.errorflag==1:
                print("wtimetable",tq.timetable)
                print("rqueue",rq.runningqueue)
                self.errorflag=1
                return -1
            x=rq.dispatchtruck(outqueue)
            if truckcount+len(x)>queueMat.size*cycles:    #对超出数量的处理，
                #x.sort(axis=1)
                sorted(x,key=lambda x:x[1])   #按照？排序
                for j in range(0,queueMat.size*cycles-truckcount):
                    waitingtime=waitingtime+abs(x[j][1])

            if(outqueue.__len__()>0):
                truckcount+=len(outqueue[0])
        return  waitingtime



