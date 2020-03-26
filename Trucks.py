#卡车类
import copy
import numpy as np
#WaitingTime  Loading+SpottingTime Hauling  Queuing    Backing+Tipping  Traveling
#TW             TL      TS         TH            TQ         TB      TP       TT

class Truck:

    def __init__(self,dataMat,type=1,id=0):  #dataMat是二维的，每一行对应不同的时间
        if len(dataMat)==0:
            self.type = 0;
            self.Id=id;
            self.circleTime=0;
            self.minpathid=0;
            return
        self.TW = dataMat[0][0]    #当前路径上的耗时，只有一条路时使用
        self.TL = dataMat[0][1]
        self.TS = dataMat[0][2]
        self.TH = dataMat[0][3]
        self.TQ = dataMat[0][4]
        self.TB = dataMat[0][5]
        self.TP = dataMat[0][6]
        self.TT = dataMat[0][7]
        if len(dataMat)>8:
            self.type=dataMat[0][8]
        else:
            self.type=type
        self.circleTime=[]                          #转一圈的时间
        self.minpathid=0                          #花费最少时间的路标号

        self.allpathtime=copy.deepcopy(dataMat)   #copy不同阶段所耗费 ，导致数组多行（维）
        #其他属性
        temp=float('inf')
        for i in range(0,len(self.allpathtime)):   #self.TH+ self.TQ + self.TB + self.TP + self.TT
            circletimetemp= self.allpathtime[i][3] +self.allpathtime[i][4]+self.allpathtime[i][5]+self.allpathtime[i][6]+self.allpathtime[i][7]
            self.circleTime.append(circletimetemp)
            if temp>circletimetemp:
                temp=circletimetemp
        self.minpathid=self.circleTime.index(temp)
        self.Id = id  # 车辆编号
        self.oilperkilo = 0  # 每公里耗油
        self.turns = 0       # 已经跑了几轮
        self.pathnum=1       # 当前有几条路可供选择
        self.curpath=1       # 当前选择第几条路
        self.wholekilo=0     # 总里程数
        self.cur_localwaitingtime=0      #自我等待时间,在truckqueue中计算
        self.wholewaitingtime=0          #总体等待时间,在truckqueue中计算
        self.initwaitingtie=0            #初始队列时的等待时间

     #获取当前类型卡车转一圈的时间，不考虑dump的排队
    def getCircleTime(self):
        return self.circleTime
    def getpathid(self):
        return self.minpathid
    def getID(self):
        return self.Id

class TrucksSet:
    def __init__(self, trucktypeset,truckdata,pathNum,shovelNum):#根据车型来构建
        self.trucksset=[]
        for i in range(0, len(trucktypeset)):
            truckdatas = []
            trucktypetemp = (int)(trucktypeset[i])   #获取type value
            for j in range(0, pathNum):  # 建立不同path上的时间数组
                baserow = trucktypetemp * pathNum - 1
                truckdatas.append(truckdata[baserow + j])  #获取车辆巡游数据
                trucks = Truck(truckdatas, trucktypetemp, i+1)
                self.trucksset.append(trucks)
        if (len(trucktypeset)%shovelNum!=0):   #补充空数据
            xtmp=shovelNum-len(trucktypeset)%shovelNum
            for k in range(0,xtmp):
                trucks = Truck([], 0, len(trucktypeset)+k)
                self.trucksset.append(trucks)
    def gettrucksets(self):
        return  self.trucksset
    def __len__(self):
        return len(self.trucksset)