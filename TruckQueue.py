'''
* @copyright DIB
* @author Bao Yu
'''
import numpy as np
import copy
from Trucks import Truck


'''
*卡车的等待队列，是一个跟铲车有关的动态数组，qmat存放当前队列中的truck对象
'''
class TrucksWaitingQueue:
    def __init__(self, trucksset, queueMat):
        self.trucknum=len(trucksset)
        #self.trucks=copy.deepcopy(trucksset)
        self.errorflag=0;
        self.trucks = trucksset
        self.trucktype=[]                       #存放type
        truckidtmp=[]                         #存放ID
        for i in range(0, self.trucknum):   #多个铲车的队列，用数组表示,存的是车的类型表示
            self.trucktype.append((trucksset[i]).type)
            truckidtmp.append(trucksset[i].Id)
        self.shovelnum=len(queueMat)
        if self.shovelnum<=0:
            return None
        self.truckqueue = np.array(self.trucktype).reshape(self.shovelnum,len(trucksset)//self.shovelnum)  #change to np
        self.truckid=np.array(truckidtmp).reshape(self.shovelnum,len(trucksset)//self.shovelnum)
        if self.truckqueue.min()<=0 and self.truckqueue.max()==0:
            return None
        if self.truckqueue.max()>len(trucksset):
            print("truck 队列建立错误")
            return None
        self.pointTrucks = np.zeros((self.shovelnum))  # 指向不同队列的指针
        self.pointTruckid = np.zeros((self.shovelnum,len(self.truckqueue[0])))  # 记录id
        self.timetable=np.zeros((self.shovelnum,len(self.truckqueue[0])))   #车辆的装车时间，第一辆表示正在装车，
        # 与qmat 一一对应

        for i in range(0,len(self.truckqueue)):
            for j in range(0,len(self.truckqueue[0])):
                if (queueMat[i][j]).type<=0:
                    self.timetable[i][j] =0
                    self.truckid[i][j]=0
                else:
                    self.timetable[i][j]=(queueMat[i][j]).TL+(queueMat[i][j]).TS
                    self.truckid[i][j] = (queueMat[i][j]).Id

    def dequeue(self):   #返回出队车辆和装车时间（距离上车走后）包括：Truck，time
        outqueue=[]
        #x=min(self.timetable[0:len(self.timetable), 0:1])   #消去最小的等待时间，让他出队
        x=self.timetable.max()
        if x<=0:
            return  outqueue
        minpoint=0          #没用上
        for i in range(0, len(self.timetable)):      #找最小的,不应该有0，等于0表示已经出队
            if  x>self.timetable[i][int(self.pointTrucks[i])]:
                x=self.timetable[i][int(self.pointTrucks[i])]
                minpoint=i
        #print(self.timetable,x)
        if x<=0:
            print("出队0数据")
            self.errorflag = 1;
            return outqueue
        for i in range(0,len(self.timetable)):  #消去最小的工作时间
            self.timetable[i][int(self.pointTrucks[i])]-=x
        #self.timetable[0:len(self.timetable), 0:1] -=x
        for i in range(0,len(self.timetable)):    #出队
            if self.timetable[i][int(self.pointTrucks[i])]==0:
                idtmp=int(self.truckid[i][int(self.pointTrucks[i])])
                if idtmp==0:
                    print("等铲出队idtemp==0",self.timetable,self.truckid)
                    exit(0)
                outqueue.append((self.trucks[idtmp-1],x))  #所有为0的出队
                self.truckid[i][int(self.pointTrucks[i])]=0                              #取消id
                self.pointTrucks[i] = (int(self.pointTrucks[i]) + 1) % len(self.timetable[0])  #move point
                self.pointTruckid[i]=self.pointTrucks[i]                    #移动id表的指针
        return outqueue

    def enqueue(self,intruck):  #加入车辆，该车辆的预估等待时间，选择所有队列中最短等待时间队列加入
        wt=self.timetable.sum(axis=1)   #row sum
        waitingtime=wt.min()
        wtt=wt.argmin(axis=0)   #返回index
        for j in range(0,len(self.timetable[0])):
            idtemp=int(self.pointTrucks[wtt]+j)%len(self.timetable[0])
            if self.timetable[wtt][idtemp]==0:   #新加入队列的车
                self.truckqueue[wtt][idtemp] =intruck.type
                self.truckid[wtt][idtemp]=intruck.Id
                self.timetable[wtt][idtemp] =intruck.TL+intruck.TS
                #-------加waittime-----
                break
        #print(self.timetable)
        return waitingtime


'''
*根据运输路径产生的pathQueue，对应n条路为n维数组,return truck and - waiting time
'''
class TrucksRunningQueue:
    def __init__(self, trucksset, shovelnum,pathnum=1):  #shovelnum表示运行结束的车数不大于铲车数目
        self.trucknum = len(trucksset)
        self.pathnum=pathnum
        self.shovelnum=shovelnum
        self.trucktypenum = len(trucksset)               #运行各阶段时间获取
        #self.trucks = copy.deepcopy(trucksset)
        self.trucks = trucksset

        #计算不同路径能够运行车辆数目，不超过len（trucksset/pathnum）数量
        self.truckinpath=int(len(trucksset)/pathnum)
        # runningqueue：2*i行存储车辆序号和车辆已经行驶时间2*i+1,
        # 从0开始,第二维表示已经运行时间,要向上取整+1，要空一个+1，总体+1
        self.runningqueue=np.zeros((pathnum*2,self.truckinpath+1))
        self.queuelen = len(self.runningqueue[0])

        self.firstpoint=np.zeros(pathnum*2)   #数组指针，最前面的车
        self.lastpoint=np.zeros(pathnum*2)

    def __selectpathGreedily__(self,truckid):  #选择路径，该方法使用greedy
        truck=self.trucks[truckid-1]            #队列从0开始
        return truck.getpathid()
    def enqueue(self,inqueue):    #inqueue表示inqueue第一维为Truck对象，第二维为入队时，上车运行时间
        #入队车辆类型,一次有可能多辆,同时出队时间完成的车辆类型，返回值有可能是多辆车,
        # 二维数组表示，1表示id，2表示出队后时间---应该小于等于0
        if len(inqueue)<=0:
                return -1
        for i in range(0,self.pathnum):
            if (self.lastpoint[int(i*2)] + 1) % self.queuelen == self.firstpoint[int(i*2)]:  # 如果timetable满了，返回错误，需要修正
                return  -2
        for i in range(0, len(inqueue)):
            if inqueue[i][0].Id == 0:  # or inqueue[i][0].Id == 14:
                print("INQ", inqueue[i][0].Id)
                exit(0)
        # 刷新timetable，表示运行了新进入车辆的铲车装车时间，所有运行时间减去该时间,
        # 应该是出来序列里面装车时间最小的(应该相同）
        mintime = inqueue[0][1]
        # count0=np.sum(self.runningqueue <= 0)
        for i in range(0, self.pathnum):
            if self.firstpoint[i] == self.lastpoint[i]:       #减去装车时间，表示装车时的运行
                if self.runningqueue[i * 2, int(self.firstpoint[i*2])]!=0:  #no truck is in the road, nothing should be done, otherwise overflowing.
                    print("the running queue overflows.")
                    exit(0);
            elif (self.firstpoint[i] < self.lastpoint[i]):
                self.runningqueue[i * 2 + 1, int(self.firstpoint[i*2+1]):int(self.lastpoint[i*2+1])] -= mintime
            else:
                self.runningqueue[i * 2 + 1, 0:self.queuelen] -= mintime
                self.runningqueue[i * 2 + 1, int(self.lastpoint[i*2+1]):int(self.firstpoint[i*2+1])]= 0
        # count1 = np.sum(self.runningqueue <= 0)
        # if(count1-count0>=3):
        #     print(self.runningqueue)
        #为不同的车选路

        for i in range(0, len(inqueue)):   #对第i辆
            trucktemp=(inqueue[i][0])
            truckidtemp=int(trucktemp.getID())
            selpathid=self.__selectpathGreedily__(truckidtemp)
            circletimetemp=self.trucks[truckidtemp-1]
            '''if self.runningqueue[0][self.firstpoint]==0:   #第一批进入运行的车,没有前车
                for i in range(0,len(inqueue)):
                    self.runningqueue[2*selpathid][self.lastpoint[i]]=inqueue[i][0]
                    self.runningqueue[2*selpathid+1][self.lastpoint] = self.circleTime
                    self.lastpoint=(self.lastpoint+1)%self.queuelen
            '''
            #进入新车辆,第一次会把所有的时间都减，但id表不会变
            pointid=int(selpathid*2)
            self.runningqueue[pointid][int(self.lastpoint[pointid])] = truckidtemp
            self.runningqueue[pointid+1][int(self.lastpoint[pointid+1])] =trucktemp.circleTime[selpathid]
            self.lastpoint[pointid] = (self.lastpoint[pointid] + 1) % self.queuelen  # point
            self.lastpoint[pointid+ 1] = (self.lastpoint[pointid + 1] + 1) % self.queuelen
        return 1

    def isEmpty(self):
        for i in range(0,self.pathnum):
            if(self.firstpoint[i+i]!=self.lastpoint[i+i]):
                return False
        return True

    def dequeue(self): #定义剩下的车辆出列，只出一组
        rtrucks = []
        if self.isEmpty():
            print("running queue is empty.")
            return rtrucks
        else:                                   # 返回出队的车
            for i in range(0,self.pathnum):      #对每条路径进行判断
                rowid=int(i*2)
                if self.runningqueue[rowid+1][int(self.firstpoint[rowid+1])]>0:   #还没有到达终点
                    continue
                elif self.runningqueue[rowid+1][int(self.firstpoint[rowid+1])]<0: #已经到达终点，并等待了
                    startp=int(self.firstpoint[rowid+1])
                    lastp=int(self.lastpoint[rowid+1])
                    flag=True
                    if startp>lastp:
                        flag=False
                    for j in range(0,len(self.runningqueue[0])):       #所有满足条件的出队
                        if flag and (j<startp or j>lastp):
                            continue
                        elif not flag and j<startp and j>lastp:      #lastpoint绕回来了
                            continue
                        fpoint = int(self.firstpoint[rowid])        #fpoint不能用j代替，j是从头开始，fpoint可能从后面开始，否则fpoint移动错误
                        if self.runningqueue[rowid+1][fpoint]<0:
                            idtmp=int(self.runningqueue[rowid][fpoint])
                            #----计算排队尾的时间-------
                            self.trucks[idtmp-1].cur_localwaitingtime+=abs(self.runningqueue[rowid+1][fpoint])
                            self.trucks[idtmp-1].wholewaitingtime+=abs(self.runningqueue[rowid+1][fpoint])

                            rtrucks.append((self.trucks[idtmp-1],self.runningqueue[rowid+1][fpoint]))  #返回车和该车已经等待时间
                            self.runningqueue[rowid+ 1][fpoint]=0
                            self.runningqueue[rowid ][fpoint]=0
                            self.firstpoint[rowid] = int((self.firstpoint[rowid ] + 1) % self.queuelen)
                            self.firstpoint[rowid+1] = int((self.firstpoint[rowid+1] + 1) % self.queuelen)
                            if self.firstpoint[rowid ] == self.lastpoint[rowid]:
                                break
                else:                                                       #刚到达终点或初始0
                    if (self.firstpoint[rowid]==self.lastpoint[rowid]):   #初始0
                        continue
                    else:                                                    #终点
                        for j in range(self.firstpoint[rowid + 1], self.lastpoint[rowid + 1]):  # 所有满足条件的出队
                            if self.runningqueue[rowid+ 1][j] < 0:
                                idtmp = self.runningqueue[rowid][self.firstpoint[rowid]]
                                rtrucks.append((self.trucks[idtmp-1], self.runningqueue[rowid + 1][j]))  # 返回车和该车已经等待时间
                                self.runningqueue[rowid+ 1][j] = 0
                                self.runningqueue[rowid][j] = 0
                                self.firstpoint[rowid] = (self.firstpoint[rowid] + 1) % self.queuelen
                                self.firstpoint[rowid+ 1] = (self.firstpoint[rowid + 1] + 1) % self.queuelen
                                if self.firstpoint[rowid] == self.lastpoint[rowid]:
                                   break
            # if len(rtrucks)>=3:
            #     print(self.runningqueue)
        return rtrucks

    def dispatchtruck(self,inqueue):
        x=self.enqueue(inqueue)
        return self.dequeue()