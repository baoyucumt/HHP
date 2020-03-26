
import MyFun as fun
from MTWT import MTWT
from Trucks import Truck
from MyPSO import DISCRETEPSO as pso
import numpy as np
from GATrucks import GeneticAlgorithm as GA
from GATrucks import GeneticAlgorithmByT as GAT


#if __name__ == "__main__":

#truck info and shovel info
#time inof


def worker():
    truckdata=[]
    try:
        truckdata = fun.loadDataSet("Truck.txt")  # 类型对应的不同事务处理时间，为n行，每种车辆类型占用的行数同可选路径数相同
        trucksNumdata = fun.loadDataSet("usingtrucks.txt")  # 车辆编号和车辆类型，为两行数据，序号从1开始
    except IOError as e:
        print(e)
        exit(0)
    #type and number info
    trucktypenum=2   #车辆类型个数值
    truckNum=[20,10]
    shoveltypenum=1    #铲车类型个数数
    shovelNum=2       #铲车数量

    #初始化卡车队列
    trucksset=[]              #卡车集合
    QueueNum=shovelNum       #根据车辆类型确定的队列数目,这是卡车等待铲车的队列
    pathNum=(int)(len(truckdata)/trucktypenum)  #存在的路径数目
        #计算车辆编号放入的队列，前QueueNum行是编号，后面行的是车的类型，初始文件为两行
    #trucksNumDatas=np.array(trucksNumdata).reshape(QueueNum*2,(int)(trucksNumdata[0].__len__()/QueueNum))
        #初始化车队，效果等于trucksNumDatas实现具体化
    for i in range(0,len(trucksNumdata[0])):
        truckdatas=[]
        trucktypetemp = (int)(trucksNumdata[1][i])
        for j in range(0,pathNum):                      #建立不同path上的时间数组
            baserow=trucktypetemp*pathNum-1
            truckdatas.append(truckdata[baserow+j])
        trucks = Truck(truckdatas, trucktypetemp,trucksNumdata[0][i])
        trucksset.append(trucks)

        #重新将车队改为铲车对应的初始队列
    qmat=np.array(trucksset).reshape(QueueNum,(int)(trucksNumdata[0].__len__()/QueueNum))

    '''
    GA
    '''
    cycles=30
    para=[0.3,0.4,0.01]
    bound=np.zeros((2,len(trucksset)))
    for i in range(0,len(trucksset)):
        bound[0,i]=1
        bound[1,i]=2
    #gat=GA(20,len(trucksset),bound,50,para)
    gat=GAT(20,len(trucksset),bound,50,para,2)
    gat.setTrucksPara(truckdata,pathNum,shovelNum,cycles)
    gat.solve()

if __name__=='__main__':
    # t=[x for x in range(2)]
    # for i in range(2):
    worker()

        # t[i] = multiprocessing.Process(target=worker, args=())
        # t[i].daemon = True
        # t[i].start()
    # t[0].join()
    # t[1].join()


#show truckset
#for i in range(0,len(trucksset)):
#    print(trucksset[i].Id,trucksset[i].wholewaitingtime)
# ts = []
# ts2 = []
# cycles = 400
# QueueMaxLen=14   #车辆数目
#
# for i in range(0,QueueMaxLen):
#     qmat = np.array([(1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1)])
#     j=0
#     while j<4:
#         k0=np.random.randint(0,2)
#         k=np.random.randint(0,7)
#         if qmat[k0][k]!=1:
#             qmat[k0][k]=1
#             j+=1
#     #qmat[0][k]=1
#     # k = np.random.randint(0, 7)
#     # qmat[1][k]=1
#     print(qmat)
#     t = mtwt.compute_AllQueuewaitingtime(qmat, mtwt.shoovelNum, cycles)
#     ts.append(t/cycles)
#
#     t = mtwt.compute_AllQueuewaitingtime2(qmat, mtwt.shoovelNum, cycles,2)
#     ts2.append(t/cycles)
#
# #qmat5 = np.array([(2, 1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2)])
# print((np.max(ts)-min(ts))/max(ts))
# print((np.max(ts2)-min(ts2))/max(ts2))
# print(ts)
# print(ts2)
# print("ok")
# quit()
#
# #method 1: The least waiting time for each truck
# mtwt=MTWT(trucksset,shoovelNum,shoovelNum=2,pathNum=1)
#
#
# # 1 type truck to 1 type shovel
#
#     # 这个计算函数不考虑卡车可能在排土或排煤时的等待，计算可以有多少辆车在循环
#     #  WaitingTime  Loading+SpottingTime Hauling  Queuing    Backing+Tipping  Traveling
#     #   TW             TL      TS         TH            TQ         TB      TP       TT
#     # trucks.TW+trucks.TL+trucks.TS+trucks.TH+trucks.TQ+trucks.TB+trucks.TP+trucks.TT
#     # 不考虑不同车循环时耗费的时间不同
#
#
# qmat = np.array([(1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1)])
# qmat2 = np.array([(2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)])
# qmat3 = np.array([(1, 1, 1, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2, 1)])
# #qmat4 = np.array([(1, 1, 2, 1, 1, 2, 2), (2, 1, 1, 1, 2, 1, 1)])
# qmat4 = np.array([(2, 2, 2, 2, 2, 2, 1), (2, 2, 2, 2, 2, 2, 2)])
# qmat5 = np.array([(2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 1, 2,2)])
# # pso=PSO(5,1,100,self.trucksset)
# ts = []
# ts2 = []
# cycles = 30
# t = mtwt.compute_AllQueuewaitingtime(qmat, mtwt.shoovelNum, cycles)
# ts.append(t / cycles)
# t = mtwt.compute_AllQueuewaitingtime(qmat2, mtwt.shoovelNum, cycles)
# ts.append(t / cycles)
# t = mtwt.compute_AllQueuewaitingtime(qmat3, mtwt.shoovelNum, cycles)
# ts.append(t / cycles)
# t = mtwt.compute_AllQueuewaitingtime(qmat4, mtwt.shoovelNum, cycles)
# ts.append(t / cycles)
# t = mtwt.compute_AllQueuewaitingtime(qmat5, mtwt.shoovelNum, cycles)
# ts.append(t / cycles)
#
# t = mtwt.compute_AllQueuewaitingtime2(qmat3, mtwt.shoovelNum, cycles, 2)
# ts2.append(t / cycles)
# t = mtwt.compute_AllQueuewaitingtime2(qmat4, mtwt.shoovelNum, cycles, 2)
# ts2.append(t / cycles)
# t = mtwt.compute_AllQueuewaitingtime2(qmat5, mtwt.shoovelNum, cycles, 2)
# ts2.append(t / cycles)
# print(ts)
# print(ts2)
# q=[1,1,1,1,1,1,1,2,2,2,2,2,2,1]
# resultq=list(itertools.permutations(q,14))
# ff=[]
# for i in range(0,len(resultq)):
#     tm=resultq[i]
#     qmat=np.array(tm).reshape(2,7)
#     t = self.compute_AllQueuewaitingtime(qmat, self.shoovelNum, 1)
#     print(t)
#     ff.append(t)

#   circletime+=self.trucksset[0]

# self.compute_waitingtime()

#write results for each optimal prog.

