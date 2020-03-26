import numpy as np
import random
import matplotlib.pyplot as plt

#计算采煤、排土场的运输距离是否和车容量有关系
#假设排土场是一个RECtangle，Width is W， Length is L
#两种装车容量C!,C2
#求解总运输距离最小，入口设计位置
def mineMapDeploy_onebyone(C1, C2, L,w, MaxLineC, MMap):
    lastC = C2
    linesum = 0
    for i in range(1, w):
        for j in range(1, int(MaxLineC)):
            if (lastC == C1):
                MMap[i][j] = C2
                linesum += C2
                lastC = C2
            else:
                MMap[i][j] = C1
                linesum += C1
                lastC = C1
            if (linesum >= L):
                linesum = 0
                break

def mineMapDeploy_twosides(C1, C2, L,w, MaxLineC, MMap):

    linesum = 0
    for i in range(1, w):
        for j in range(1, int(MaxLineC/2+1)):
            if(linesum+C1+C2 >= L):
                if (linesum + min(C1, C2) >= L):
                    MMap[i][j] = min(C1, C2)
                else:
                    if linesum + max(C1, C2) >= L :
                        MMap[i][j] = max(C1, C2)
                    else:
                        MMap[i][j] = C1
                        MMap[i][int(MaxLineC) - j] = C2
                linesum=0
                break
            linesum += C1 + C2
            MMap[i][j]=C1
            MMap[i][int(MaxLineC)-j]=C2




def compute_dis_ByMap(W, MaxLineC, MMap,X):  #计算矩阵的距离和
    linesum = 0
    dissum = 0
    for i in range(1,W):
        for j in range(1,int(MaxLineC)):
            if MMap[i][j]!=0 :
                linesum+=MMap[i][j]

                if(L-X-linesum)>0:
                    dis=np.sqrt(i**2+(L-X-linesum)**2)
                else:
                    dis = np.sqrt(i ** 2+(linesum-X)**2)
                dissum+=dis
    return dissum

w=150
L=180
C1=5
C2=4
maxLineC=L/min(C1,C2)
#入口在x位置
mineMap=np.zeros((int(w),int(maxLineC)))
mineMap2=np.zeros((int(w),int(maxLineC)))
mineMap3=np.zeros((int(w),int(maxLineC)))
mineMap4=np.zeros((int(w),int(maxLineC)))
#测试一车接一车，入口在一个角


mineMapDeploy_onebyone(C1, C2, L,w, maxLineC, mineMap)
dissum=compute_dis_ByMap(w, maxLineC, mineMap,0)
file_object = open('map.txt', 'w')
#file_object.write(str(mineMap.reshape(int(w),int(L/2))))
#print(*mineMap.reshape(50,int(maxLineC)))
print(dissum)
mineMapDeploy_onebyone(C1, C2, L,w, maxLineC, mineMap2)
dissum1=compute_dis_ByMap(w, maxLineC, mineMap2,int(L/2))
print(dissum1)
mineMapDeploy_twosides(C1, C2, L,w, maxLineC, mineMap3)
dissum3=compute_dis_ByMap(w, maxLineC, mineMap3,0)
file_object.write("=====")
file_object.write(str(mineMap3))
#print(*mineMap3.reshape(50,int(maxLineC)))
print(dissum3)
mineMapDeploy_twosides(C2, C1, L,w, maxLineC, mineMap4)
dissum=compute_dis_ByMap(w, maxLineC, mineMap4,int(L/2))
print(dissum)
dissum2=compute_dis_ByMap(w, maxLineC, mineMap4,int(L*C1/C2))
print(dissum2)
dissum=compute_dis_ByMap(w, maxLineC, mineMap4,int(L*(C2/C1)))
print(dissum)

print((dissum1-dissum2)/dissum1)
print((dissum3-dissum)/dissum3)


file_object.close( )
