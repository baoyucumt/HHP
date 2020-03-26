'''
   * @author 鲍宇
   * 载入数据
   ** datatype表示载入数据的类型，1表示默认的读入数据，不做处理，待扩展
'''

def loadDataSet(filename,datatype=1):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        line = line.replace('"',' ')
        line = line.replace(',', ' ')
        curLine=line.strip().split()
        aa = [float(i) for i in curLine]
        dataMat.append(aa)
    return dataMat

#dataMat=loadDataSet('datum')
#print (dataMat)

def writeDataSet(filename,datatype=1,mtwttime=-1,truckset=[],idserial=[]):
    if(datatype==1):   #写入truckset
        fw = open(filename,'w+')
        wstr=str(mtwttime)+";"
        for i in range(0,len(idserial)):
            wstr+=str(idserial[i])+","
        print(wstr)
        fw.write(wstr)
    return