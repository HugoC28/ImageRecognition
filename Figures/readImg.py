
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv


def greyPngToArray(fileName):
    img=plt.imread(fileName)
    N=len(img)
    M=len(img[0])
    result = np.zeros([N,M])
    for i in range(N):
        for j in range(M):
            result[i][j]=1-img[i][j][0]
    return result

def greyPngToVector(fileName):
    img=plt.imread(fileName)
    N=len(img)
    M=len(img[0])
    result = np.zeros([N*M])
    for i in range(N):
        for j in range(M):
            result[M*i+j]=1-img[i][j][0]
    return result

def csvWrite(fileName,data):
    with open(fileName+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def numberToAnswerVector(answer):
    result=np.zeros(10)
    result[answer]=1
    return result

dataList=np.array([])
answerList=np.array([])
for nb in range(10):
    filePath = "dataset/"+str(nb)+"/"
    for i in range(1,21):
        fileStrName=filePath+str(i)+".png"
        dataList=np.concatenate((dataList,greyPngToVector(fileStrName)),axis=None)
        answerList=np.concatenate((answerList,numberToAnswerVector(nb)),axis=None)
csvWrite('BDD_img',dataList)
csvWrite('BDD_ans',answerList)