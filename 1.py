#install opencv - pip install opencv-python
import cv2
import numpy as np
import glob
#training dataset - 20 images of cats and dogs each
#testing dataset - 5 images of cats and dogs each
meanSDMatrixCats =[] #stores data points of cats
meanSDMatrixDogs = [] #stores data points of dogs
global trainCount
global testCount
trainCount=0
testCount=0
noOfClasses = 2

print("Processing images")
for img in glob.glob("./Dataset/CatsNDogs/Train/Cats/*.jpg"): #making the data points for cat images
	image= cv2.imread(img)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rows,cols = gray.shape
	sumPixels = 0
	SDSum=0
	for i in range(0,rows):
		for j in range(0,cols):
			sumPixels+=gray[i,j]
	mean = sumPixels/(rows*cols)
	for i in range(0,rows):
		for j in range(0,cols):
			SDSum+= (gray[i,j] - mean)**2

	SD = ((1/(rows*cols))*SDSum)**0.5
	tempArr = [mean,SD,1]
	meanSDMatrixCats.append(tempArr)
	trainCount+=1
	"""if(count==6):
		break
	count+=1"""
count=0
	
for img in glob.glob("./Dataset/CatsNDogs/Train/Dogs/*.jpg"): #making the data points for dog images
	image= cv2.imread(img)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rows,cols = gray.shape
	sumPixels = 0
	SDSum=0
	for i in range(0,rows):
		for j in range(0,cols):
			sumPixels+=gray[i,j]
	mean = sumPixels/(rows*cols)
	for i in range(0,rows):
		for j in range(0,cols):
			SDSum+= (gray[i,j] - mean)**2

	SD = ((1/(rows*cols))*SDSum)**0.5
	tempArr = [mean,SD,2]
	meanSDMatrixDogs.append(tempArr)
	trainCount+=1
	"""if(count==6):
		break
	count+=1"""


def main():
	Data=[]
	Data.append(meanSDMatrixCats)
	Data.append(meanSDMatrixDogs)
	print("predicted Class = 0 =>   Ambiguous")
	print("predicted Class = 1 =>   Cats")
	print("predicted Class = 2 =>   Dogs")

	kTestCount=[0]*len(meanSDMatrixCats) #stores the accuracy results of each k
	for img in glob.glob("./Dataset/CatsNDogs/Test/Cats/*.jpg"): #for each test image of Cats finding if it's classified or misclassified for each k from 1-20
		#print("\n")
		image= cv2.imread(img)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rows,cols = gray.shape
		sumPixels = 0
		SDSum=0
		for i in range(0,rows):
			for j in range(0,cols):
				sumPixels+=gray[i,j]
		mean = sumPixels/(rows*cols)
		for i in range(0,rows):
			for j in range(0,cols):
				SDSum+= (gray[i,j] - mean)**2

		SD = ((1/(rows*cols))*SDSum)**0.5
		testDataPoint=[mean,SD,1]
		predictedClassArray=[]
		for i in range(1,21):
			predictedClass = predictClass(testDataPoint,Data,i) # we get predicted class here
			predictedClassArray.append(predictedClass)
		for i in range(0,len(predictedClassArray)):
			if(predictedClassArray[i]==1):
				kTestCount[i]+=1
		global testCount
		testCount+=1
	

	for img in glob.glob("./Dataset/CatsNDogs/Test/Dogs/*.jpg"): #for each test image of Dogs finding if it classifies or misclassifies for each k from 1-20
		#print("\n")
		image= cv2.imread(img)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rows,cols = gray.shape
		sumPixels = 0
		SDSum=0
		for i in range(0,rows):
			for j in range(0,cols):
				sumPixels+=gray[i,j]
		mean = sumPixels/(rows*cols)
		for i in range(0,rows):
			for j in range(0,cols):
				SDSum+= (gray[i,j] - mean)**2

		SD = ((1/(rows*cols))*SDSum)**0.5
		testDataPoint=[mean,SD,1]
		predictedClassArray=[]
		for i in range(1,21):
			predictedClass = predictClass(testDataPoint,Data,i) # we get predicted class here
			predictedClassArray.append(predictedClass)
		for i in range(0,len(predictedClassArray)):
			if(predictedClassArray[i]==2):
				kTestCount[i]+=1
		testCount+=1
	
	for i in range(0,len(kTestCount)):
		kTestCount[i] = kTestCount[i]/testCount # we have a total of 10 test images
		if(i==0):
			print("K    Accuracy")
		print(f"{i+1}    {round(kTestCount[i]*100,2)}")
	
	bestK=kTestCount.index(max(kTestCount))+1
	print("The best suited K is",bestK)



def getEuclidean(point1,point2):
	total=0
	for i in range(0,len(point1)):
		total+= abs(point1[i]-point2[i]) **2

	return round(total**0.5,2)



def predictClass(data,trainingData,k): # finds the distances of all training data from given test data and based on given value of k returns the class it should belong to
	testPoint = data
	distArray=[]
	for row in trainingData:
		for trainPoint in row:
			point1=[]
			point2=[]
			for i in range(0,len(trainPoint)-1):
				point1.append(trainPoint[i])
			for i in range(0,len(testPoint)-1):
				point2.append(testPoint[i])
			"""point1 = [(float)(trainPoint[0]),(float)(trainPoint[1]),(float)(trainPoint[2]),(float)(trainPoint[3])]
			point2 = [(float)(testPoint[0]),(float)(testPoint[1]),(float)(testPoint[2]),(float)(testPoint[3])]"""
			dist = []
			dist.append(getEuclidean(point1,point2))
			lenOfTrainPoint = len(trainPoint)
			indexOfClass = lenOfTrainPoint-1
			dist.append(trainPoint[indexOfClass]) # to every distance we append its corresponding class
			distArray.append(dist)# A 2D array whose columns are distances of each test data from traning data and the rows corresspond to each test data
	predictedClass = findKnnClass(distArray,k)
	return predictedClass

def findKnnClass(distArray,k):  #takes a distance array each of whose elements are an array which has euclidean distance and the class it corresponds and returns the class with the highest frequency within the given value of k
	
	sortedList = sorted(distArray,key=lambda x:x[0])
	classCount=[0]*noOfClasses
	kCount=0
	for element in sortedList:
		if(kCount<k):
			if(element[1]==1):
				classCount[0]+=1
			elif(element[1]==2):
				classCount[1]+=1
			elif(element[1]==3):
				classCount[2]+=1
			kCount+=1
		else:
			break

	maxVal = max(classCount)
	maxCount=0
	for val in classCount:
		if(val==maxVal):
			maxCount+=1
	if(maxCount==1):
		return classCount.index(maxVal) +1
	elif(maxCount>1): 
		return 0



main()
