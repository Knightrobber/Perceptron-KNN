#install opencv - pip install opencv-python
import cv2
import numpy as np
import glob

#training dataset - 20 images of cats and dogs each
#testing dataset - 5 images of cats and dogs each
# i have taken 2 features as mean and SD of all pixels in the image
meanSDMatrixCats =[] #stores data points of cats
meanSDMatrixDogs = []#stores data points of dogs
trainCount=0
testCount=0



print("Processing training images")
for img in glob.glob("./Dataset/CatsNDogs/Train/Cats/*.jpg"): #creating data points of cat images
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
	


for img in glob.glob("./Dataset/CatsNDogs/Train/Dogs/*.jpg"): #creating data points of dog images
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
	

eta=1 # fixing eta value
w = [1,1,1] # starting weights 
# updating weights according to a(k+1) = a(k) + sigma(Tj - W*Xj)Xj where Y is misclassified patterns
for i in range(0,10000): #looping 10000 times to get updated weights
	sumVector = [0]*3
	change=0
	for point in meanSDMatrixCats:
		val = w[0]*1+ w[1]*point[0] +w[2]*point[1]
		if val<0:
			sumVector[0] += (1-(-1)) * 1
			sumVector[1] += (1-(-1)) * point[0]
			sumVector[2] += (1-(-1)) * point[1]
			
			
	
	for point in meanSDMatrixDogs:
		val = w[0]*-1+ w[1]*-point[0] +w[2]*-point[1]
		if(val<0):
			sumVector[0] += (-1-(1)) * 1
			sumVector[1] += (-1-(1)) * point[0]
			sumVector[2] += (-1-(1)) * point[1]
			
		


	w[0]+= eta*sumVector[0]
	w[1]+= eta*sumVector[1]
	w[2]+= eta*sumVector[2]

trainClassification =0
trainCat=0
trainDog=0
for point in meanSDMatrixCats:
	val = w[0]*1 + w[1]*point[0] + w[2]*point[1]
	if(val>0):
		trainClassification+=1
		trainCat+=1
for point in meanSDMatrixDogs:
	val = w[0]*1 + w[1]*point[0] + w[2]*point[1]
	if(val<0):
		trainClassification+=1
		trainDog+=1

accuracy = trainClassification/trainCount
print("Training accuracy ",accuracy*100)



properClassificationCount=0
catCount=0
dogCount=0
print("Processing Test images")
print("Test Class    Predicted Class" )
for img in glob.glob("./Dataset/CatsNDogs/Test/Cats/*.jpg"): #processing Cat test images
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

	val = w[0]*1 + w[1]*mean + w[2]*SD
	if(val>0):
		properClassificationCount+=1
		dogCount+=1
		print("Cat            Cat")
	else:
		print("Cat            Dog")
	testCount+=1






for img in glob.glob("./Dataset/CatsNDogs/Test/Dogs/*.jpg"): #processing Cat test images
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

	val = w[0]*1 + w[1]*mean + w[2]*SD
	if(val<0):
		properClassificationCount+=1
		catCount+=1
		print("Dog            Dog")
	else:
		print("Dog            Cat")
	testCount+=1

accuracy = properClassificationCount/testCount
print("The accuracy is ",accuracy*100)






