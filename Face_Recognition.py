import numpy as np
import cv2
import os

#implementing in KNN algorithm

def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())

def Knn(X_train,test,k=5):
	dist = []

	#Since in training data there is also classification or i.e. label column
	#So we will extract that data

	for i in range(X_train.shape[0]):
	# spliting label and features
		ix = X_train[i, :-1]
		iy = X_train[i, -1]

		calc_dist = distance(test,ix)
		dist.append([calc_dist,iy])
#sorting dist list on the basis of calc_dist
	sorted_dist = sorted(dist, key=lambda x: x[0])[:k]
	labels = np.array(sorted_dist)[:,-1] #Extracting only labels

	output = np.unique(labels,return_counts=True)

	index = np.argmax(output[1])
	return output[0][index]

######################################
#Initialising Camera
cap = cv2.VideoCapture(0)
#loading haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0

face_data_path = './face_data/'

face_data = []
label = []

class_id = 0 
#label for the given value
names = {} 

# Data Preperations
for fx in os.listdir(face_data_path):
	if fx.endswith('.npy'):

		names[class_id] = fx[:-4]
		data_item = np.load(face_data_path+fx)
		face_data.append(data_item)
		
		target = class_id * np.ones((data_item.shape[0],))
		class_id+=1
		label.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels  = np.concatenate(label,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)		

trainset = np.concatenate((face_dataset,face_labels),axis=1)




while True:
	bool_val,frame = cap.read()

#if frame is not captured for some reason Try again		
	if(bool_val == False):
		continue
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	
	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces:
		x,y,w,h = face
		

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		out = Knn(trainset,face_section.flatten())

		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
	
	
	cv2.imshow("Face",frame)
	

#it breaks when user presses q	
	key_pressed = cv2.waitKey(1) & 0xFF
	if(key_pressed == ord('q')):
		break
cap.release()
cv2.destroyAllWindows()		
