import cv2
import numpy as np

skip = 0
face_data = []
face_data_path = './face_data/'
filename = input("Enter the name of the person :")
#Initialising Camera
cap = cv2.VideoCapture(0)

#loading haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	
	bool_val,frame = cap.read()

#if frame is not captured for some reason Try again		
	if(bool_val == False):
		continue
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)

	if len(faces)==0:
		continue

	faces = sorted(faces,key=lambda f:f[2]*f[3])
	
	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip+=1
		if(skip%10 == 0):
			face_data.append(face_section)
			print(len(face_data))
	
	
	cv2.imshow("Face",frame)
	cv2.imshow("face_part",face_section)
	

#it breaks when user presses q	
	key_pressed = cv2.waitKey(1) & 0xFF
	if(key_pressed == ord('q')):
		break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(face_data_path+filename+'.npy',face_data)
print("Data Save successfully"+face_data_path+filename+'.npy')
cap.release()
cv2.destroyAllWindows()		