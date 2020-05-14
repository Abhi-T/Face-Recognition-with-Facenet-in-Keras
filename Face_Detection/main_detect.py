import numpy as np
import cv2
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.preprocessing import image
from os import listdir
from inception_resnet_v1 import *

#FaceNet model expects 160×160 RGB images whereas it produces 128-dimensional representations
def preprocess_image(image_path):
    img=load_img(image_path, target_size=(160,160))
    img=img_to_array(img)
    img=np.expand_dims(img, axis=0)
    return img

# model=model_from_json(open("facenet_model.json","r").read())
model = InceptionResNetV1()
model.load_weights('facenet_weights.h5')
print(model.summary())

def findEuclidianDistance(source_representation, test_representation):
    euclidian_distance=source_representation - test_representation # difference
    euclidian_distance=np.sum(np.multiply(euclidian_distance, euclidian_distance)) # matrix multiplication and then sum
    euclidian_distance=np.sqrt(euclidian_distance) # sqrt
    return euclidian_distance


threshold=21

employee_pictures="database/"

employees=dict()

for file in listdir(employee_pictures):
    employee, extension=file.split(".")
    img=preprocess_image('database/%s.jpg'%(employee))
    representation=model.predict(img)[0,:]
    employees[employee]=representation

print("Employee representations retrieved successfully")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0) # default cam

while(cap.isOpened()):
    ret, img= cap.read()
    faces=face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:
        if w>130: #discard small detected faces
            print("Abhishek")
            cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) # draw rectangle to main image
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.resize(detected_face, (160, 160))  # resize to 224x224
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 127.5
            img_pixels -= 1

            captured_representation = model.predict(img_pixels)[0, :]

            distances = []

            for i in employees:
                employee_name = i
                source_representation = employees[i]

                distance = findEuclidianDistance(captured_representation, source_representation)
                print(employee_name,": ",distance)
                distances.append(distance)

            label_name = 'unknown'
            index = 0

            for i in employees:
                employee_name = i
                if index == np.argmin(distances):
                    if distances[index] <= threshold:
                        # print("detected: ",employee_name)

                        # label_name = "%s (distance: %s)" % (employee_name, str(round(distance,2)))
                        similarity = 100 + (20 - distance)
                        if similarity > 99.99: similarity = 99.99

                        label_name = "%s (%s%s)" % (employee_name, str(round(similarity, 2)), '%')

                        break

                index = index + 1
        cv2.putText(img, label_name, (int(x + w + 15), int(y - 64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (67, 67, 67), 2)

        # connect face and text
        cv2.line(img, (x + w, y - 64), (x + w - 25, y - 64), (67, 67, 67), 1)
        cv2.line(img, (int(x + w / 2), y), (x + w - 25, y - 64), (67, 67, 67), 1)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# kill open cv things
cap.release()
cv2.destroyAllWindows()