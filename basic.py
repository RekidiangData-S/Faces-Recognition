import cv2
import face_recognition

img = face_recognition.load_image_file('ImagesBasic/mandela.jpg')
img_test = face_recognition.load_image_file('ImagesBasic/mandela1.jpg')
# convert from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# find face in an image and there encoding
faceLoc = face_recognition.face_locations(img)[0]
encodeElon = face_recognition.face_encodings(img)[0]
print(faceLoc)
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
faceLocTest = face_recognition.face_locations(img_test)[0]
encodeTest = face_recognition.face_encodings(img_test)[0]
print(faceLocTest)
cv2.rectangle(img_test,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# Compare and find distance between faces
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
# TEXT
cv2.putText(img_test, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# DISPLAY
image_ref = 'Mandela'
image_test = 'Mandela1'
cv2.imshow(image_ref, img)
cv2.imshow(image_test, img_test)
cv2.waitKey(0)

# https://www.murtazahassan.com/courses/opencv-projects/lesson/complete-code-7/
