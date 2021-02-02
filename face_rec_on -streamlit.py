import cv2
import face_recognition
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


def detect(image):

    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml')
    except Exception:
        st.write("Error loading cascade classifiers")

    image = np.array(image.convert('RGB'))

    faces = face_cascade.detectMultiScale(image, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 5)

    # cv2.imshow("Result", img)
    # cv2.waitKey(0)
    return image, faces


def main():
    timg = Image.open("images/facerec.jpg")
    st.image(timg, use_column_width=True)

    activities = ["Face Detection", "Face Recognition",
                  "Face Recognition in Attendance System", "About"]
    choice = st.sidebar.selectbox("Select Operation", activities)

    ############## Face Detection ##########################

    if choice == "Face Detection":
        st.title("Face Detection")

        # You can specify more file types below if you want
        st.subheader("Upload Image")
        image_file = st.file_uploader(
            "____", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file is not None:

            image = Image.open(image_file)

            if st.button("Detection"):

                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img, result_faces = detect(image=image)
                st.image(result_img, use_column_width=True)
                st.success("Found {} faces\n".format(len(result_faces)))

    ############## Face Recognition ###################
    if choice == "Face Recognition":

        st.title("Face Recognition")
        # You can specify more file types below if you want
        image_file1 = st.file_uploader(
            "Upload image1", type=['jpeg', 'png', 'jpg', 'webp'])
        image_file2 = st.file_uploader(
            "Upload image2", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file1 is not None:
            image1 = image_file1
        if image_file2 is not None:
            image2 = image_file2

        imgElon = face_recognition.load_image_file(image1)
        imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
        imgTest = face_recognition.load_image_file(image2)
        imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

        faceLoc = face_recognition.face_locations(imgElon)[0]
        encodeElon = face_recognition.face_encodings(imgElon)[0]
        cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]),
                      (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

        faceLocTest = face_recognition.face_locations(imgTest)[0]
        encodeTest = face_recognition.face_encodings(imgTest)[0]
        cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),
                      (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

        imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
        imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

        results = face_recognition.compare_faces([encodeElon], encodeTest)
        faceDis = face_recognition.face_distance([encodeElon], encodeTest)
        print(results, faceDis[0])
        # cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (
        # 50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

        # cv2.imshow('Elon Musk', imgElon)
        # cv2.imshow('Elon Test', imgTest)
        # cv2.waitKey(0)
        # return imgElon, imgTest

        if st.button("Recognition"):
            # result_img is the image with rectangle drawn on it (in case there are faces detected)
            # result_faces is the array with co-ordinates of bounding box(es)
            # result_img, result_faces = recognition(image1, image2)

            if results[0] == True:
                st.success(
                    f'Similarity : {results[0]} Distance value : {round(faceDis[0],2)}')
                cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),
                              (faceLocTest[1], faceLocTest[2]), (0, 255, 0), 3)
            else:
                st.error(
                    f'Similarity : {results[0]} Distance value : {round(faceDis[0],2)}')
                cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),
                              (faceLocTest[1], faceLocTest[2]), (255, 0, 0), 3)
            st.image(imgTest, use_column_width=True)
            # st.success("Found {} faces\n".format(len(result_faces)))
    if choice == "Face Recognition in Attendance System":
        st.title("Face Recognition in Attendance System")

    elif choice == "About":
        about()


if __name__ == "__main__":
    main()
