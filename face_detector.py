import cv2
import numpy as np
import urllib.request

def url_to_image(url):
    res = urllib.request.urlopen(url)
    image = np.asarray(bytearray(res.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def detect_faces(photograph):
    if photograph.any():
        # Use pretrained model
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # perform face detection
        bboxes = classifier.detectMultiScale(photograph)
        # bboxes will be numpy array if a face is found, or a tuple if no face is found
        if not type(bboxes) is tuple:
            print("face found")
            return True
        else:
            print("face not found")
            return False
    print("Image not retrieved from URL")
    return False
    
def main():
    url = input()
    image = url_to_image(url)
    detect_faces(image)

main()