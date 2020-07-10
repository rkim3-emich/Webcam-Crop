import cv2
import numpy as np

#Get video from default camera
cap = cv2.VideoCapture(0)

#Exits internally
while True:
    #Read frame from camera
    ret, frame = cap.read()

    #Get the height and width of the camera
    height, width, _ = frame.shape

    '''
    In order to prevent the top and bottom of the webcam from being registered as an edge,
    I had to crop the image. Mileage may vary based on the type of webcam
    '''
    frame = frame[100:height-70, 50:width-50].copy()

    #Video from my camera comes mirrored so flip to fix
    flipped = cv2.flip(frame, 1)


    #START EDGE FIND
    #convert image to gray scale
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)

    #Apply a gaussian threshold to gray scale image to reduce noise and provide outline of major shapes
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 8)

    #Find the contours in the image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Add contours that are only one level deep to list
    one_deep = []
    for index, data in enumerate(hierarchy[0]):
        if data[3] == -1:
            one_deep.append(np.insert(data.copy(), 0, [index]))

    #Find the area contained by contour and appedn to list
    contain_area = []
    for data in one_deep:
        index = data[0]
        contour = contours[index]
        area = cv2.contourArea(contour)
        contain_area.append([contour, area, index])

    #Sort list by area largest to smallest and get largets contour
    contain_area.sort(key=lambda meta: meta[1], reverse=True)
    largest_contour = contain_area[0][0]

    #Create image with rectangle determined by the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    out = flipped[y:y+h, x:x+w].copy()

    #Draw contour on original image
    cv2.drawContours(flipped, [largest_contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    #Display images
    cv2.imshow('Flipped with Contour', flipped)
    cv2.imshow('Cropped', out)

    #Exit loop if user eneters q and has numlock on
    if cv2.waitKey(1) & 0b11111111 == ord('q'):
        break

#Release webcam and destroy any cv2 windows
cap.release()
cv2.destroyAllWindows()
