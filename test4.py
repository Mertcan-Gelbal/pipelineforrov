import cv2 
import numpy as np 
import time

time.sleep(2)
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

  
while 1: 
    ret,frame =cap.read()  
    
    if not ret:
        break
    
    # ret will return a true value if the frame exists otherwise False 
    into_hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
    # changing the color format from BGr to HSV  
    # This will be used to create the mask 
    L_limit=np.array([0,0,0]) # setting the black lower limit 
    U_limit=np.array([180,255,35]) # setting the black upper limit 
         
  
    b_mask=cv2.inRange(into_hsv,L_limit,U_limit) 
    # creating the mask using inRange() function 
    # this will produce an image where the color of the objects 
    # falling in the range will turn white and rest will be black 
    black=cv2.bitwise_and(frame,frame,mask=b_mask) 
    
    edges = cv2.Canny(b_mask, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(black, (x1, y1), (x2, y2), (0, 165, 255), 2)
    
    
    # Tracking black
    # contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if(area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    #         cv2.putText(img, "NEGRO: ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
            
            
             
    cv2.imshow('Original',frame) # to display the original frame 
    cv2.imshow('Black Detector',black) # to display the black object output 
  
    if cv2.waitKey(1)==27: 
        break
	# this function will be triggered when the ESC key is pressed 
	# and the while loop will terminate and so will the program 
cap.release() 

cv2.destroyAllWindows() 
