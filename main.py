import cv2
import numpy as np
import matplotlib.pyplot as plt 

def detect_line(image_path):
    img = cv2.imread(image_path)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([115, 115, 115])
    mask = cv2.inRange(img, lower_black, upper_black)
    
    edges = cv2.Canny(mask, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    img_lines = img.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1,y1), (x2,y2), (0, 165, 255), 2)
            
    plt.figure(figsize=(8,9))
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2GRAY))
    plt.axis('off')
    plt.show()
    
image_path = 'deneme4test.png'
detect_line(image_path)