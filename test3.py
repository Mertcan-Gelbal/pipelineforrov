import cv2
import numpy as np

def merge_lines(lines, threshold=30):
    merged_lines = []
    used = [False] * len(lines)
    
    for i, line1 in enumerate(lines):
        if used[i]:
            continue
        x1, y1, x2, y2 = line1
        for j, line2 in enumerate(lines):
            if i == j or used[j]:
                continue
            x3, y3, x4, y4 = line2
            
            if (np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3])) < threshold and 
                np.linalg.norm(np.array([x2, y2]) - np.array([x4, y4])) < threshold):
                x1, y1 = min((x1, y1), (x3, y3))
                x2, y2 = max((x2, y2), (x4, y4))
                used[j] = True
        
        merged_lines.append((x1, y1, x2, y2))
        used[i] = True
    
    return merged_lines

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([150, 255, 40])   # 180, 255, 35  
    # upper black ==> 150 255 40 BEST 
    
    mask = cv2.inRange(hsv, lower_black, upper_black)
    edges = cv2.Canny(mask, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        lines = [line[0] for line in lines]
        merged_lines = merge_lines(lines)
        
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
    
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)
        
        cv2.imshow('Processed Frame', processed_frame)
        
        if cv2.waitKey(1) == 27:  # ESC tuşu ile çıkış yapılabilir
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
