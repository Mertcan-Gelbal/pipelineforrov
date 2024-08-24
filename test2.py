import cv2
import numpy as np
import matplotlib.pyplot as plt

"""Bu kod çizgi oluşturma konusunda son eklenen duruma göre hazırlanmış bir kod."""

def detect_line(image_path):
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
        return
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([90, 90, 90])  # Gölge algılamayı azaltmak için 85 ile 90 
    mask = cv2.inRange(img, lower_black, upper_black)

    edges = cv2.Canny(mask, 50, 150) # Kenar algılama

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100) # Çizgileri algılama 

    img_height, img_width = img.shape[:2]
    mid_x = img_width // 2

    right_side_lines = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        if x1 > mid_x or x2 > mid_x: 
            right_side_lines.append((x1, y1, x2, y2))
    
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

    merged_lines = merge_lines(right_side_lines) 

    img_lines = img.copy()
    if merged_lines:
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 165, 255), 5) 

    plt.figure(figsize=(8, 9))
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

image_path = 'deneme4test.png'
detect_line(image_path)
