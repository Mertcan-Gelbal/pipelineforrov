import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_line(image_path):
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
        return
    
    # Siyah renk için maske oluştur
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([115, 115, 115])
    mask = cv2.inRange(img, lower_black, upper_black)

    # Turuncu bir görüntü oluştur (aynı boyutta)
    orange_mask = np.zeros_like(img)
    orange_mask[mask != 0] = [0, 165, 255]  # Turuncu rengini uygula

    # Siyah bölgeleri turuncu ile maskele
    masked_img = cv2.addWeighted(img, 1, orange_mask, 0.5, 0)

    # Görüntüyü göster
    plt.figure(figsize=(8, 9))
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

image_path = 'deneme4test.png'
detect_line(image_path)
