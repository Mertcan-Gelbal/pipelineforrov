import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

"""
Bu kod, kameradan alınan görüntü üzerinde siyah renkli nesneleri tespit etmek, çizgileri belirlemek ve Lyapunov tabanlı kontrol stratejisi uygulamak için tasarlanmıştır. Kamera açılışı için belirli bir süre beklenir ve ardından görüntüler işlenir. Görüntü, HSV formatına dönüştürülerek siyah renk için bir maske oluşturulur. Kenar tespiti ve Hough dönüşümü kullanılarak çizgiler belirlenir. Çizgiler arasında hesaplanan açı ile kontrol sinyali üretilir ve sonuçlar, görüntü üzerinde turuncu çizgilerle gösterilir.
"""

# Boru hattı takibi ve Lyapunov kontrolü için parametreler
def calculate_heading_angle(lines):
    """
    Bu fonksiyon, tespit edilen çizgiler arasındaki açıyı hesaplar. 
    Eğer yeterli çizgi tespit edilememişse, fonksiyon None döndürür.
    """
    if lines is not None and len(lines) >= 2:
        rho1, theta1 = lines[0][0]
        rho2, theta2 = lines[1][0]
        angle = abs(theta1 - theta2)
        return angle
    return None

def lyapunov_control(y, psi, psi_n):
    """
    Bu fonksiyon, Lyapunov tabanlı bir kontrol stratejisi uygular. 
    Sistemdeki hatayı azaltmak için bir kontrol sinyali üretir. 
    'y' parametresi, 'psi' mevcut açı ve 'psi_n' hedef açıyı temsil eder.
    """
    c = 1.0  # Pozitif kazanç değeri
    error = psi - psi_n
    control_signal = -c * error
    return control_signal

# Kameradan görüntü yakalama
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

time.sleep(2)  # Kamera açılışı için bekleme süresi

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    """
    Görüntü işleme adımları:
    1. Görüntü, BGR formatından HSV formatına dönüştürülür.
    2. Siyah renk için alt ve üst limitler tanımlanır.
    3. Belirlenen limitlere göre bir maske oluşturulur.
    4. Kenar tespiti yapılır ve Hough dönüşümü ile çizgiler belirlenir.
    5. Tespit edilen çizgiler arasında açısal fark hesaplanır ve Lyapunov tabanlı kontrol stratejisi uygulanır.
    6. Hesaplanan açı ve kontrol sinyali görüntü üzerine yazılır ve turuncu çizgiler ile gösterilir.
    """

    # Görüntüyü HSV formatına dönüştürme
    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Siyah renk için alt ve üst limitler
    L_limit = np.array([0, 0, 0])
    U_limit = np.array([180, 255, 35])
    
    # Maske oluşturma
    b_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    
    # Maskeyi kullanarak siyah renkleri tespit etme
    black = cv2.bitwise_and(frame, frame, mask=b_mask)
    
    # Kenarları algılama
    edges = cv2.Canny(b_mask, 50, 150)
    
    # Hough dönüşümü ile çizgileri tespit etme
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    
    # Lyapunov tabanlı kontrol stratejisini uygulama
    angle = calculate_heading_angle(lines)
    control_signal = lyapunov_control(0.0, angle if angle is not None else 0.0, np.pi/4)
    
    # Çizgileri görüntü üzerinde gösterme
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # Turuncu renkte çizgi ekleme
            cv2.line(black, (x1, y1), (x2, y2), (0, 140, 255), 2)
    
    # Sonuçları görüntü üzerinde gösterme
    cv2.putText(black, f'Angle: {angle if angle is not None else "N/A"}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(black, f'Control Signal: {control_signal:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Orijinal ve işlenmiş görüntüyü gösterme
    cv2.imshow('Original', frame)
    cv2.imshow('Black Detector', black)
    
    # ESC tuşuna basıldığında döngüden çıkma
    if cv2.waitKey(1) == 27:
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
