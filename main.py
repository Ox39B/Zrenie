import cv2
import time
import numpy as np
from square_rotating_test import find_squares, order_points, four_point_transform

def main():
    # Захватываем видео с камеры
    cap = cv2.VideoCapture(0)
    
    # Список для хранения последних 15 фото
    photo_list = []
    
    while True:
        # Читаем кадр с камеры
        ret, frame = cap.read()
        
        if not ret:
            break
        
        _, tresh = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)


        # Добавляем новый кадр в список
        if len(photo_list) < 15:
            photo_list.append(frame)
        else:
            # Удаляем первый кадр и добавляем новый в конец списка
            photo_list.pop(0)
            photo_list.append(frame)
        
        # Проводим поиск квадратов на всех фото в списке
        for photo in photo_list:
            squares = find_squares(photo)
            
            # Проверка размеров квадратов и рисование квадратов
            for square in squares:
                x, y, w, h = cv2.boundingRect(square)
                if w >= 40 and h >= 40:
                    cv2.drawContours(photo, [square], 0, (0, 255, 0), 3)
        
        # Отображаем последнюю фотографию с нарисованными квадратами
        cv2.imshow("Squares Detection", photo_list[-1])
        cv2.imshow("tresh", tresh)
        
        # Ждём 1 секунду перед захватом следующего кадра
        time.sleep(1)
        
        # Прерываем цикл по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()