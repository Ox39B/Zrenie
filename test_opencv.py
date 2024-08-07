import cv2
import numpy as np
import os
from square_rotating_test import perspective_transform

def detect_white_shapes(image_path, lower_white=(0, 0, 200), upper_white=(180, 25, 255), contour_color=(0, 255, 0), contour_thickness=2):
    # Чтение изображения
    image = cv2.imread(image_path)

    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Создание маски для выделения белого цвета
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Поиск внешних контуров на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисование только внешних контуров
    for contour in contours:
        # Аппроксимация контура для уменьшения количества точек
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Рисование контура на изображении
        cv2.drawContours(image, [approx], 0, contour_color, contour_thickness)

    return image, contours, cv2.imread(image_path)  # Возвращаем также исходное изображение для вырезки контуров


def save_contours(contours, original_image, output_dir="Squares"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, contour in enumerate(contours):
        # Создание ограничивающего прямоугольника для каждого контура
        x, y, w, h = cv2.boundingRect(contour)
        # Вырезка области контура из исходного изображения
        contour_image = original_image[y:y+h, x:x+w]
        # Сохранение изображения контура
        contour_image_path = os.path.join(output_dir, f"contour_{i}.png")
        cv2.imwrite(contour_image_path, contour_image)


def find_corner():
    
    

    return 


def main():
    image_path = 'C:\Study\square_rotating_test\images\sq_test.png'
    output_image, contours, original_image = detect_white_shapes(image_path)
    
    # Сохранение контуров в качестве отдельных изображений
    save_contours(contours, original_image)
    find_corner(contours)

    # Показ результата
    cv2.imshow('Detected White Shapes', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
