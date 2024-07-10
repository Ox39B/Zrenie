import cv2
import numpy as np
import math
import pytesseract
import time
from collections import deque
import os
import threading

# Настройка пути к исполняемому файлу Tesseract (если требуется)
pytesseract.pytesseract.tesseract_cmd = r'D:\\Tesseract\\tesseract.exe'

def find_squares(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            squares.append(approx)
    return squares

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_camera_frames(stop_event):
    # Инициализация камеры
    cap = cv2.VideoCapture(0)

    # Очередь для хранения последних 7 скриншотов
    frames = deque(maxlen=7)

    # Создание папки для сохранения скриншотов
    output_dir = "Squares"
    os.makedirs(output_dir, exist_ok=True)

    # Счетчик для нумерации файлов
    file_counter = 1

    try:
        while not stop_event.is_set():
            start_time = time.time()
            for _ in range(7):
                ret, frame = cap.read()
                if not ret:
                    print("Не удалось захватить изображение")
                    continue
                frames.append(frame)

                # Обработка каждого кадра
                squares = find_squares(frame)
                if squares:
                    for j, square in enumerate(squares):
                        warped = four_point_transform(frame, square.reshape(4, 2))
                        output_path = os.path.join(output_dir, f'{file_counter}.jpg')
                        cv2.imwrite(output_path, warped)
                        file_counter += 1
                        text = pytesseract.image_to_string(warped, lang='eng')
                        print(f"Текст на квадрате {j}: {text}")

                # Обводим найденные квадраты на изображении
                cv2.drawContours(frame, squares, -1, (0, 255, 0), 3)

                # Отображаем изображение с обведенными квадратами
                cv2.imshow('Camera View', frame)

                # Ожидание перед захватом следующего кадра
                elapsed_time = time.time() - start_time
                time.sleep(max(0, (1 / 7) - elapsed_time))

            print(f"Обработано {len(squares)} квадратов в текущем кадре.")

            # Прерывание по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("Захват завершен.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stop_event = threading.Event()

    # Создание и запуск потока для обработки кадров с камеры
    camera_thread = threading.Thread(target=process_camera_frames, args=(stop_event,))
    camera_thread.start()

    try:
        # Основной поток продолжает выполнение, обрабатывая события окна
        while not stop_event.is_set():
            time.sleep(0.1)  # Небольшая задержка для снижения нагрузки на CPU

    except KeyboardInterrupt:
        print("Завершение программы.")
        stop_event.set()
    finally:
        camera_thread.join()
        cv2.destroyAllWindows()
