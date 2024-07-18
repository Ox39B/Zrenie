import cv2
import time
import numpy as np
from square_rotating_test import find_squares, order_points, four_point_transform, is_contour_white


def load_image(file_path):
    return cv2.imread(file_path)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def apply_morphological_operations(thresh):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)
    return dilate

def find_and_draw_contours(frame, dilated):

    cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
    for c in cnts:
        if is_contour_white(frame, c):
            area = cv2.contourArea(c)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)


def display_images(thresh, dilated, frame):
    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilated)
    cv2.imshow('frame', frame)
    cv2.waitKey()

def main():
    image_path = 'sq_test.png'

    frame = load_image(image_path)
    
    thresh = preprocess_image(frame)

    dilated = apply_morphological_operations(thresh)
    
    find_and_draw_contours(frame, dilated)
    
    display_images(thresh, dilated, frame)

    
armenian_alphabet = 'ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ'

# Windows
# pytesseract.pytesseract.tesseract_cmd = r'take_path'

# custom_config = r'--oem 3 --psm 10 -l hye'
# details = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)


if __name__ == "__main__":
    main()

