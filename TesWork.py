import pytesseract
from PIL import Image

#На вход получает кортеж из квадратов и координат квадратов
#Если не удается распознать символ - удаляет его и его координаты
#Возвращает кортеж из номеров символов и координат квадратов

# Функция распознает армянский символ на изображении и возвращает его номер
def recognize_armenian_symbol(image_path):
    try:
        # Открываем изображение
        img = Image.open(image_path)

        # Используем pytesseract для распознавания текста на изображении
        text = pytesseract.image_to_string(img, lang='hye')

        # Армянский алфавит
        armenian_alphabet = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ"

        # Ищем первый символ армянского алфавита в тексте
        for char in text:
            if char in armenian_alphabet:
                return armenian_alphabet.index(char) + 1

        # Если символ не найден, возвращаем None
        return None

    except Exception as e:
        # В случае ошибки возвращаем None
        return None

# Основная функция, которая принимает кортеж изображений и возвращает кортеж с номерами символов
def CV_symbol_filter(images):
    result = []

    for image_path, latt, long in images:
        sym_num = recognize_armenian_symbol(image_path)
        
        # Если символ распознан, добавляем его в результат
        if sym_num is not None:
            result.append((sym_num, latt, long))

    # Преобразуем список в кортеж и возвращаем
    return tuple(result)