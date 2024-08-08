from GPSwork import getcoords #Железо
from CAMwork import getPhoto  #Железо
from OSwork import writeFiltredList #Железо
from CVwork import getSquares
from TesWork import CV_symbol_filter


def main():
    coords = [0, 0, 180] #Координаты long, latt, az
    current_photo = 0

    coords = getcoords()       #Получить координаты
    current_photo = getPhoto() #Получить фото, минимум задержка по времени

    squares_list = [(None, None, None), (None, None, None)] #(sq1, latt, long) Сами квадраты, полученные в текущем кадре. Содержит уникальные кадры квадратов и массив из повёрнутых кадров (1квадрат на фото = 4кв в этом листе)
    

    squares_list = getSquares(coords, current_photo)        #Получить список квадратов и их координаты

    filtred_list = [(None, None, None),(None, None, None)] #Символ и соотвествующие ему координаты

    filtred_list = CV_symbol_filter(squares_list)  #Получить номер символа, удалить неопозанные квадраты
    #Первый элемент кортежа - путь к файлу с квадратом!!!
    
    writeFiltredList(filtred_list)  #Ищет одни и те же координаты с схожими символами, усредняет координаты и записывает в файл

if __name__ == "__main__":
    main() #Потом запихнуть в while true