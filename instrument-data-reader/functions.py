import cv2 as cv  # файл с функциями, которые могут применяться в других частях работы
import numpy as np
import math
import os


print ("n")
class PrMain(object):  # класс для сохранения информации о приборе, не используется, для перспективной разработки
    """Класс прибора"""

    def __init__(self, name, picture, type_is, zero_angle, one_angle, one_angle_cost):
        """Constructor"""
        self.name = name  # имя прибора
        self.picture = picture  # картинка
        self.type_is = type_is  # тип прибора, круглый или квадратный, пока не используется
        self.zero_angle = zero_angle  # угол на ноль, град. За абсолютный ноль принят горизонтальный угол н
        self.one_angle = one_angle  # угол одного деления, град
        self.one_angle_cost = one_angle_cost  # цена деления№


def main(name_of_picture, type_is, zero_angle, one_angle, one_angle_cost):  # основная функция метода моментов
    global atn
    P1 = PrMain(name_of_picture, cv.imread(name_of_picture, cv.IMREAD_GRAYSCALE), type_is, zero_angle, one_angle,
                one_angle_cost)
    img = P1.picture
    show(img, P1.name)

    try:
        img_p = get_ready2(img)

        show(img_p, P1.name)
        img_p = resize(img_p)

        find_tg(img_p)
    except:
        atn = sq_method(name_of_picture)
        tan = ""

    result = res_from_angle(name_of_picture, type_is, zero_angle, one_angle, one_angle_cost, atn)
    return "Тангенс", tan, "Угол:", atn, "(град)", "Прибор показывает:", result, "+/-", P1.one_angle_cost / 2


def show(image, name):  # показать картинку
    cv.imshow(name, image)
    cv.waitKey(0)


def aver(x, y):
    return int((x + y) // 2)


def get_ready(img, gauss_cell_size1, gauss_cell_size2, gauss_mov, dual_treshold):  # подготовка картинки через гаусса
    # и дуализацию, работает корректно, но нужно подбиирать все
    # параметры вручную
    img_g = cv.GaussianBlur(img, (gauss_cell_size1, gauss_cell_size2), gauss_mov)
    ret, image_dw = cv.threshold(img_g, dual_treshold, 255, 60)
    return image_dw


def get_ready2(img):  # адаптивное ядро гаусса
    img = cv.GaussianBlur(img, (25, 25), 0)
    image_dw = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 20)

    show(img, "g")

    return image_dw


def find_wd_gate(ar):  # ищет переход с черного на белый пиксель - это и будет внутренней границей корпуса
    for i in range(len(ar)):
        if ar[i] == 0 and ar[i + 1] == 255:
            r = i + 1
            return r  # возвращает первый переход чб в массиве, работает корректно


def get_wd_gates(img_p):
    h = img_p.shape[0]  # находим переходы из черного в белое по 4 сторонам, работает корректно
    w = img_p.shape[1]
    x = [0, 0, 0, 0]
    y = [0, 0, 0, 0]
    a = list(img_p[1:h, w // 2])  # вертикаль
    y[0] = find_wd_gate(a)
    a.reverse()
    y[2] = h - find_wd_gate(a)
    a = list(img_p[h // 2, 1:w])  # горизонталь
    x[1] = find_wd_gate(a)
    a.reverse()
    x[3] = w - find_wd_gate(a)
    x[0] = x[2] = w // 2
    y[1] = y[3] = h // 2
    return x, y


def xy_aver(x, y):
    avx = [0, 0, 0, 0]  # среднее из пар пикселей, работаеет корректно
    avy = [0, 0, 0, 0]
    for i in range(4):
        avx[i] = aver(x[i - 1], x[i])
        avy[i] = aver(y[i - 1], y[i])
    x1 = min(avx)
    x2 = max(avx)
    y1 = min(avy)
    y2 = max(avy)
    return x1, y1, x2, y2


def resize(img):  # обрезка
    x, y = get_wd_gates(img)  # 02 верт 13 гор
    x1, y1, x2, y2 = xy_aver(x, y)

    img = img[y1:y2, x1:x2]  # обрезка
    show(img, "0")
    return img


def find_tg(img):  # обрезает картинку по найденным точкам и ищет тангенс стрелочки, возвращает арктангенс
    # не работает
    img = resize(img)
    show(img, "")

    height = img[0]
    width = img[1]

    M = cv.moments(img)
    # print(M)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    img1 = img[0:height // 2, 0:width]
    # M1 = cv.moments(img_p[y1:aver(y1, y2), x1:x2]) # верхняя полукартинка
    M1 = cv.moments(img1)
    # print(M1)
    cX1 = int(M1["m10"] / M1["m00"])
    cY1 = int(M1["m01"] / M1["m00"])
    show(img1, "1")

    img2 = img[height // 2:height, 0:width]
    M2 = cv.moments(img2)
    # M2 = cv.moments(img_p[aver(y1, y2):y2, x1:x2]) # нижняя полукартинка
    # print(M2)
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])
    show(img2, "2")
    print("0p", cX, cY, "1p", cX1, cY1, "2p", cX2, cY2)

    try:
        tan = (cY1 - cY) / (cX1 - cX)  # проверить
        atn = math.degrees(math.atan(tan))  # моментов полукартинок и всей картинки), скорее всего содержит лажу !!!
    except:
        atn = 90
        tan = "n/e"
        print("ex1")
    return atn, tan


def res_from_angle(name_of_picture, type_is, zero_angle, one_angle, one_angle_cost, atn):  # пример формулы
    # преобразования угла в ответ, актуально только для одной из картинок
    P1 = PrMain(name_of_picture, cv.imread(name_of_picture, cv.IMREAD_GRAYSCALE), type_is, zero_angle, one_angle,
                one_angle_cost)

    result = (P1.zero_angle - atn) / P1.one_angle * P1.one_angle_cost * P1.one_angle_cost // P1.one_angle_cost  # !!!

    return "Угол:", atn, "(град)", "Прибор показывает:", result, "+/-", P1.one_angle_cost / 2


def sq_method(name_of_picture):
    img = cv.imread(name_of_picture)

    hsv_min = np.array((0, 50, 50), np.uint8)
    hsv_max = np.array((10, 255, 255), np.uint8)

    # hold = height * width // 5000
    hold = 0

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    angles = []
    areas = []

    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        sx = sy = 0
        for i in range(4):
            sx = sx + box[i][0]
            sy = sx + box[i][1]

        center_of_box = (sx // 4, sy // 4)

        is_invert = (box[1][1] > center_of_box[1])

        area = int(rect[1][0] * rect[1][1])

        if area >= hold:
            cv.drawContours(img, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник
            edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

            if cv.norm(edge2) > cv.norm(edge1):
                usedEdge = edge2
            else:
                usedEdge = edge1

            reference = (1, 0)
            angle = 180.0 / math.pi * math.acos((reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (
                    cv.norm(reference) * cv.norm(usedEdge)))  # угол с горизонтом
            if is_invert:
                angle = 180 + angle
            areas.append(area)
            angles.append(angle)

    cv.imshow('contours', img)  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()

    area_of_angle_of_biggest_box = 0
    for i in range(len(angles)):
        if area_of_angle_of_biggest_box < areas[i]:
            area_of_angle_of_biggest_box = areas[i]
            angle_of_biggest_box = angles[i]

    return angle_of_biggest_box


def findfiles():  # ищет изображения с заданными расширениями в текущей папке, не используется
    res = []
    subs = [".jfif", ".jpg", ".cam"]
    for file in os.listdir(os.getcwd()):
        for sub in subs:
            if file.endswith(sub):
                res.append(os.path.join(os.getcwd(), file))
    return res
