from functions import *# не используется

name_of_picture = "p5.jpg"

# s1 = main(name_of_picture, "round", 90, 90 / 6, 1)  # name, type_is, zero_angle, one_angle, one_angle_cost
# print(s1)
s2 = sq_method(name_of_picture)
print(s2)
s2 = res_from_angle(name_of_picture, "p5", 90, 90 / 10, 5, s2)  # доделать формулу вычисления угла
print(s2)
