import cv2
import numpy as np

# # img_path = cv2.imread('./rest/yellow.png')
# # img_path = cv2.imread('./rest/blue.jpg')
# img_path = cv2.imread('8.jpg')
# cv2.imshow('origin', img_path)
#
# height = img_path.shape[0]
# width = img_path.shape[1]
# print('面积：', height * width)
#
#
# # 设定阈值
# lower_blue = np.array([100, 43, 46])
# upper_blue = np.array([124, 255, 255])
# lower_yellow = np.array([15, 55, 55])
# upper_yellow = np.array([50, 255, 255])
# lower_green = np.array([0, 3, 116])
# upper_green = np.array([76, 211, 255])
#
# # 转换为HSV
# hsv = cv2.cvtColor(img_path, cv2.COLOR_BGR2HSV)
#
# # 根据阈值构建掩膜
# mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
# mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)  #
# mask_green = cv2.inRange(hsv, lower_green, upper_green)  #
#
# # 对原图像和掩膜进行位运算
# # src1：第一个图像（合并的第一个对象）src2：第二个图像（合并的第二个对象）mask：理解为要合并的规则。
# res_blue = cv2.bitwise_and(img_path, img_path, mask=mask_blue)
# res_yellow = cv2.bitwise_and(img_path, img_path, mask=mask_yellow)
# res_green = cv2.bitwise_and(img_path, img_path, mask=mask_green)
#
# # 显示图像
# # cv2.imshow('frame', img_path)
# cv2.imshow('mask_blue', mask_blue)
# cv2.imshow('mask_yellow', mask_yellow)
# cv2.imshow('mask_green', mask_green)
# # cv2.imshow('res', res)
#
# # 对mask进行操作--黑白像素点统计  因为不同颜色的掩膜面积不一样
# # 记录黑白像素总和
#
# blue_white = 0
# blue_black = 0
# yellow_white = 0
# yellow_black = 0
# green_white = 0
# green_black = 0
#
# # 计算每一列的黑白像素总和
# for i in range(width):
#     for j in range(height):
#         if mask_blue[j][i] == 255:
#             blue_white += 1
#         if mask_blue[j][i] == 0:
#             blue_black += 1
#         if mask_yellow[j][i] == 255:
#             yellow_white += 1
#         if mask_yellow[j][i] == 0:
#             yellow_black += 1
#         if mask_green[j][i] == 255:
#             green_white += 1
#         if mask_green[j][i] == 0:
#             green_black += 1
#
# print('蓝色--白色 = ', blue_white)
# print('蓝色--黑色 = ', blue_black)
# print('黄色--白色 = ', yellow_white)
# print('黄色--黑色 = ', yellow_black)
# print('绿色--白色 = ', green_white)
# print('绿色--黑色 = ', green_black)
#
# color_list = ['蓝色','黄色','绿色']
# num_list = [blue_white,yellow_white,green_white]
#
# print('车牌的颜色为:',color_list[num_list.index(max(num_list))])
#
# cv2.waitKey(0)
#


import tkinter as tk
from tkinter import filedialog
import vehicleLicense as vl
root = tk.Tk()

root.title("车牌识别")
root.geometry("720x360")

b = 0


def addPic():  # 一个方法，每次按button就给b+1
    select_file_path = tk.filedialog.askopenfilename()
    colors , imgs = vl.PlateRecognize(select_file_path)
    for img in imgs:
        cv2.imshow("result",img)
        cv2.waitKey(0)

def showProcess():
    print("展示过程")

a = tk.Button(root, text="读取照片", command=addPic).pack()  # button，这里面的command就是调用前面的addOne方法

b = tk.Button(root,text="展示过程", command=showProcess).pack()


root.mainloop()  # 不能少的东西
