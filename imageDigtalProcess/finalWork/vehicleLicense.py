import cv2
import numpy as np
import re
from paddleocr import PaddleOCR


def PlateRecognize(filename, cardColor):  # 车牌识别

    # 1、图片加载和预处理
    img = cv2.imread(filename)

    # 灰度化处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 高斯模糊
    GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=0)

    # 轮廓检测
    canny = cv2.Canny(GaussianBlur_img, 150, 255)

    # 图像二值化
    ret, binary_img = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary_img")

    # 2、形态学运算
    kernel = np.ones((5, 15), np.uint8)

    # 先闭运算将车牌数字部分连接，再开运算将不是块状的或是较小的部分去掉
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)

    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    dilation_img = cv2.dilate(open_img, np.ones(shape=[5, 5], dtype=np.uint8), iterations=3)

    # 3、获取轮廓
    contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for c in contours:
        x = []
        y = []
        for point in c:
            y.append(point[0][0])
            x.append(point[0][1])
        r = [min(y), min(x), max(y), max(x)]
        rectangles.append(r)

    # 4、根据HSV颜色空间查找汽车上车牌的位置
    dist_r = []
    max_mean = 0
    for r in rectangles:
        block = img[r[1]:r[3], r[0]:r[2]]
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        if cardColor == "蓝色":
            low = np.array([100, 60, 60])
            up = np.array([140, 255, 255])

        if cardColor == "绿色":
            low = np.array([70, 43, 46])
            up = np.array([77, 255, 255])

        if cardColor == "黄色":
            low = np.array([15, 55, 55])
            up = np.array([50, 255, 255])
        result = cv2.inRange(hsv, low, up)
        # 用计算均值的方式找车牌颜色最多的区块
        mean = cv2.mean(result)
        if mean[0] > max_mean:
            max_mean = mean[0]
            dist_r = r

    # 画出识别结果，由于之前多做了一次膨胀操作，导致矩形框稍大了一些，因此这里对于框架+3-3可以使框架更贴合车牌
    cv2.rectangle(img, (dist_r[0] + 3, dist_r[1]), (dist_r[2] - 3, dist_r[3]), (0, 255, 0), 2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("img", img)

    img = img[dist_r[1]:dist_r[3], dist_r[0]:dist_r[2]]

    return dist_r, img


# 车牌文字识别
def characterRecognize(img):
    ocr = PaddleOCR(lang='ch')
    result = ocr.ocr(img)
    for line in result:
        print(line)
    return result[0]


# 正则表达式匹配汉字字母和数字
def extract_first_chinese_others_alphanumeric(string):
    # # 首位匹配汉字 其他位置为字母或数字
    pattern = '[\u4e00-\u9fa5|A-Za-z0-9]+'
    result = re.findall(pattern, string)
    return result


# 颜色识别
def getColor(filename):
    img_path = cv2.imread(filename)
    # cv2.imshow('origin', img_path)

    height = img_path.shape[0]
    width = img_path.shape[1]
    print('面积：', height * width)

    # 设定阈值
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    lower_green = np.array([0, 3, 116])
    upper_green = np.array([76, 211, 255])

    # 转换为HSV
    hsv = cv2.cvtColor(img_path, cv2.COLOR_BGR2HSV)

    # 根据阈值构建掩膜
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 对原图像和掩膜进行位运算
    # src1：第一个图像（合并的第一个对象）src2：第二个图像（合并的第二个对象）mask：理解为要合并的规则。
    res_blue = cv2.bitwise_and(img_path, img_path, mask=mask_blue)
    res_yellow = cv2.bitwise_and(img_path, img_path, mask=mask_yellow)
    res_green = cv2.bitwise_and(img_path, img_path, mask=mask_green)

    # 显示图像
    # cv2.imshow('frame', img_path)
    # cv2.imshow('mask_blue', mask_blue)
    # cv2.imshow('mask_yellow', mask_yellow)
    # cv2.imshow('mask_green', mask_green)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)

    # 对mask进行操作--黑白像素点统计  因为不同颜色的掩膜面积不一样
    # 记录黑白像素总和
    blue_white = 0
    blue_black = 0
    yellow_white = 0
    yellow_black = 0
    green_white = 0
    green_black = 0

    # 计算每一列的黑白像素总和
    for i in range(width):
        for j in range(height):
            if mask_blue[j][i] == 255:
                blue_white += 1
            if mask_blue[j][i] == 0:
                blue_black += 1
            if mask_yellow[j][i] == 255:
                yellow_white += 1
            if mask_yellow[j][i] == 0:
                yellow_black += 1
            if mask_green[j][i] == 255:
                green_white += 1
            if mask_green[j][i] == 0:
                green_black += 1

    print('蓝色--白色 = ', blue_white)
    print('蓝色--黑色 = ', blue_black)
    print('黄色--白色 = ', yellow_white)
    print('黄色--黑色 = ', yellow_black)
    print('绿色--白色 = ', green_white)
    print('绿色--黑色 = ', green_black)

    color_list = ['蓝色', '黄色', '绿色']
    num_list = [blue_white, yellow_white, green_white]

    print('车牌的颜色为:', color_list[num_list.index(max(num_list))])
    return color_list[num_list.index(max(num_list))]


def deal_license(car_plate):
    # 灰
    gray_img = cv2.cvtColor(car_plate, cv2.COLOR_RGBA2GRAY)

    # 均值滤波 去除噪声
    kernel = np.ones((3, 3), np.float32) / 9
    gray_img = cv2.filter2D(gray_img, -1, kernel)

    ret, thresh = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

    return thresh


# 主程序
# if __name__ == '__main__':
#     # 框出车牌号
#     dist, car_plate = PlateRecognize("2.jpg")
#     cv2.imshow("cut", car_plate)
#     thresh = deal_license(car_plate)
#     cv2.imshow("test", thresh)
#     res = characterRecognize(car_plate)
#     str = res[0][0][1][0]
#     print(str)
#     num = extract_first_chinese_others_alphanumeric(str)
#     print(num)
#     cv2.waitKey(0)

