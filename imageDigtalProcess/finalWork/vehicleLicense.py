import cv2
import numpy as np
from paddleocr import PaddleOCR


def PlateRecognize(filename):      #车牌识别

    # 1、图片加载和预处理
    img = cv2.imread(filename)

    # 灰度化处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 高斯模糊
    GaussianBlur_img = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=0)

    # 轮廓检测
    canny = cv2.Canny(GaussianBlur_img,150,255)

    # 图像二值化
    ret, binary_img = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary_img")

    # 2、形态学运算
    kernel = np.ones((5, 15), np.uint8)

    # 先闭运算将车牌数字部分连接，再开运算将不是块状的或是较小的部分去掉
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)

    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    dilation_img = cv2.dilate(open_img, np.ones(shape= [5,5],dtype=np.uint8), iterations=3)

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
        low = np.array([100, 60, 60])
        up = np.array([140, 255, 255])
        result = cv2.inRange(hsv, low, up)
        # 用计算均值的方式找蓝色最多的区块
        mean = np.mean(result)
        mean = cv2.mean(result)
        if mean[0] > max_mean:
            max_mean = mean[0]
            dist_r = r

    # 画出识别结果，由于之前多做了一次膨胀操作，导致矩形框稍大了一些，因此这里对于框架+3-3可以使框架更贴合车牌
    cv2.rectangle(img, (dist_r[0] + 3, dist_r[1]), (dist_r[2] - 3, dist_r[3]), (0, 255, 0), 2)

    img = img[dist_r[1]:dist_r[3],dist_r[0]:dist_r[2]]

    return dist_r,img

# 车牌文字识别

def characterRecognize(img):
     ocr = PaddleOCR(lang='ch')
     result = ocr.ocr(img)
     for line in result:
         print(line)

def deal_license(car_plate):

    #灰
    gray_img = cv2.cvtColor(car_plate,cv2.COLOR_RGBA2GRAY)

    #均值滤波 去除噪声
    kernel = np.ones((3,3),np.float32)/9
    gray_img = cv2.filter2D(gray_img,-1,kernel)

    ret,thresh = cv2.threshold(gray_img,120,255,cv2.THRESH_BINARY)

    return thresh

# 主程序
if __name__ == '__main__':
    #框出车牌号
    dist,car_plate = PlateRecognize("3.jpg")
    cv2.imshow("cut",car_plate)
    thresh = deal_license(car_plate)
    cv2.imshow("test", thresh)
    cv2.waitKey(0)


    characterRecognize(car_plate)





