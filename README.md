# 数字图像处理HW3
- 林汉宁  自动化52 2150504042
- 提交日期：2019-03-18
## 摘要
本次作业使用python3.67面向对象方式编程，使用了opencv与numpy等库。
## 技术讨论
### 1. 直方图画出
#### 源代码
	def drawHist(self):
        h = np.zeros((256,256,3))
        bins = np.arange(256).reshape(256,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ] #BGR三种颜色  
        for ch, col in enumerate(color):  
            originHist = cv.calcHist([self.img],[ch],None,[256],[0,256])  
            cv.normalize(originHist, originHist,0,255*0.9,cv.NORM_MINMAX)  
            hist=np.int32(np.around(originHist))  
            pts = np.column_stack((bins,hist))  
            cv.polylines(h,[pts],False,col) 
        h=np.flipud(h)
        # cv.imshow("Hist",h)
        cv.imwrite('Hist_of_{}.bmp'.format(self.name),h)
        print("[LOG]:Hist_of_{}.bmp created Successfully!".format(self.name))
#### 思路与结果
使用cv2.calcHist求出直方图，再使用cv2.polylines函数将直方图画出
如下，选择几张直方图展示：
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/Hist_of_elain2.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/Hist_of_elain.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/Hist_of_elain3.bmp)

### 2.  直方图均衡 
#### 源代码
	def equalizeH(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        dst=cv.equalizeHist(gray)
        #test
        # cv.imshow("Equalized",dst)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite('2-equalized_of_{}.bmp'.format(self.name),dst)
        print("[LOG]:2-equalized_of_{}.bmp created successfully!".format(self.name))
#### 代码思路
首先使用opencv库函数quaequalizeHist可实现直方图均衡
#### 结果
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/2-equalized_of_woman2.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/2-equalized_of_woman1.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/2-equalized_of_woman.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/2-equalized_of_lena4.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/2-equalized_of_lena2.bmp)



### 3.计算lena图像的均值方差
#### 源代码
	def histogramMatching(self,reference_img):
        self.target_img = self.img
        # reference_img is the one to be learned
        reference_histogram = ExactHistogramMatcher.get_histogram(reference_img)
        new_target_img = ExactHistogramMatcher.match_image_to_histogram(self.target_img, reference_histogram)
        new_target_img = new_target_img.astype(np.uint16)
        print("[LOG]:Pic \"3-histogrammatching_of_{}.bmp\"lib called successfully!".format(self.name))
        self.target_img=new_target_img
        cv.imwrite('3-histogrammatching_of_{}.bmp'.format(self.name),self.target_img)
        print("[LOG]:3-histogrammatching_of_{}.bmp created sucessfully!".format(self.name))
#### 代码思路
在此感谢Stefano Di Martino 所编写的开源直方图匹配库函数，其原理是分别计算两幅图像的随机变量累积分布函数值，之后在之间建立映射关系，对于原图像的每一个值，对应另一函数中中索引为一像素值。通过这种方式可以将原图像处理为近似于模板的直方图分布的图像。
#### 最终结果
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/3-histogrammatching_of_lena.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/3-histogrammatching_of_elain.bmp)


### 4.局部图像增强  
#### 源代码
	def LocalEnhancement(self):
        E = 2
        para0 = 0.4
        para1 = 0.02
        para2 = 0.4
        height,width,channels = self.img.shape
        new_img = np.zeros([height, width], np.uint8)
        (mean, stddev) = cv.meanStdDev(self.img)
        var = stddev * stddev
        for i in range(height):
            for j in range(width):
                localm = 0
                localv = 0
                for k in range(i-3,i+4):
                    if k < 0:
                        kk = abs(k)
                    elif k > height-1:
                        kk = 2*height - 1 - k
                    else:
                        kk = k
                    for l in range(j-3,j+4):
                        if l < 0:
                            ll = abs(l)
                        elif l > width-1:
                            ll = 2*width-1 - l
                        else:
                            ll = l
                        localm = localm + self.img[kk][ll][0]
            localm = localm/(7*7)
            for k in range(i-3,i+4):
                if k < 0:
                    kk = abs(k)
                elif k > height-1:
                    kk = 2*height-1 - k
                else:
                    kk = k
                for l in range(j-3,j+4):
                    if l < 0:
                        ll = abs(l)
                    elif l > width-1:
                        ll = 2*width-1 - l
                    else:
                        ll = l
                    localv = localv + (localm - self.img[kk][ll][0]) ** 2
            localv = localv/48
            if localm <= para0 * mean[0] and localv >= para1 * var[0] and localv <= para2 * var[0]:
                new_img[i][j] = E * self.img[i][j][0]
            else:
                new_img[i][j] = self.img[i][j][0]
        cv.imwrite('4-Local_Enhancement_of_{}.bmp'.format(self.name),new_img)
        print("[LOG]:4-Local_Enhancement_of_{}.bmp created sucessfully!".format(self.name))
#### 代码思路
首先计算均值与方差，计算区域局部方差均值。找到较暗区域后增强区域
#### 部分结果展示
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/4-Local_Enhancement_of_elain.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/4-Local_Enhancement_of_lena.bmp)


### 5、图像分割
#### 源代码
	def histogramSegementation(self):
        gray=cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        ret2,th2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imwrite('5-histogramSegmentation_of_{}.bmp'.format(self.name),th2)
        print("[LOG]:5-histogramSegmentation_of_{}.bmp created sucessfully!".format(self.name))
#### 代码思路
使用opencv threshold函数可实现双峰法直方图图像分割
#### 结果展示
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/5-histogramSegmentation_of_woman.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/5-histogramSegmentation_of_lena.bmp)
![Alt text](https://github.com/HanningLin/hw3/blob/master/img/5-histogramSegmentation_of_elain.bmp)

