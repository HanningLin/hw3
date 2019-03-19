#project3-linhanning-auto52-2150504042
import cv2 as cv
import numpy as np
from histogram_matching import ExactHistogramMatcher
#OOP create class Pic
class Pic:
    def __init__(self, name, path):
         self.name=name
         self.path=path
         self.img=cv.imread(path)
         self.target_img=self.img
         print("[LOG]:Object {} created successfully!".format(self.name))
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
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    def equalizeH(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        dst=cv.equalizeHist(gray)
        #test
        # cv.imshow("Equalized",dst)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite('2-equalized_of_{}.bmp'.format(self.name),dst)
        print("[LOG]:2-equalized_of_{}.bmp created successfully!".format(self.name))
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
        # # misc.imsave('F:/grey_out.png', new_target_img)
        # filename = 'F:/grey_out.png'
        # with open(filename, 'wb') as f:
        # writer = png.Writer(width=new_target_img.shape[1], height=new_target_img.shape[0], bitdepth=16, greyscale=True)
        # zgray2list = new_target_img.tolist()
        # writer.write(f, zgray2list)
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
        
            

    def histogramSegementation(self):
        gray=cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        ret2,th2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imwrite('5-histogramSegmentation_of_{}.bmp'.format(self.name),th2)
        print("[LOG]:5-histogramSegmentation_of_{}.bmp created sucessfully!".format(self.name))
        
#3-1
print("-------------------------------------")
print("#ANS:3-1 Show Histgram\n")
elain=Pic("elain","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/elain.bmp")
elain1=Pic("elain1","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/elain1.bmp")
elain2=Pic("elain2","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/elain2.bmp")
elain3=Pic("elain3","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/elain3.bmp")
citywall=Pic("citywall","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/citywall.bmp")
citywall1=Pic("citywall1","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/citywall1.bmp")
citywall2=Pic("citywall2","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/citywall2.bmp")
lena=Pic("lena","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/lena.bmp")
lena1=Pic("lena1","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/lena1.bmp")
lena2=Pic("lena2","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/lena2.bmp")
lena4=Pic("lena4","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/lena4.bmp")
woman=Pic("woman","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/woman.bmp")
woman1=Pic("woman1","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/woman1.bmp")
woman2=Pic("woman2","/home/hanninglin/Documents/CV/PROJECT/hw3/3rd-pro/woman2.bmp")
elain.drawHist()
elain1.drawHist()
elain2.drawHist()
elain3.drawHist()
citywall.drawHist()
citywall1.drawHist()
citywall2.drawHist()
lena.drawHist()
lena1.drawHist()
lena2.drawHist()
lena4.drawHist()
woman.drawHist()
woman1.drawHist()
woman2.drawHist()
3-2
print("-------------------------------------")
print("#ANS:3-2 equalized Based on Histgram\n")
elain.equalizeH()
elain1.equalizeH()
elain2.equalizeH()
elain3.equalizeH()
citywall.equalizeH()
citywall1.equalizeH()
citywall2.equalizeH()
lena.equalizeH()
lena1.equalizeH()
lena2.equalizeH()
lena4.equalizeH()
woman.equalizeH()
woman1.equalizeH()
woman2.equalizeH()
print("-------------------------------------")
print("#ANS:3-3 Histgram matching\n")
elain.histogramMatching(lena2.img)
lena.histogramMatching(lena1.img)
print("-------------------------------------")
print("#ANS:3-4 Local Enhancement\n")
elain.LocalEnhancement()
lena.LocalEnhancement()
print("-------------------------------------")
print("#ANS:3-5 Histogram Segementation\n")
elain.histogramSegementation()
woman.histogramSegementation()


























# elain
# elain1
# elain2
# elain3
# citywall
# citywall1
# citywall2
# lena
# lena1
# lena2
# lena4
# woman
# woman1
# woman2