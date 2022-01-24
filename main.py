#BUSE AYYILDIZ 150170099

import os,glob
import cv2
from scipy.io import loadmat
from sklearn.metrics import precision_score


def precisionCalculation(img,groundTruth):
    i = img
    t = groundTruth
    prec = precision_score(i,t,average='micro')*100
    return prec

main_dir = 'test-groundtruth'
mats = []

images_groundTruth = []
images_canny = []
for file in os.listdir(main_dir): #reading mat files
    mats.append(loadmat(main_dir+'\\'+file))

images = []

os.chdir("test")
for file in glob.glob("*.jpg"):  #reading images
    image = cv2.imread(file)
    images.append(image)


for i in range (0,200):
  x = [[element for element in upperElement] for upperElement in mats[i]['groundTruth']]


for j in range (0,200):  #merging the boundaries
    img = mats[j]['groundTruth'][0][0][0][0][1]*255

    for i in range (1,len(mats[j]['groundTruth'][0])):

        img = cv2.add(img,mats[j]['groundTruth'][0][i][0][0][1]*255)
    images_groundTruth.append(img)

#canny edge detection algorithm
prec2 = 0
for z in range (0,200):
    image_gray = cv2.cvtColor(images[z], cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image_gray, 100, 200)
    b = precisionCalculation(images_groundTruth[z], edges)
    prec2 = prec2 + b


averagePrec = prec2/200
print(averagePrec)