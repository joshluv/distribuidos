from sparkdl import readImages
from pyspark.sql.functions import lit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from os import listdir
import numpy as np
import sys
import cv2

class tiffSpark:

    @staticmethod
    def define_trainTest(original_df, otsu_df):
        original_dfTrain, original_dfTest = original_df.randomSplit([0.6, 0.4])
        # dataTrain = original_dfTrain.unionAll(original_dfTrain)
        return original_dfTrain, original_dfTest

    @staticmethod
    def getImageMat(imageMat):
        imageMat = cv2.imread(imageMat, cv2.IMREAD_UNCHANGED)
        try:
            if imageMat.shape[2]:
                imageMat = cv2.cvtColor(imageMat, cv2.COLOR_BGR2GRAY)
        except IndexError:
            pass
        return imageMat

    @staticmethod
    def load_dataset(originalImg, otsuImg, dataArray, labelArray, iterator):
        Img, ImgData = getImageMat(otsuImg), getImageMat(originalImg)
        rows, columns = Img.shape[0], Img.shape[1]
        print(rows, columns)  # dimensionalidad                              # número de ejemplos
        # iterator = 0
        # dataArray, labelArray = np.zeros(arraySize), np.zeros(arraySize)

        for pixelX in range(int(rows)):
            for pixelY in range(columns):
                if Img[pixelX, pixelY] == 0:
                    # labels[pixelX, pixelY] = 1
                    labelArray[iterator] = 1
                else:
                    # labels[pixelX, pixelY] = 0
                    labelArray[iterator] = 0
                dataArray[iterator] = ImgData[pixelX, pixelY]
                # dataArray.append(ImgData[pixelX, pixelY])
                iterator += 1

        return dataArray, labelArray, iterator

    @staticmethod
    def read_imageDir(img_dir):
        original_df = readImages(img_dir + "/jobs").withColumn("label", lit(1))
        otsu_df = readImages(img_dir + "/zuckerberg").withColumn("label", lit(0))
        return original_df, otsu_df

    @staticmethod
    def train_model():
        lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
        p = Pipeline(stages=[lr])
        p_model = p.fit(train_df)

if __name__ == '__main__':
    originalPath = sys.argv[1]
    otsuPath = sys.argv[2]

    #originalPath = r"/content/gdrive/My Drive/Maestria/4to semestre/villalon-analisis/dATASET/Original"
    originalFiles = [f for f in listdir(originalPath)]
    originalFiles.sort()
    print(originalFiles)

    #otsuPath = r"/content/gdrive/My Drive/Maestria/4to semestre/villalon-analisis/dATASET/OtsuProcessed"
    otsuFiles = [f for f in listdir(otsuPath)]
    otsuFiles.sort()
    print(otsuFiles)

    arraySize = ((8151 * 7131) * 4) + (1956 * 960) + (3913 * 1921)
    trainOriginal, trainEtiquetas = np.empty(arraySize), np.empty(arraySize)
    print(trainOriginal.shape)
    iterator = 0

    for imageIt in range(len(originalFiles)):
        originalImg = "{}/{}".format(originalPath, originalFiles[imageIt])
        print("image: ", originalImg)
        otsuImg = "{}/{}".format(otsuPath, otsuFiles[imageIt])
        trainOriginal, trainEtiquetas, iterator = load_dataset(originalImg, otsuImg, trainOriginal, trainEtiquetas,
                                                               iterator)