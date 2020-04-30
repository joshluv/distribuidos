import cv2
import math
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType
import sys
import os

class imageProcessing:

    @staticmethod
    def get_grayClasses(landsatGrayImage):
        rows, columns = landsatGrayImage.shape
        grayClasses = set()

        for pixelX in range(rows):
            for pixelY in range(columns):
                #print(landsatGrayImage[pixelX, pixelY])
                grayClasses.add(landsatGrayImage[pixelX, pixelY])
        return grayClasses

    @staticmethod
    def get_grayClassesProbability(grayClasses, landsatGrayImage):
        P = []
        rows, columns = landsatGrayImage.shape

        for grayClass in grayClasses:
            P.append(grayClass / (rows * columns))
        return P

    @staticmethod
    def get_momentum(P):
        omega = []
        omega.append(P[1])
        mu = []
        mu.append(0)

        for grayIndex in range(1, len(P)):
            omega.append(omega[grayIndex - 1] + P[grayIndex])
            mu.append(mu[grayIndex - 1] + ((grayIndex-1)*P[grayIndex]))
        return mu, omega



    @staticmethod
    def image_normalization(img, umbral):
        #print(landsatImage.shape[0])
        cv2.normalize(img, img, 0, umbral, cv2.NORM_MINMAX, -1)#poner antes de otsu
        #cv2.imshow("normaltiff", img)
        ret, img = cv2.threshold(img, 0, umbral, cv2.THRESH_BINARY)
        #cv2.thre

#        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #cv2.imwrite(r"..\imageOut\NoSquare.tif",img)
        return img

    @staticmethod
    def __otsu_thesholdMAX(mu, omega):
        sigmaMax = 0
        mut = mu[-1]


        for index in range(1, len(omega)):
            omega1 = omega[index]
            omega2 = 1 - omega1
            if omega1!=0 and omega2!=0:
                mu1 = mu[index] / omega1
                mu2 = (mut - mu[index]) / omega2

                sigma = omega1 * math.pow((mu1 - mut), 2) + omega2 * math.pow((mu2 - mut),2)
                if sigma > sigmaMax:
                    sigmaMax = sigma
                    umbralOptimo = index - 1
        return sigmaMax, umbralOptimo

    @staticmethod
    def otsu_threshold(landsatGrayImage):
        grayClasses = imageProcessing.get_grayClasses(landsatGrayImage) #Pi
        print("Gray Classes detected: ",len(grayClasses))
        P = imageProcessing.get_grayClassesProbability(grayClasses, landsatGrayImage)
        #print("Valor de  P calculado: ", P)
        mu, omega = imageProcessing.get_momentum(P)
        #print ("Mu:, ", mu)
        #print("Omega, ", omega)
        Max, umbral = imageProcessing.__otsu_thesholdMAX(mu, omega)
        print("FINAL Max-> ",Max, umbral)
        #Min, umbral = imageProcessing.__otsu_thesholdMAX(mu, omega)
        #print("FINAL Min -> ", Min, umbral)
        return umbral

    @staticmethod
    def main(landsatTiffImage, df_total):

        landsatImage = cv2.imread(landsatTiffImage, cv2.IMREAD_UNCHANGED)
        if imageProcessing.get_imageChannels(landsatImage):
            landsatGrayImage = cv2.cvtColor(landsatImage, cv2.COLOR_RGB2GRAY)
        else:
            landsatGrayImage = landsatImage
        umbral = imageProcessing.otsu_threshold(landsatGrayImage)
        landsatImage = imageProcessing.image_normalization(landsatGrayImage, umbral)
        return imageProcessing.createDataFrame(landsatImage, df_total)

    @staticmethod
    def createDataFrame(landsatImage, df_total):
        testOriginal = [landsatImage[pixelX, pixelY] for pixelX in range(landsatImage.shape[0]) for pixelY in range(landsatImage.shape[1])]
        #testOriginal = [1,2,3]
        schema = StructType([
            StructField('pixelsLabel', IntegerType(), True)
        ])
        rdd = spark.sparkContext.parallelize(testOriginal, numSlices=1000)
        if df_total is None:
            df_total = spark.createDataFrame(rdd, schema)
            df_total.show()
        else:
            dfOtsu = spark.createDataFrame(rdd, schema)
            df_total = df_total.union(dfOtsu)
        return df_total

    @staticmethod
    def get_imageChannels(img):
        try:
            return img.shape[2]
        except IndexError:
            return None


if __name__ == '__main__':
    originalPath = sys.argv[1]
    df_total = None
    sc = SparkContext('local')
    spark = SparkSession(sc)
    for landsatTiffImage in os.listdir(originalPath):
        originalImg = "{}\{}".format(originalPath, landsatTiffImage)
        df_total = imageProcessing.main(originalImg, df_total)

    #saving as parquet file
    df_total.write.parquet("output/proto.parquet")