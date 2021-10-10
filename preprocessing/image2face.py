from mtcnn.mtcnn import MTCNN
import cv2
import os
import glob
from pprint import pprint



def image2face(src_path, dst_directory, square_size=None, padding_rate=0, confidence_threshold=0.9, biggest_confidence=True, ext='jpg'):

    # 読み込み
    src_data = cv2.imread(src_path)
    h,w,c = src_data.shape
    pixels = cv2.cvtColor(src_data, cv2.COLOR_BGR2RGB)

    # 顔抽出
    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    # パス作成
    os.makedirs(dst_directory, exist_ok=True)
    basename = os.path.splitext(os.path.basename(src_path))[0]
    dst_path = os.path.join(dst_directory, basename+"."+ext)

    max_confidence = -1
    for i in range(len(faces)):
        # 閾値よりも低い信頼度であればスキップ
        confidence = faces[i]['confidence']
        if confidence < confidence_threshold:
            continue

        x1, y1, width, height = faces[i]['box']

        # 正方形切り取り(正方形で切り取れる部分しか残せない)
        if square_size:
            if width > height:
                length = width
                diff = width-height
                y1 -= int(diff/2)
                height = width
                # はみ出していたら
                if y1 < 0:
                    y1 = 0
                if w < width:
                    height = h
                    width = h
            else:
                length = height
                diff = height-width
                x1 -= int(diff/2)
                width = height
                # はみ出していたら
                if x1 < 0:
                    x1 = 0
                if w < width:
                    width = w
                    height = w

        # パディング(上下左右それぞれに、padding_rate分の、上下左右で平等なパディングを施す)
        if padding_rate:
            min_p = int(length*padding_rate)
            if x1-min_p < 0 and x1 < min_p:
                min_p = x1
            if x1+width+min_p > w and w-(x1+width) < min_p:
                min_p = w-(x1+width)
            if y1-min_p < 0 and y1 < min_p:
                min_p = y1
            if y1+height+min_p > h and h-(y1+height) < min_p:
                min_p = h-(y1+height)
            x1 -= min_p
            y1 -= min_p
            width += min_p*2
            height += min_p*2

        x2, y2 = x1+width, y1+height
        croped_data = src_data[y1:y2, x1:x2]

        # 正方形ならリサイズ
        if square_size:
            croped_data = cv2.resize(croped_data, dsize=(square_size,square_size))

        if biggest_confidence:
            max_confidence = confidence
            md = croped_data
        else:
            cv2.imwrite(dst_path, croped_data)

    # 最大信頼度の時のみ保存
    if biggest_confidence and max_confidence > 0:
        cv2.imwrite(dst_path, md)





if __name__=='__main__':
    ###パラメータ###
    # src_directory = "../data/Celeb-real-image"
    # dst_directory = "../data/Celeb-real-image-face"
    src_directory = "../data/Celeb-synthesis-image-learning"
    dst_directory = "../data/Celeb-synthesis-image-learning-face"
    square_size = 512
    padding_rate = 0.1
    confidence_threshold = 0.9
    
    images = glob.glob(src_directory+"/*")
    for image in images:
        print(image)
        image2face(
            image,
            dst_directory,
            square_size=square_size,
            padding_rate=padding_rate,
            confidence_threshold=confidence_threshold
        )
