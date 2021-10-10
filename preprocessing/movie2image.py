import cv2
import os
import glob


def save_all_frames(video_path, dir_path, basename, step=1, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if (n%step)==1:
                cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return



if __name__=='__main__':
    video_directory = "../../Celeb-DF-v2/Celeb-synthesis/*"
    # image_directory = "../data/Celeb-synthesis-image-learning"
    image_directory = "../data/Celeb-synthesis-image-sub"
    step = 10    #何フレームごとに画像を保存するか

    videos = glob.glob(video_directory)
    for video in videos:
        basename = os.path.splitext(os.path.basename(video))[0]
        print(basename)
        save_all_frames(video, image_directory, basename, step)