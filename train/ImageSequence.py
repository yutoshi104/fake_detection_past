from tensorflow.keras.utils import Sequence


class ImageSequence(Sequence):
    def __init__(self, pairs, num_classes, root='.', batch_size=1):
        self.x = [str(root / Path(x)) for x in pairs[0]]
        self.y = np_utils.to_categorical(pairs[1], num_classes)
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # バッチサイズ分取り出す
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # 画像を1枚づつ読み込んで、前処理をする
        batch_x = np.array([self.preprocess(imread(file_name)) for file_name in batch_x])
        return batch_x, np.array(batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def preprocess(self, image):
        # いろいろ前処理
        return image
