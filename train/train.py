####################################################################################################
###実行コマンド###
# nohup python train.py > train_out.log &
###強制終了コマンド###
# jobs      (実行中のジョブ一覧)
# kill 1    (1のジョブを強制終了)
# fg 1      (1をフォアグランド実行に戻す)
# ctrl+c    (強制終了)
    # ctrl+z    (停止)
    # bg %1     (1をバックグラウンド実行で再開)
####################################################################################################


from common_import import *


###パラメータ###
# model_structure = "SampleCnn"
# model_structure = "Vgg16"
# model_structure = "Inception"
model_structure = "Xception"
# model_structure = "EfficientNetV2"
# model_structure = "OriginalNet"
epochs = 5
gpu_count = 2
batch_size = 32 * gpu_count
validation_rate = 0.1
test_rate = 0.1
# ↑動画ごとに分けているので最終的な画像でのデータ数はだいたい...
cp_period = 10
data_dir = '/data/toshikawa/datas'
# classes = ['yuto', 'b']
# classes = ['Celeb-real-image', 'Celeb-synthesis-image']
classes = ['Celeb-real-image-face', 'Celeb-synthesis-image-face']
# image_size = (480, 640, 3)
# image_size = (240, 320, 3)
image_size = (256, 256, 3)
# image_size = (32, 32, 3)
es_flg = False


###data augmentation###
rotation_range=15.0#0.0
width_shift_range=0.15#0.0
height_shift_range=0.15#0.0
brightness_range = None#None
shear_range=0.0#0.0
zoom_range=0.1#0.0
channel_shift_range = 0.0#0.0
horizontal_flip=True#False
vertical_flip=False#False


###モデルの生成###
model = globals()['load'+model_structure](input_shape=image_size,gpu_count=gpu_count)
model.summary()


###Generator作成###
class_file_num = {}
class_weights = {}
data = []
data_num = 0


for l,c in enumerate(classes):
    image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
    path_num = len(image_path_list)
    data_num += path_num
    regexp = r'^.+?id(?P<id>(\d+))(_id(?P<id2>\d+))?_(?P<key>\d+)_(?P<num>\d+).(?P<ext>.{2,4})$'
    past_path = image_path_list[0]
    movie_image_list = []
    for i in range(1,len(image_path_list)):
        past_ids = re.search(regexp,past_path).groupdict()
        now_ids = re.search(regexp,image_path_list[i]).groupdict()
        if (past_ids['id']==now_ids['id']) and (past_ids['id2']==None or past_ids['id2']==now_ids['id2']) and (past_ids['key']==now_ids['key']):
            movie_image_list.append([image_path_list[i],l])
        else:
            data.append(movie_image_list)
            movie_image_list = []
            movie_image_list.append([image_path_list[i],l])
        past_path = image_path_list[i]
    # print(len(data))
    # 不均衡データ調整
    class_file_num[c] = path_num
    if l==0:
        n = class_file_num[c]
    class_weights[l] = 1 / (class_file_num[c]/n)
print("\tMOVIE NUM: " + str(len(data)))
random.shuffle(data)
train_rate = 1 - validation_rate - test_rate
s1 = (int)(len(data)*train_rate)
s2 = (int)(len(data)*(train_rate+validation_rate))
train_data = data[ : s1]
validation_data = data[s1 : s2]
test_data = data[s2 : ]
del data
train_data = list(chain.from_iterable(train_data))
validation_data = list(chain.from_iterable(validation_data))
test_data = list(chain.from_iterable(test_data))
train_data_num = len(train_data)
validation_data_num = len(validation_data)
test_data_num = len(test_data)
print("\tALL IMAGE DATA NUM: " + str(data_num))
print("\tTRAIN IMAGE DATA NUM: " + str(train_data_num))
print("\tVALIDATION IMAGE DATA NUM: " + str(validation_data_num))
print("\tTEST IMAGE DATA NUM: " + str(test_data_num))
def makeGenerator(data):
    return ImageIterator(
        data,
        batch_size=batch_size,
        target_size=image_size[:2],
        color_mode='rgb',
        seed=1,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=1./255,
        data_format='channels_last',
        subset='train')
train_generator = makeGenerator(train_data)
validation_generator = makeGenerator(validation_data)
test_generator = makeGenerator(test_data)
del train_data
del validation_data
del test_data


###ディレクトリ作成###
t = time.strftime("%Y%m%d-%H%M%S")
model_dir = f'../model/{model_structure}_{t}_epoch{epochs}'
os.makedirs(model_dir, exist_ok=True)
cp_dir = f'../model/{model_structure}_{t}_epoch{epochs}/checkpoint'
os.makedirs(cp_dir, exist_ok=True)


###callback作成###
cb_list = []
if es_flg:
    ##↓監視する値の変化が停止した時に訓練を終了##
    es_callback = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=1,
        mode='auto'
    )
    cb_list.append(es_callback)
cp_callback = callbacks.ModelCheckpoint(
    filepath=cp_dir+"/cp_weight_{epoch:03d}-{accuracy:.2f}.hdf5",
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    period=cp_period
)
cb_list.append(cp_callback)


###サンプルデータセット使用###
# train_generator,test_generator = getSampleData()
# validation_generator = None


###条件出力###
print("\tGPU COUNT: " + str(gpu_count))
print("\tBATCH SIZE: " + str(batch_size))
print("\tVALIDATION RATE: " + str(validation_rate))
print("\tTEST RATE: " + str(test_rate))
print("\tTRAIN DATA NUM: " + str(train_data_num))
print("\tVALIDATION DATA NUM: " + str(validation_data_num))
print("\tTEST DATA NUM: " + str(test_data_num))
print("\tCHECKPOINT PERIOD: " + str(cp_period))
print("\tDATA DIRECTORY: " + str(data_dir))
print("\tCLASSES: " + str(classes))
print("\tCLASSES NUM: " + str(class_file_num))
print("\tIMAGE SIZE: " + str(image_size))
print("\tEARLY STOPPING: " + str(es_flg))
print("\tROTATION RANGE: " + str(rotation_range))
print("\tWIDTH SHIFT RANGE: " + str(width_shift_range))
print("\tHEIGHT SHIFT RANGE: " + str(height_shift_range))
print("\tBRIGHTNESS RANGE: " + str(brightness_range))
print("\tSHEAR RANGE: " + str(shear_range))
print("\tZOOM RANGE: " + str(zoom_range))
print("\tCHANNEL SHIFT RANGE: " + str(channel_shift_range))
print("\tHORIZONTAL FLIP: " + str(horizontal_flip))
print("\tVERTICAL FLIP: " + str(vertical_flip))
print("")


###テスト,モデル保存###
def test_and_save(a=None,b=None):
    global model
    global model_dir
    global history
    global test_generator

    ###テスト###
    print("テスト中")
    loss_and_metrics = model.evaluate_generator(
        test_generator
    )
    print("Test loss:",loss_and_metrics[0])
    print("Test accuracy:",loss_and_metrics[1])
    print("Test AUC:",loss_and_metrics[2])
    print("Test Precision:",loss_and_metrics[3])
    print("Test Recall:",loss_and_metrics[4])
    print("Test TP:",loss_and_metrics[5])
    print("Test TN:",loss_and_metrics[6])
    print("Test FP:",loss_and_metrics[7])
    print("Test FN:",loss_and_metrics[8])


    ###モデルの保存###
    print("Save model...")
    try:
        model.save(f'{model_dir}/model.h5')
    except NotImplementedError:
        print('Error')
    model.save_weights(f'{model_dir}/weight.hdf5')


    ###グラフ化###
    try:
        fig = plt.figure()
        plt.plot(range(1, len(history.history['accuracy'])+1), history.history['accuracy'], "-o")
        plt.plot(range(1, len(history.history['val_accuracy'])+1), history.history['val_accuracy'], "-o")
        plt.title('Model accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend(['accuracy','val_accuracy'], loc='best')
        fig.savefig(model_dir+"/result.png")
        # plt.show()
    except NameError:
        print("The graph could not be saved because the process was interrupted.")

    print("Finish")


###学習###
print("学習中...")
signal.signal(signal.SIGINT, test_and_save)
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    # steps_per_epoch=10,
    epochs=epochs,
    class_weight=class_weights,
    verbose=1,
    workers=8,
    use_multiprocessing=False,
    callbacks=cb_list
)

test_and_save()
