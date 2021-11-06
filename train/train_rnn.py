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
model_structure = "SampleRnn"
epochs = 50
gpu_count = 2
batch_size = 32 * gpu_count
validation_rate = 0.1
test_rate = 0.1
cp_period = 10
data_dir = '/data/toshikawa/datas'
# classes = ['yuto', 'b']
# classes = ['Celeb-real-image', 'Celeb-synthesis-image-learning-2']
classes = ['Celeb-real-image', 'Celeb-synthesis-image']
nt = 10
image_size = (480, 640, 3)
# image_size = (256, 256, 3)
# image_size = (32, 32, 3)
es_flg = False


###data augmentation###
# rotation_range=15.0#0.0
# width_shift_range=0.15#0.0
# height_shift_range=0.15#0.0
# shear_range=0.0#0.0
# zoom_range=0.1#0.0
# horizontal_flip=True#False
# vertical_flip=False#False


###不均衡データ調整###
class_file_num = {}
class_weights = {}
for i,c in enumerate(classes):
    class_file_num[c] = sum(os.path.isfile(os.path.join(data_dir+"/"+c,name)) for name in os.listdir(data_dir+"/"+c))
    if i==0:
        n = class_file_num[c]
    class_weights[i] = 1 / (class_file_num[c]/n)
print(class_file_num)


###ディレクトリ作成###
t = time.strftime("%Y%m%d-%H%M%S")
model_dir = f'../model/{model_structure}_{t}_epoch{epochs}'
os.makedirs(model_dir, exist_ok=True)
cp_dir = f'../model/{model_structure}_{t}_epoch{epochs}/checkpoint'
os.makedirs(cp_dir, exist_ok=True)


###モデルの生成###
model = globals()['load'+model_structure](input_shape=image_size,gpu_count=gpu_count)
model.summary()


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
print("\tCHECKPOINT PERIOD: " + str(cp_period))
print("\tDATA DIRECTORY: " + str(data_dir))
print("\tCLASSES: " + str(classes))
print("\tCLASSES NUM: " + str(class_file_num))
print("\tIMAGE SIZE: " + str(image_size))
print("\tEARLY STOPPING: " + str(es_flg))
# print("\tROTATION RANGE: " + str(rotation_range))
# print("\tWIDTH SHIFT RANGE: " + str(width_shift_range))
# print("\tHEIGHT SHIFT RANGE: " + str(height_shift_range))
# print("\tSHEAR RANGE: " + str(shear_range))
# print("\tZOOM RANGE: " + str(zoom_range))
# print("\tHORIZONTAL FLIP: " + str(horizontal_flip))
# print("\tVERTICAL FLIP: " + str(vertical_flip))
print("")


###pathリスト作成###
data = []
label = []
for l,c in enumerate(classes):
    image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
    path_num = len(image_path_list)
    regexp = r'^.+id(?P<id>(\d+))_(?P<id2>\d+)_?(?P<num>\d+).(?P<ext>.{2,4})$'
    past_path = image_path_list[0]
    i = 0
    while path_num > i+nt:
        sequence_path_list = []
        for j in range(nt):
            past_ids = re.search(regexp,past_path).groupdict()
            now_ids = re.search(regexp,image_path_list[i+j]).groupdict()
            if (past_ids['id']==now_ids['id']) and (past_ids['id2']==now_ids['id2']):
                sequence_path_list.append(image_path_list[i+j])
            else:
                i += j
                past_path = image_path_list[i]
                break
            past_path = image_path_list[i+j]
        else:
            data.append(sequence_path_list)
            label.append([c])
            i += 1


    print(len(data))
    print(len(label))
    exit()


###data分割###
data_num = len(label)
train_num = (int)(data_num * (1-(validation_rate+test_rate)))
validation_num = (int)(data_num * validation_rate)
train_data = data[:train_num]
train_label = label[:train_num]
validation_data = data[train_num:train_num+validation_num]
validation_label = label[train_num:train_num+validation_num]
test_data = data[train_num+validation_num:]
test_label = label[train_num+validation_num:]


###学習###
train_data_num = len(train_label)
for epc in range(epochs):
    rnd_idx = random.shuffle(range(len(label)))
    _train_data = train_data[rnd_idx]
    _train_label = train_label[rnd_idx]
    i = 0
    while i < epochs:
        x = []
        for path in _train_data[i:i+batch_size]:
            img = cv2.imread(path)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            x.append(img)
        y = _train_label[i:i+batch_size]
        x = np.array(x,dtype=np.float32)
        y = np.array(y,dtype=np.float32)
        history = model.train_on_batch(x,y,class_weight=class_weights)
        i += batch_size


exit()

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
model.save(f'{model_dir}/model.h5')
model.save_weights(f'{model_dir}/weight.hdf5')


###グラフ化###
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

print("Finish")