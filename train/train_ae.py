from common_import import *




###パラメータ###
model_structure = "AutoEncoder"
epochs = 10
batch_size = 32
validation_rate = 0.1
# test_rate = 0.1
cp_period = 2
data_dir = '../data'
# classes = ['yuto']
classes = ['Celeb-real-image']
# classes = ['Celeb-real-image-augmented']
image_size = (480, 640, 3)
es_flg = False



###ディレクトリ作成###
t = time.strftime("%Y%m%d-%H%M%S")
model_dir = f'../model/{model_structure}_{t}_epoch{epochs}'
os.makedirs(model_dir, exist_ok=True)
cp_dir = f'../model/{model_structure}_{t}_epoch{epochs}/checkpoint'
os.makedirs(cp_dir, exist_ok=True)



###モデルの生成###
model = globals()['load'+model_structure](input_shape=image_size)
model.summary()



###Generator作成###
file_num = sum(os.path.isfile(os.path.join(data_dir, name)) for name in os.listdir(data_dir))
datagen = ImageDataGenerator(
    rescale=1./255,
    data_format='channels_last',
    validation_split=validation_rate
)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size[:2],
    batch_size=batch_size,
    class_mode='input',
    shuffle=True,
    seed=1,
    color_mode="rgb",
    classes=classes,
    subset = "training" 
)
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size[:2],
    batch_size=batch_size,
    class_mode='input',
    shuffle=True,
    seed=1,
    color_mode="rgb",
    classes=classes,
    subset = "validation"
)


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
    filepath=cp_dir+"/cp_weight_{epoch:03d}-{val_loss:.2f}.hdf5",
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    period=cp_period
)
cb_list.append(cp_callback)



###学習###
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    # steps_per_epoch=2000,
    epochs=epochs,
    verbose=1,
    callbacks=cb_list
)



###テスト###
loss_and_metrics = model.evaluate_generator(
    validation_generator
)
print("Test loss:",loss_and_metrics[0])
print("Test accuracy:",loss_and_metrics[1])



###モデルの保存###
print("Save model...")
model.save(f'{model_dir}/model.h5')
model.save_weights(f'{model_dir}/weight.hdf5')



###グラフ化###
fig = plt.figure()
plt.plot(range(1, len(history.history['acc'])+1), history.history['acc'], "-o")
plt.plot(range(1, len(history.history['val_acc'])+1), history.history['val_acc'], "-o")
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(['acc','val_acc'], loc='best')
fig.savefig(model_dir+"/result.png")
# plt.show()



