####################################################################################################
# サンプル
####################################################################################################

# GPU無効化
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'



from common_import import *




# model_structure = "InceptionV3"
# model_structure = "Xception"
# model_structure = "EfficientNetV2"
model_structure = "OriginalNet"
gpu_count = 2
image_size = (256, 256, 3)

model = globals()['load'+model_structure](input_shape=image_size,gpu_count=gpu_count)
model.summary()






# inc = keras.applications.inception_v3.InceptionV3(include_top=False,input_shape=(256,256,3))
# inc.summary()

# xcep = keras.applications.xception.Xception(include_top=False,input_shape=(256,256,3))
# xcep.summary()

# effb0 = efficientnetv2.effnetv2_model.get_model('efficientnetv2-b0', include_top=False)
# effb0.summary()

# effb1 = efficientnetv2.effnetv2_model.get_model('efficientnetv2-b1', include_top=False)
# effb1.summary()

# effb2 = efficientnetv2.effnetv2_model.get_model('efficientnetv2-b2', include_top=False)
# effb2.summary()

# effb3 = efficientnetv2.effnetv2_model.get_model('efficientnetv2-b3', include_top=False)
# effb3.summary()

# effs = efficientnetv2.effnetv2_model.get_model('efficientnetv2-s', include_top=False)
# effs.summary()

# effm = efficientnetv2.effnetv2_model.get_model('efficientnetv2-m', include_top=False)
# effm.summary()

# effl = efficientnetv2.effnetv2_model.get_model('efficientnetv2-l', include_top=False)
# effl.summary()

# effxl = efficientnetv2.effnetv2_model.get_model('efficientnetv2-xl', include_top=False)
# effxl.summary()