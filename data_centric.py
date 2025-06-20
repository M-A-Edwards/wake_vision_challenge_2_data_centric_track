import tensorflow as tf
import numpy as np
import os
from enhancing_script import *

model_name = 'wv_quality_mcunet-320kb-1mb_vww'

input_shape = (144,144,3)
color_mode = 'rgb'
num_classes = 2

batch_size = 128
epochs = 50
learning_rate = 0.001

path_to_training_set = './wake_vision/train_quality'
path_to_validation_set = './wake_vision/validation'
path_to_test_set = './wake_vision/test'

inputs = tf.keras.Input(shape=input_shape)
#
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
x = tf.keras.layers.Conv2D(16, (3,3), padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(8, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(240, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(160, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(160, (1,1), padding='valid')(x)
#
x = tf.keras.layers.AveragePooling2D(5)(x)
x = tf.keras.layers.Conv2D(2, (1,1), padding='valid')(x)
outputs = tf.keras.layers.Reshape((num_classes,))(x)

model = tf.keras.Model(inputs, outputs)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_training_set,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=batch_size,
    image_size=input_shape[0:2],
    shuffle=True,
    seed=11
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_validation_set,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=batch_size,
    image_size=input_shape[0:2],
    shuffle=True,
    seed=11
)

#data augmentation
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.2)])

train_ds = train_ds.map(lambda x, y: (get_advanced_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".keras",
    monitor='val_accuracy',
    mode='max', save_best_only=True)

model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[model_checkpoint_callback])
 
model = tf.keras.models.load_model(model_name + ".keras")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
        
#Test quantized model
interpreter = tf.lite.Interpreter(model_name + ".tflite")
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_test_set,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=1,
    image_size=input_shape[0:2],
    shuffle=True,
    seed=11
)

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {(correct/(correct+wrong)*100)}\n\n")
