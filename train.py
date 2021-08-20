import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from backbones.resnet import ResNet_
from backbones.mobilenet_v2 import MobileNetV2_
from utils.layers import ArcFaceLayer
from utils.datasets import MS1MRecognition
from utils.losses import ArcFaceLoss
from utils.metrics import ArcFaceAccuracy
import os
from config import *


os.environ['CUDA_VISIBLE_DEVICES'] = GPU
tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.threading.set_intra_op_parallelism_threads(6)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
      try:
          # Currently, memory growth needs to be the same across GPUs
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
          # Memory growth must be set before GPUs have been initialized
          print(e)


def build_model():
    if BACKBONE == 'resnet':
        _net = ResNet_(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            num_classes=NUM_FEATURES,
            kernel_regularizer=l2(L2),
            repetitions=NET_CONFIG
        )
    elif BACKBONE == 'mobilenet':
        _net = MobileNetV2_(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            num_classes=NUM_FEATURES,
            regularizer=l2(L2),
        )
    else:
        raise ValueError("Unknown Backbone: `{}`.".format(BACKBONE))

    x = _net.model.input

    if HEAD == 'arcface':
        head = ArcFaceLayer(n_classes=NUM_CLASSES, regularizer=l2(L2))
    elif HEAD == 'softmax':
        head = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', use_bias=False,
                                     kernel_regularizer=l2(L2))
    else:
        raise ValueError("Unknown Head: `{}`.".format(HEAD))
    output = head(_net.model.output)

    _net.model = tf.keras.models.Model(inputs=x, outputs=output, name=MODEL_NAME)

    return _net


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = build_model()

    if HEAD == 'arcface':
        net.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=MOMENTUM),
            loss=ArcFaceLoss(),
            metrics=[ArcFaceAccuracy()]
        )
    elif HEAD == 'softmax':
        net.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=MOMENTUM),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        raise ValueError("Unknown Head: `{}`.".format(HEAD))

# net.plot_model('resnet50_arcface.png')

start_epoch = 0
epochs = EPOCHS
batch_size = BATCH_SIZE

model_name = MODEL_NAME

if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)
log_dir = WORK_DIR + '/' + model_name
output_model_file = WORK_DIR + '/' + model_name + '.h5'
weight_file = WORK_DIR + '/' + model_name + '_weights.h5'
checkpoint_path = WORK_DIR + '/checkpoint-' + model_name + '.h5'

model = net.model

if os.path.exists(weight_file):
    model.load_weights(weight_file, by_name=True, skip_mismatch=True)
    print('Found latest weights file: {}, load weights.'.format(weight_file))
else:
    print('No weights file found. Skip loading weights.')

train_dataset = MS1MRecognition(
    tfrecord_path=TFRECORDS_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    argument=True,
)

tensorboard = TensorBoard(log_dir=log_dir, write_images=False)

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch'
)

def scheduler(epoch, lr):
    if epoch in SCHEDULE:
        return lr * 0.1
    else:
        return lr
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

train_samples = train_dataset.tfrecords_to_dataset()

model.fit(
    x=train_samples,
    epochs=epochs,
    callbacks=[tensorboard, checkpoint, lr_scheduler],
    initial_epoch=start_epoch,
    shuffle=False,
    verbose=1
)

model.save_weights(weight_file)
