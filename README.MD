# Face Recognition Using ArcFace and TensorFlow2

This repository contains codes for face recognition. [ArcFace Loss](https://arxiv.org/abs/1801.07698) is implemented
in Keras and is easy to use.

## Requirements

- numpy~=1.19.5
- matplotlib~=3.4.1
- opencv-python~=4.5.2.54
- tensorflow~=2.4.1
- bcolz~=1.2.1 (For evaluation only)
- tqdm~=4.62.0 (For evaluation only)
- sklearn~=0.0 (For evaluation only)
- scikit-learn~=0.24.2 (For evaluation only)

## Usage

### Using ArcFace

**ArcFaceLayer**, **ArcFaceLoss** and **ArcFaceAccuracy** are implemented in Keras and TensorFlow 2.X. It is easy to
use by Model.complie() and Model.fit() API.
```python
in_layer = Input(shape=(112, 112, 3), name='input_image')

basemodel = tf.keras.applications.resnet.ResNet50(
    include_top=False, 
    input_tensor=in_layer,
    weights='imagenet',
)
x = GlobalAveragePooling2D()(basemodel.output)
x = Dropout(dropout)(x)
x = Dense(256, name='embd_features')(x)

output = ArcFaceLayer(n_classes=n_classes, regularizer=l2(0.0005))(x)
model = tf.keras.models.Model(inputs=in_layer, outputs=output, name='resnet50_arcface')

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.8),
    loss=ArcFaceLoss(),
    metrics=[ArcFaceAccuracy()]
)

model.fit(
    x=train_images,
    y=train_labels,
    epochs=epochs,
    batch_size=batch_size,
)
```

### Preparing Data

MS-Celeb-1M dataset is used for training. Download aligned
[MS-Celeb-1M](https://drive.google.com/file/d/1X202mvYe5tiXFhOx82z4rPiPogXD435i/view?usp=sharing) dataset. Config the
dataset path in config.py and convert images to tfrecords file.
```bash
cd utils
python datasets.py
```
- Note: The convertion can be really slow and cost a lot of time. But the converted tfrecords file can be read and loaded really
fast during the training process.
- Note: The last 100 IDs (i.e. ID 85642~85741) are reserved for testing. So the number of classes is 85642 rather than 85742. 

[LFW](https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view?usp=sharing), 
[Aged30](https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp/view?usp=sharing) and 
[CFP-FP](https://drive.google.com/file/d/1-sDn79lTegXRNhFuRnIRsgdU88cBfW6V/view?usp=sharing) datasets are used for 
evaluation. Downloads these datasets and
then extract them to `/path/to/your/test/dataset/dir`. These testing data are already binary files, so it's not 
necessary to do any preprocessing. The directory structure should be like bellow.
```
/your/path/to/test_dataset/
    -> lfw_align_112/lfw
        -> data/
        -> meta/
        -> ...
    -> agedb_align_112/agedb_30
        -> ...
    -> cfp_align_112/cfp_fp
        -> ...
```
- Note: Evaluation codes and datasets credit to [this repository](https://github.com/peteryuX/arcface-tf2).

### Training

Modify configurations in config.py and run train.py:
```bash
python train.py
```
And a progress bar can been seen in terminal:
```text
Epoch 1/25 
22719/22719 [==============================] - 5515s 242ms/step - loss: 19.1998 - accuracy: 0.3787                                                                                                                        
Epoch 00001: saving model to /home/USER/checkpoints/arcface/checkpoint-resnet50_im112_ms1m_arcface.h5              
Epoch 2/25                                                                                                            
22719/22719 [==============================] - 5497s 242ms/step - loss: 11.0482 - accuracy: 0.8703                                                                                                                   
Epoch 00002: saving model to /home/USER/checkpoints/arcface/checkpoint-resnet50_im112_ms1m_arcface.h5 
```

### Testing
#### Test using reserved IDs in MS-Celeb-1M dataset

We choose 20 IDs in the reserved MS-Celeb-1M dataset for testing, and choose 20 sample images for each class. So, there
are 400 sample images for testing. Run embd_images.py to do the embedding:
```bash
cd tools
python embd_images.py
```
And some info can be seen in terminal:
```text
ID:85741 Embedding done.
ID:85740 Embedding done.
ID:85739 Embedding done.
ID:85738 Embedding done.
ID:85737 Embedding done.
ID:85736 Embedding done.
ID:85735 Embedding done.
ID:85734 Embedding done.
ID:85733 Embedding done.
ID:85732 Embedding done.
ID:85731 Embedding done.
ID:85730 Embedding done.
ID:85729 Embedding done.
ID:85728 Embedding done.
ID:85727 Embedding done.
ID:85726 Embedding done.
ID:85725 Embedding done.
ID:85724 Embedding done.
Skipping ID:85723.
ID:85722 Embedding done.
ID:85721 Embedding done.
```
Then we get the matrix showing the similarity of the 400 embedded test images: 

![image](https://github.com/rao1140427950/face_recognition_tf2/blob/master/results/similarity.png?raw=true)

- Note: The number of test samples and test IDs can be modified in embd_images.py

#### Test using face recognition benchmark

Run evaluate.py to do the evalutations:
```bash
python evaluate.py
```
Evaluation results (%) of different backbones and configurations are shown below:

| Backbone | Head | CCrop | embd_shape | LFW | AgeDB-30 | CFP-FP | Checkpoint |
|----------|------|-------|------------|-----|----------|--------|------------|
| ResNet50 | ArcFace | False | 256 | 99.35 | 95.03  |  90.36   | [GoogleDrive](https://drive.google.com/file/d/1VCknt0BG7NXb4Jry--NMW-q2bWra9O1j/view?usp=sharing) |
| ResNet50 | ArcFace | True  | 256 | 98.97 | 93.03  |  90.26   | [GoogleDrive](https://drive.google.com/file/d/1VCknt0BG7NXb4Jry--NMW-q2bWra9O1j/view?usp=sharing) |
| MobileNetV2 | ArcFace | False  | 256 | 98.73 | 89.22  |  90.96   | [GoogleDrive](https://drive.google.com/file/d/1plu40JoiaXK1QTL1TecumuTbQT-YCmyA/view?usp=sharing) |
| MobileNetV2 | ArcFace | True  | 256 | 98.95 | 90.98  |  91.63   | [GoogleDrive](https://drive.google.com/file/d/1plu40JoiaXK1QTL1TecumuTbQT-YCmyA/view?usp=sharing) |

- Note: Trained model weigths can be downloaded from the GoogleDrive link.

## References:
- https://github.com/peteryuX/arcface-tf2
- https://github.com/4uiiurz1/keras-arcface


