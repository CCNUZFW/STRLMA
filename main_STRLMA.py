import glob
#import numpy as np
import random
#import librosa
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelBinarizer
from keras.models import *
import keras
from keras.layers import LSTM, Dense, Dropout, Flatten,Conv2D
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import pandas as pd

from keras.layers import Conv2D, BatchNormalization, Activation, Bidirectional,GlobalAveragePooling2D
from keras.layers import AveragePooling2D, Input, concatenate, Lambda
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf


from keras.layers import add,Input,Conv1D,Activation,Flatten,Dense



# 分类问题的类数，fc层的输出单元个数
NUM_CLASSES = 45
# 更新中心的学习率
ALPHA = 0.2
# center-loss的系数
LAMBDA = 0.0005555


def ResBlock(x,filters,kernel_size,dilation_rate):
    h=Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    s=Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(h)
    r=Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(s)
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    d=concatenate()
    o=add([r,s,h,shortcut])
    o=Activation('relu')(o)  #激活函数
    return o


def softmax_loss(labels, features):
    """
    计算softmax-loss
    :param labels: 等同于y_true，使用了one_hot编码，shape应为(batch_size, NUM_CLASSES)
    :param features: 等同于y_pred，模型的最后一个FC层(不是softmax层)的输出，shape应为(batch_size, NUM_CLASSES)
    :return: 多云分类的softmax-loss损失，shape为(batch_size, )
    """
    return K.categorical_crossentropy(labels, K.softmax(features))


def categorical_accuracy(y_true, y_pred):
    """
    重写categorical_accuracy函数，以适应去掉softmax层的模型
    :param y_true: 等同于labels，
    :param y_pred: 等同于features。
    :return: 准确率
    """
    # 计算y_pred的softmax值
    sm_y_pred = K.softmax(y_pred)
    # 返回准确率
    return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(sm_y_pred, axis=-1)), K.floatx())


def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):

    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    return x

def pad_depth(inputs, desired_channels):
    from keras import backend as K
    y = K.zeros_like(inputs, name='pad_depth1')
    return y

def My_freq_split1(x):
    from keras import backend as K
    return x[:,0:32,:,:]

def My_freq_split2(x):
    from keras import backend as K
    return x[:,32:128,:,:]

def mul(x):
    return x[0] * x[1]


num_filters =24
My_wd=1e-3

#strides=[1]
learning_rate = 0.001
#batch_size = 64
#n_epochs = 50
dropout = 0.5
n_classes=21
n_features = 64
max_length = 39
steps_per_epoch = 50
input_shape = (n_features,max_length)

input_MFCC = keras.layers.Input(shape=[650, 39])
input_GSV = keras.layers.Input(shape=[39, 64])
def scheduler(epoch):
    # 每隔30个epoch，学习率减小为原来的1/10
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(m.optimizer.lr)
        K.set_value(m.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(m.optimizer.lr)

def build_model():


    hiddenatt1 = keras.layers.Reshape((39,64,1))(input_GSV)

    hiddenatt2 = keras.layers.Conv2D(16, (3, 3), strides=(2, 2),activation='relu', padding='same', data_format='channels_last', name='att5')(hiddenatt1)
    hiddenatt3 = keras.layers.Conv2D(32, (3, 3), strides=(2, 2),activation='relu', padding='same', data_format='channels_last', name='att6')(hiddenatt2)
    hiddenatt4 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(hiddenatt3)
    hiddenatt5 = keras.layers.Reshape([896])(hiddenatt4)
    hiddenatt6 = keras.layers.Dense(2496, activation='sigmoid')(hiddenatt5)
    hiddenatt7 = keras.layers.Reshape((39, 64))(hiddenatt6)
    hiddenatt8 = keras.layers.multiply([hiddenatt7, input_GSV])
    hiddenatt9 = keras.layers.Reshape([2496])(hiddenatt8)


    #hiddenatt9 = keras.layers.Dense(2496, activation='relu', name='layers_fully1')(hiddenatt1)
    hidden2 = keras.layers.Dense(1024, activation='relu', name='layers_fully2')(hiddenatt9)
    hidden3 = keras.layers.Dense(1024, activation='relu', name='layers_fully3')(hidden2)
    hidden4 = keras.layers.Dropout(dropout)(hidden3)

    hiddenatt10 = keras.layers.Reshape((650, 39, 1))(input_MFCC)
    hiddenatt11 = keras.layers.Conv2D(16, (5, 5), strides=(3, 3),activation='relu', padding='same', data_format='channels_last', name='att1')(hiddenatt10)
    hiddenatt12 = keras.layers.Conv2D(32, (3, 3), strides=(3, 3),activation='relu', padding='same', data_format='channels_last', name='att2')(hiddenatt11)
    hiddenatty = keras.layers.MaxPooling2D(pool_size=(5, 1), strides=(5, 1), padding='valid')(hiddenatt12)
    hiddenatt13 = keras.layers.MaxPooling2D(pool_size=(7, 1), strides=(7, 1), padding='valid')(hiddenatty)
    hiddenatt14 = keras.layers.Reshape([320])(hiddenatt13)
    hiddenatt15 = keras.layers.Dense(650, activation='sigmoid')(hiddenatt14)
    hiddenatt16 = keras.layers.Reshape((650, 1))(hiddenatt15)
    hiddenatt17 = keras.layers.multiply([hiddenatt16, input_MFCC])
    hiddenatt18 = keras.layers.Reshape((650, 39))(hiddenatt17)

    hidden5 = ResBlock(hiddenatt18, filters=6, kernel_size=5, dilation_rate=1)
    hidden6 = ResBlock(hidden5, filters=16, kernel_size=5, dilation_rate=2)
    hidden7 = ResBlock(hidden6, filters=40, kernel_size=3, dilation_rate=4)




    #hidden5 = keras.layers.Conv2D(6, (5, 5), activation='relu', padding='same', data_format='channels_last',name='layer1_con1')(hiddenatt18)
    #hidden6 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(hidden5)
    #hidden7 = keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same', data_format='channels_last',name='layer1_con2')(hidden6)
    #hidden8 = keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid')(hidden7)
    #hidden200 = keras.layers.Conv2D(40, (5, 5), activation='relu', padding='same', data_format='channels_last',name='layer1_con3')(hidden8)
    #hidden201 = keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid')(hidden200)

    #hidden13 = keras.layers.Flatten()(hidden201)
    #hidden12 = Bidirectional(LSTM(39,name='layer1_lstm1', return_sequences=True, input_shape=input_shape, dropout=dropout))(hiddenatt8)
    #hidden13 = Bidirectional(LSTM(78, name='layer1_lstm2', return_sequences=True, input_shape=input_shape, dropout=dropout))(hidden12)
    hidden14 = keras.layers.Flatten()(hidden7)
    hidden15 = keras.layers.Dense(1024, activation='relu', name='layers_fully4')(hidden14)
    hidden16 = keras.layers.Dropout(dropout)(hidden15)
    # Three loss functions
    hidden9 = keras.layers.Dense(45, activation='softmax', name='layers_softmax1')(hidden4)

    hidden50 = keras.layers.Dense(45, activation='softmax', name='layers_softmax2')(hidden16)



    hidden17 = keras.layers.concatenate([hidden4, hidden16])

    hidden18 = keras.layers.Reshape([2048])(hidden17)
    
    hidden100 = keras.layers.Reshape((2, 1024))(hidden18)

    hidden19 = keras.layers.Conv2D(16, (3, 3), strides=(1, 2),activation='relu', padding='same', data_format='channels_last', name='layer2_con8')(hidden18)
    hidden20 = keras.layers.Conv2D(32, (5, 5), strides=(1, 3),activation='relu', padding='same', data_format='channels_last', name='layer2_con9')(hidden19)
    hiddenx = keras.layers.MaxPooling2D(pool_size=(1, 5), strides=(1, 5), padding='valid')(hidden20)
    hidden21 = keras.layers.MaxPooling2D(pool_size=(1, 7), strides=(1, 7), padding='valid')(hiddenx)
    hidden22 = keras.layers.Reshape([256])(hidden21)
    hidden23 = keras.layers.Dense(2, activation='sigmoid')(hidden22)

    hidden24 = keras.layers.Reshape((2, 1))(hidden23)
    hidden25 = keras.layers.multiply([hidden24, hidden100])
    
    hidden26 = keras.layers.Flatten()(hidden25)
    

    hidden29 = keras.layers.Dense(1024, activation='relu', name='layers11')(hidden26)
    hidden30 = keras.layers.Dense(400, activation='relu', name='layers12')(hidden29)
    hidden31 = keras.layers.Dense(128, activation='relu', name='layers13')(hidden30)
    hidden32 = Dense(45, activation='softmax', name='fc3')(hidden31)


    #model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    model = keras.models.Model(inputs=[input_MFCC,input_GSV], outputs=[hidden9,hidden50,hidden32])
    return model

if __name__ == '__main__':
    m = build_model()
    adam=optimizers.adam(lr=0.001)
    m.compile(optimizer=adam,loss={
        'layers_softmax1': 'categorical_crossentropy',
        'layers_softmax2': 'categorical_crossentropy',
        'fc3': 'categorical_crossentropy'},
              loss_weights={
                  'layers_softmax1': 1.,
                  'layers_softmax2': 2.,
                  'fc3': 1.
              },
              metrics=['accuracy'])
    m.summary()

    #数据准备
    #x_train_mfccs=np.zeros((2000,2496),'float')
    #x_test_mfccs=np.zeros((400,2496),'float')
    y_train = np.zeros((45*514, 45),'float')
    y_test = np.zeros((45*128, 45),'float')

    for i in range(45):
        y_train[i * 514:(i + 1) * 514, i] = 1
        y_test[i * 128:(i + 1) * 128, i] = 1

    path = 'MFCC_train.csv'
    inputs_mfcc_train = pd.read_csv(path, header=None)
    x_train_mfccs = inputs_mfcc_train.values

    path = 'MFCC_test.csv'
    inputs_mfcc_test = pd.read_csv(path, header=None)
    x_test_mfccs = inputs_mfcc_test.values

    path = 'GSV_train64.csv'
    inputs_gsv_train = pd.read_csv(path, header=None)
    x_train_gsv = inputs_gsv_train.values

    path = 'GSV_test64.csv'
    inputs_gsv_test = pd.read_csv(path, header=None)
    x_test_gsv = inputs_gsv_test.values

    x_train_mfcc1 = x_train_mfccs.reshape(x_train_mfccs.shape[0], 39, 650)
    x_train_mfcc1=x_train_mfcc1.transpose(0,2,1)
    x_test_mfcc1 = x_test_mfccs.reshape(x_test_mfccs.shape[0], 39, 650)
    x_test_mfcc1 = x_test_mfcc1.transpose(0, 2, 1)

    x_train_gsv = x_train_gsv.reshape(x_train_gsv.shape[0], 39, 64)
    #x_train_gsv = x_train_gsv.transpose(0, 2, 1, 3)
    x_test_gsv = x_test_gsv.reshape(x_test_gsv.shape[0], 39, 64)
    #x_test_gsv = x_test_gsv.transpose(0, 2, 1, 3)



    reduce_lr = LearningRateScheduler(scheduler)
    m.fit([x_train_mfcc1,x_train_gsv], [y_train,y_train,y_train], epochs=200, batch_size=128, callbacks=[reduce_lr])
    #m.save('m_37.h5')
    accuracy = m.evaluate([x_test_mfcc1,x_test_gsv], [y_test,y_test,y_test])
    #print('loss:', loss)
    print('accuracy:', accuracy)