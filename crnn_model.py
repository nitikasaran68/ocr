from param import img_input_shape, classes
from param import max_string_len

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import BatchNormalization, Reshape
from keras.models import Model, Sequential
from keras.layers import LSTM, Lambda
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# model file

def ctc_lambda_func(args):
    y_true, y_pred, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def create_model():
    img_input = Input(name='img_input', shape=img_input_shape, dtype='float32')
    
    # CNN
    
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape = img_input_shape, name='conv_1')(img_input)
    max_pool_1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(conv_1)
    
    conv_2 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv_2')(max_pool_1)
    max_pool_2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(conv_2)

    conv_3 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_3')(max_pool_2)
    conv_4 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_4')(conv_3)
    max_pool_3 = MaxPooling2D((1, 2), strides=(1, 2), name='pool_3')(conv_4)

    conv_5 = Conv2D(512, (3,3), activation='relu', padding='same', name='conv_5')(max_pool_3)
    normalize_1 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3,3), activation='relu', padding='same', name='conv_6')(normalize_1)
    normalize_2 = BatchNormalization()(conv_6)

    max_pool_4 = MaxPooling2D((1, 2), strides=(1, 2), name='pool_4')(normalize_2)
    conv_7 = Conv2D(512, (3,3), activation='relu', padding='same', name='conv_7')(max_pool_4)

    # CNN to RNN
    reshape = Reshape((25, 1024), input_shape=(25, 2, 512))(conv_7)

    # RNN

    bi_lstm_1 = Bidirectional(LSTM(256,input_shape = (25, 1024), return_sequences = True, name='b_lstm1'))(reshape)
    bi_lstm_2 = Bidirectional(LSTM(256,input_shape = (25, 1024), return_sequences = True, name='b_lstm2'))(bi_lstm_1)

    # RNN to softmax Activations
    dense = TimeDistributed(Dense(classes,activation = 'sigmoid', name='dense'))(bi_lstm_2)
    y_pred = Activation('softmax', name='final_softmax')(dense)
    
    #  Model(inputs=img_input, outputs=y_pred).summary()

    # Input specifications for CTC Function
    y_true = Input(name='ground_truth', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_true, y_pred, input_length, label_length])

    # Loss function algo
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    crnn_model_train = Model(inputs=[img_input, y_true, input_length, label_length], outputs=loss_out)
    crnn_model_test = Model(inputs = [img_input], outputs = y_pred)
    return crnn_model_train,crnn_model_test

# Main code
def get_model(training):
    model_train, model_test = create_model()
    if training:
        return model_train
    else:
        return model_test
#model.summary()

#im = np.transpose(img,(1,0,2))
#https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/