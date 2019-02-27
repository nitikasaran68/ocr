from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend
from keras.optimizers import Adadelta
from Model import get_Model
import mjsynth 

#backend.set_learning_phase(0)

model = get_Model(training = True)

#get weights to train 
try:
    model.load_wieghts('LSTM+BN4--26--0.011.hdf5')
    print("....Previous weight data...")
except:
    print("...New weight data...")
    pass

early = EarlyStopping(monitor = 'loss' , min_delta = 0.001, patience = 4, mode = 'min', verbose =1 )
checkpoint = ModelCheckpoint(filepath = 'LSTM+BN%--{epoch:02d}--{val_loss:.3f}.hdf5', monitor = 'loss' , verbose = 1, mode = 'min', period = 1)
ada = Adadelta()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

dataset_train = mjsynth('''Enter the train parameters''')
dataset_valid = mjsynth('''Enter the valid parameters''')


model.fit_generator(generator= dataset_train.next_batch(),
                    steps_per_epoch=int(dataset_train.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint],
                    validation_data=dataset_valid.next_batch(),
                    validation_steps=int(dataset_valid.n / val_batch_size))


