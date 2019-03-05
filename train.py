import sys
from mjsynth_gen import MJSynthData
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend
from keras.optimizers import Adadelta
from crnn_model import get_model
from param import batch_size, validation_batch_size, epochs

#backend.set_learning_phase(0)
def main(args):
    model = get_model(training = True)
    model.summary()
    #get weights to train 
    try:
        model.load_weights(args[3])
        print("....Previous weight data...")
    except:
        print("...New weight data...")
        pass

    early = EarlyStopping(monitor='loss' , min_delta=0.001, patience=4, mode='auto', verbose=1)
    checkpoint = ModelCheckpoint(filepath = 'crnn_{epoch:02d}_{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='auto', period=1)

    ada = Adadelta()
    model.compile(loss={'ctc':lambda y_true, y_pred: y_pred}, optimizer=ada)

    training_gen = MJSynthData(args[0], args[1], batch_size)
    validation_gen = MJSynthData(args[0], args[2], validation_batch_size)

    hist = model.fit_generator(generator=training_gen,
                        steps_per_epoch=int(training_gen.dataset_size // batch_size),
                        epochs=epochs,
                        callbacks=[early, checkpoint],
                        validation_data=validation_gen,
                        validation_steps=int(validation_gen.dataset_size // validation_batch_size),
                        max_queue_size=32)

if __name__ == "__main__":
    if(len(sys.argv)!=5):
        print("Invalid command line inputs. Give Inputs - <name.py> <data_folder> <annotation_train> <annotation_test>")
        exit()
    else:
        main(sys.argv[1:])

