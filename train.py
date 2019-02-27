from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend
from keras.optimizers import Adadelta
from crnn_model import get_Model
import mjsynth 
from param import batch_size, validation_batch_size
import sys

#backend.set_learning_phase(0)
def main(args):
    model = get_Model(training = True)

    #get weights to train 
    try:
        model.load_wieghts('LSTM+BN4--26--0.011.hdf5')
        print("....Previous weight data...")
    except:
        print("...New weight data...")
        pass

    early = EarlyStopping(monitor = 'loss' , min_delta = 0.001, patience = 4, mode = 'min', verbose =1 )
    checkpoint = ModelCheckpoint(filepath = 'LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor = 'loss' , verbose = 1, mode = 'min', period = 1)
    ada = Adadelta()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)


    dataset_train = mjsynth(args[0],args[1],batch_size)
    dataset_valid = mjsynth(args[0],args[1],validation_batch_size)


    model.fit_generator(generator= dataset_train,
                        steps_per_epoch=int(dataset_train.dataset_size // batch_size),
                        epochs=30,
                        callbacks=[early,checkpoint],
                        validation_data=dataset_valid,
                        validation_steps=int(dataset_valid.dataset_size // validation_batch_size),
                        use_multiprocessing=True,
                        workers=16,
                        max_queue_size=32)

if __name__ == "__main__":
    if(len(sys.argv)!=3):
        sys.stderr("Invalid command line inputs. Give Inputs - <name.py><data_folder><annotation_file>")
    else:
        main(sys.argv[1:])

