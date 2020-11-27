"""
Trains benchmark nowcast model
"""
import os
import sys
sys.path.append('..')

import argparse
import datetime
import tensorflow as tf
from src.utils import make_callback_dirs
from unet_benchmark import create_model, nowcast_mse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='path to training data file',
                        default='../data/processed/nowcast_training.h5')
    parser.add_argument('--pct_val',type=float,help='Percent of training samples to use for validation', default=0.2)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--n_epochs', type=int, help='number of epochs', default=10)    
    parser.add_argument('--num_train', type=int, help='number of training sequences to read (', default=None)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--start_neurons', type=int, help='Size of U-Net hidden layers', default=16)
    parser.add_argument('--verbosity', type=int, default=2)
    parser.add_argument('--logdir', type=str, help='log directory (will create dated-directory under this)', default='./logs')
    args, unknown = parser.parse_known_args()

    return args


def main(args):

    # Set up directory to write results
    dlogdir=args.logdir+'/'+datetime.datetime.now().strftime('%y%m%d.%H%M%S')
    os.mkdir(dlogdir)

    # Load data
    x_train,x_train,x_val,y_val = load_data(args.train_data, args.num_train, args.pct_val)
    
    # Create model
    inputs,output = create_model(input_shape=x_train.shape, 
                                 num_outputs=y_train.shape[3], 
                                 start_neurons=args.start_neurons)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    loss = nowcast_mse
    metrics = get_metrics(thres=74)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr )
    model.compile(optimizer=opt, loss=nowcast_mse, metrics=metrics)
    
    # Define callbacks
    tensorboard_dir,imgs_dir,weights_dir = make_callback_dirs(dlogdir)
    metrics_file = os.path.join(logdir, 'metrics.csv')
    save_checkpoint = ModelCheckpoint(os.path.join(dlogdir, 'weights_{epoch:02d}.h5'),
                                      save_best=True, save_weights_only=False, mode='auto')

    tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)    
    csv_callback = CSVLogger( metrics_file ) 
    callbacks = [csv_callback, save_checkpoint, tensorboard_callback]

    # Train!
    model.fit(x=X_train, y=Y_train, batch_size=args.batch_size,
              epochs=args.n_epochs, validation_data=(x_val,y_val),
              callbacks=callbacks,verbose=args.verbosity)


def load_data(train_data,num_train=None, pct_val=0.2):
    with h5py.File(train_data, mode='r') as hf:
        keys = list(hf.keys())
        in_keys  = [k in keys if 'IN' in k]
        out_keys = [k in keys if 'OUT' in k]
        IN=[]
        for ik in in_keys:
            IN.append(hf[ik][:num_train])
        IN=np.concatenate(IN,axis=3)
        OUT=[]
        for ok in out_keys:
            OUT.append(hf[ok][:num_train])
        OUT=np.concatenate(OUT,axis=3)

    val_idx = int((1-pct_validation)*IN.shape[0])
    train_IN,val_IN= np.split(IN,val_idx,axis=0)
    train_OUT,val_OUT= np.split(OUT,val_idx,axis=0)
    return (train_IN,train_OUT,val_IN,val_OUT)


def get_metrics(thres):
    # probability of detection (i.e. recall)
    def pod(y_true,y_pred):
        return probability_of_detection(y_true,y_pred,np.array([thres],dtype=np.float32))
    # success rate (i.e. precision)
    def sucr(y_true,y_pred):
        return success_rate(y_true,y_pred,np.array([thres],dtype=np.float32))
    # critical success index (i.e. IoU)
    def csi(y_true,y_pred):
        return critical_success_index(y_true,y_pred,np.array([thres],dtype=np.float32))
    return [pod,sucr,csi]


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)   
    main(args)    
    print('all done')

    
