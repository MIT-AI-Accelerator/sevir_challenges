"""
Makes training and test dataset for radar nowcasting model
"""

import sys
sys.path.append('..') # add src to path
import argparse
import logging
import datetime

import os
import h5py
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

import sys
import numpy as np
import pandas as pd

from src.generator import SEVIRGenerator

def parse_args():
  parser = argparse.ArgumentParser(description='Make nowcast training & test datasets using SEVIR')
  parser.add_argument('--input_types', nargs='+', type=str, help='list of input SEVIR modalities', default=['vil'])
  parser.add_argument('--output_types', nargs='+', type=str, help='list of output SEVIR modalities', default=['vil'])
  parser.add_argument('--sevir_data', type=str, help='location of SEVIR dataset',default='../../data/sevir')
  parser.add_argument('--sevir_catalog', type=str, help='location of SEVIR dataset',default='../../data/CATALOG.csv')
  parser.add_argument('--output_location', type=str, help='location of SEVIR dataset',default='../../data/processeed')
  parser.add_argument('--n_train',type=int,help='Maximum number of samples to use for training (None=all)',default=None)
  parser.add_argument('--n_test',type=int,help='Maximum number of samples to use for testing (None=all)',default=None)
  parser.add_argument('--n_chunks', type=int, help='Number of chucks to use (increase if memory limited)',default=20)
  parser.add_argument('--split_date', type=str, help='Day (yymmdd) to split train and test',default='190601')
    

  args = parser.parse_args()
  return args


def main(args):
    """ 
    Runs data processing scripts to extract training set from SEVIR
    """
    logger = logging.getLogger(__name__)
    logger.info('making nowcasting data set from raw data')
    split_date = datetime.datetime.strptime(args.split_date,'%y%m%d')

    trn_generator = get_nowcast_train_generator(sevir_catalog=args.sevir_catalog,
                                                sevir_location=args.sevir_data,
                                                x_types=args.input_types,
                                                y_types=args.output_types,
                                                end_date=split_date)
    tst_generator = get_nowcast_test_generator(sevir_catalog=args.sevir_catalog,
                                               sevir_location=args.sevir_data,
                                               x_types=args.input_types,
                                               y_types=args.output_types,
                                               start_date=split_date )
    
    logger.info('Reading/writing training data to %s' % ('%s/nowcast_training.h5' % args.output_location))
    read_write_chunks('%s/nowcast_training.h5' % args.output_location,trn_generator,args.n_chunks,
                      args.input_types, args.output_types)
    logger.info('Reading/writing testing data to %s' % ('%s/nowcast_testing.h5' % args.output_location))
    read_write_chunks('%s/nowcast_testing.h5' % args.output_location,tst_generator,args.n_chunks,
                      args.input_types, args.output_types)


def read_write_chunks( filename, generator, n_chunks, input_types, output_types ):
    logger = logging.getLogger(__name__)
    chunksize = len(generator)//n_chunks
    # get first chunk
    logger.info('Gathering chunk 0/%s:' % n_chunks)
    X,Y=generator.load_batches(n_batches=chunksize,offset=0,progress_bar=True)
    # Create datasets
    for i,x in enumerate(X):
      with h5py.File(filename, 'w') as hf:
        hf.create_dataset('IN_%s' % input_types[i], data=x,  maxshape=(None,x.shape[1],x.shape[2],x.shape[3]))
    for i,y in enumerate(Y):
      with h5py.File(filename, 'a') as hf:
        hf.create_dataset('OUT_%s' % output_types[i], data=y, maxshape=(None,y.shape[1],y.shape[2],y.shape[3]))
    # Gather other chunks
    for c in range(1,n_chunks+1):
      offset = c*chunksize
      n_batches = min(chunksize,len(generator)-offset)
      if n_batches<0: # all done
        break
      logger.info('Gathering chunk %d/%s:' % (c,n_chunks))
      X,Y=generator.load_batches(n_batches=n_batches,offset=offset,progress_bar=True)
      for i,x in enumerate(X):
        with h5py.File(filename, 'a') as hf:
          k='IN_%s' % input_types[i]
          hf[k].resize((hf[k].shape[0] + x.shape[0]), axis = 0)
          hf[k][-x.shape[0]:]  = x
      for i,y in enumerate(Y):
        with h5py.File(filename, 'a') as hf:
          k='OUT_%s' % output_types[i]
          hf[k].resize((hf[k].shape[0] + y.shape[0]), axis = 0)
          hf[k][-y.shape[0]:]  = y


class NowcastGenerator(SEVIRGenerator):
    """
    Generator that loads full VIL sequences, and spilts each
    event into three training samples, each 12 frames long.

    Event Frames:  [-----------------------------------------------]
                   [----13-----][---12----]
                               [----13----][----12----]
                                          [-----13----][----12----]
    """
    def get_batch(self, idx,return_meta=False):
        (X,_),meta = super(NowcastGenerator, self).get_batch(idx,return_meta=True)  # N,L,W,49
        x_out,y_out=[],[]
        for t in range(len(X)):
          x1,x2,x3 = X[0][:,:,:,:13],X[0][:,:,:,12:25],X[0][:,:,:,24:37]
          y1,y2,y3 = X[0][:,:,:,13:25],X[0][:,:,:,25:37],X[0][:,:,:,37:49]
          Xnew = np.concatenate((x1,x2,x3),axis=0)
          Ynew = np.concatenate((y1,y2,y3),axis=0)
          x_out.append(Xnew)
          y_out.append(Ynew)
        if return_meta:
            # meta is duplicated three times, with adjusted times
            meta['minute_offsets']=':'.join([str(n) for n in range(-60,65,5)])
            meta1,meta2,meta3=meta,meta,meta
            meta1['time_utc'] = meta['time_utc'] - pd.Timedelta(hours=1)
            meta3['time_utc'] = meta['time_utc'] + pd.Timedelta(hours=1)
            return (x_out,y_out),pd.concat((meta1,meta2,meta3))
        else:
            return x_out,y_out

def get_nowcast_train_generator(sevir_catalog,
                                sevir_location,
                                x_types=['vil'],
                                y_types=['vil'],
                                batch_size=8,
                                start_date=None,
                                end_date=datetime.datetime(2019,6,1),
                                **kwargs):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return NowcastGenerator(catalog=sevir_catalog,
                            sevir_data_home=sevir_location,
                            x_img_types=x_types,
                            y_img_types=y_types,
                            batch_size=batch_size,
                            start_date=start_date,
                            end_date=end_date,
                            catalog_filter=filt,
                            **kwargs)

def get_nowcast_test_generator(sevir_catalog,
                               sevir_location,
                               x_types=['vil'],
                               y_types=['vil'],
                               batch_size=8,
                               start_date=datetime.datetime(2019,6,1),
                               end_date=None,
                               **kwargs):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return NowcastGenerator(catalog=sevir_catalog,
                            sevir_data_home=sevir_location,
                            x_img_types=x_types,
                            y_img_types=y_types,
                            batch_size=batch_size,
                            start_date=start_date,
                            end_date=end_date,
                            catalog_filter=filt,
                            **kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    args=parse_args()
    main(args)
