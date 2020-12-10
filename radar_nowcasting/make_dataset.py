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
  parser.add_argument('--n_chunks', type=int, help='Number of chucks to use (increase if memory limited)',default=8)
  parser.add_argument('--split_date', type=str, help='Day (yymmdd) to split train and test',default='190601')
  parser.add_argument('--append',action='store_true',help='Wrtie chunks into one single file instead of individual files')
  parser.add_argument('--shuffle',action='store_true',help='Shuffle dataset before writing to h5 files')
    
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
                                                end_date=split_date,
                                                shuffle=args.shuffle)
    tst_generator = get_nowcast_test_generator(sevir_catalog=args.sevir_catalog,
                                               sevir_location=args.sevir_data,
                                               x_types=args.input_types,
                                               y_types=args.output_types,
                                               start_date=split_date,
                                               shuffle=args.shuffle)
    
    logger.info('Reading/writing training data to %s' % ('%s/nowcast_training.h5' % args.output_location))
    read_write_chunks('%s/nowcast_training.h5' % args.output_location,trn_generator,args.n_chunks,
                      args.input_types, args.output_types,append=args.append)
    logger.info('Reading/writing testing data to %s' % ('%s/nowcast_testing.h5' % args.output_location))
    read_write_chunks('%s/nowcast_testing.h5' % args.output_location,tst_generator,args.n_chunks,
                      args.input_types, args.output_types,append=args.append)


def read_write_chunks( out_filename, generator, n_chunks, input_types, output_types, append=False ):
    logger = logging.getLogger(__name__)
    chunksize = len(generator)//n_chunks
    # get first chunk
    logger.info('Gathering chunk 0/%s:' % n_chunks)
    (X,Y),meta=generator.load_batches(n_batches=chunksize,offset=0,progress_bar=True,return_meta=True)

    # Create datasets
    fn,ext=os.path.splitext(out_filename)
    cs = '' if append else '_000'
    filename=fn+cs+ext
    for i,x in enumerate(X):
      with h5py.File(filename, 'w' if i==0 else 'a') as hf:
        hf.create_dataset('IN_%s' % input_types[i], data=x,  maxshape=(None,x.shape[1],x.shape[2],x.shape[3]))
    for i,y in enumerate(Y):
      with h5py.File(filename, 'a') as hf:
        hf.create_dataset('OUT_%s' % output_types[i], data=y, maxshape=(None,y.shape[1],y.shape[2],y.shape[3]))
    if not append:
      meta.to_csv(fn+cs+'_META.csv')
    # Gather other chunks
    for c in range(1,n_chunks+1):
      cs = '' if append else '_%.3d' % c
      filename=fn+cs+ext
      offset = c*chunksize
      n_batches = min(chunksize,len(generator)-offset)
      if n_batches<0: # all done
        break
      logger.info('Gathering chunk %d/%s:' % (c,n_chunks))
      (X,Y),metac=generator.load_batches(n_batches=n_batches,offset=offset,progress_bar=True,return_meta=True)
      if append:
        meta=pd.concat((meta,metac))
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
      else: # write to a new file
        for i,x in enumerate(X):
          with h5py.File(filename, 'w' if i==0 else 'a') as hf:
            hf.create_dataset('IN_%s' % input_types[i], data=x,  maxshape=(None,x.shape[1],x.shape[2],x.shape[3]))
        for i,y in enumerate(Y):
          with h5py.File(filename, 'a') as hf:
            hf.create_dataset('OUT_%s' % output_types[i], data=y, maxshape=(None,y.shape[1],y.shape[2],y.shape[3]))
        metac.to_csv(fn+cs+'_META.csv')
    if append:
        meta.to_csv(fn+cs+'_META.csv')


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
        """
        Splits batch into three hour-long past/future sequences
        """
        (X,_),meta = super(NowcastGenerator, self).get_batch(idx,return_meta=True)  # N,L,W,49
        x_out,y_out=[],[]
        for t in range(len(X)):
          x1,x2,x3 = X[t][:,:,:,:13],X[t][:,:,:,12:25],X[t][:,:,:,24:37]
          y1,y2,y3 = X[t][:,:,:,13:25],X[t][:,:,:,25:37],X[t][:,:,:,37:49]
          Xnew = np.concatenate((x1,x2,x3),axis=0)
          Ynew = np.concatenate((y1,y2,y3),axis=0)
          x_out.append(Xnew)
          if self.x_img_types[t] in self.y_img_types:
            y_out.append(Ynew)
        if return_meta:
            return (x_out,y_out),meta
        else:
            return x_out,y_out
    
    def get_batch_metadata(self,idx):
        """
        Duplicates meta three times and adjusts time stamps
        """
        meta = super(NowcastGenerator, self).get_batch_metadata(idx)
        meta['minute_offsets']=':'.join([str(n) for n in range(-60,65,5)])
        meta1,meta2,meta3=meta.copy(),meta.copy(),meta.copy()
        meta1['time_utc'] = meta['time_utc'] - pd.Timedelta(hours=1)
        meta3['time_utc'] = meta['time_utc'] + pd.Timedelta(hours=1)
        return pd.concat((meta1,meta2,meta3))
        
        
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
