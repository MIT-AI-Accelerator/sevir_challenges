"""
Makes training and test dataset for synrad model using SEVIR
"""

# -*- coding: utf-8 -*-
import argparse
import logging
import datetime

import os
import h5py
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

import sys
import numpy as np

from src.data.generator import SEVIRGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Make synrad training & test datasets using SEVIR')
    parser.add_argument('--input_types', nargs='+', type=str, help='list of input SEVIR modalities', default=['ir069','ir107','lght'])
    parser.add_argument('--output_types', nargs='+', type=str, help='list of output SEVIR modalities', default=['vil'])
    parser.add_argument('--sevir_data', type=str, help='location of SEVIR dataset',default='../../data/sevir')
    parser.add_argument('--sevir_catalog', type=str, help='location of SEVIR dataset',default='../../data/CATALOG.csv')
    parser.add_argument('--output_location', type=str, help='location of SEVIR dataset',default='../../data/processeed')
    parser.add_argument('--n_chunks', type=int, help='Number of chucks to use (increase if memory limited)',default=20)
    parser.add_argument('--split_date', type=str, help='Day (yymmdd) to split train and test',default='190601')
    args = parser.parse_args()
    return args

def main(args):
    """ 
    Runs data processing scripts to extract training set from SEVIR
    """
    logger = logging.getLogger(__name__)
    logger.info('making synthetic radar data set from raw data')
    split_date = datetime.datetime.strptime(args.split_date,'%y%m%d')

    trn_generator = get_synrad_train_generator(sevir_catalog=args.sevir_catalog,
                                               sevir_location=args.sevir_data,
                                               x_types=args.input_types,
                                               y_types=args.output_types,
                                               end_date=split_date)
    tst_generator = get_synrad_test_generator(sevir_catalog=args.sevir_catalog,
                                              sevir_location=args.sevir_data,
                                              x_types=args.input_types,
                                              y_types=args.output_types,
                                              start_date=split_date)
    logger.info('Reading/writing training data to %s' % ('%s/synrad_training.h5' % args.output_location))
    read_write_chunks('%s/synrad_training.h5' % args.output_location,
                      trn_generator,
                      args.n_chunks,
                      args.input_types,
                      args.output_types)
    logger.info('Reading/writing testing data to %s' % ('%s/synrad_testing.h5' % args.output_location))
    read_write_chunks('%s/synrad_testing.h5' % args.output_location,
                      tst_generator,
                      args.n_chunks,
                      args.input_types,
                      args.output_types)

def read_write_chunks( filename, generator, n_chunks, input_types, output_types ):
    logger = logging.getLogger(__name__)
    chunksize = len(generator)//n_chunks
    # get first chunk
    logger.info('Gathering chunk 0/%s:' % n_chunks)
    X,Y=generator.load_batches(n_batches=chunksize,offset=0,progress_bar=True)
    # Create datasets
    for i,x in enumerate(X):
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset(input_types[i],  data=x, maxshape=(None,x.shape[1],x.shape[2],x.shape[3]))
    for i,y in enumerate(Y):
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset(output_types[i],  data=y, maxshape=(None,y.shape[1],y.shape[2],y.shape[3]))

    # Gather other chunks
    for c in range(1,n_chunks+1):
        offset = c*chunksize
        n_batches = min(chunksize,len(generator)-offset)
        if n_batches<0: # all done
            break
        logger.info('Gathering chunk %d/%s:' % (c,n_chunks))
        X,Y=generator.load_batches(n_batches=n_batches,offset=offset,progress_bar=True)
        for i,x in enumerate(X):
            with h5py.File(filename, 'w') as hf:
                k=input_types[i]
                hf[k].resize((hf[k].shape[0] + x.shape[0]), axis = 0)
                hf[k][-x.shape[0]:]  = x
        for i,y in enumerate(Y):
            with h5py.File(filename, 'w') as hf:
                k=output_types[i]
                hf[k].resize((hf[k].shape[0] + y.shape[0]), axis = 0)
                hf[k][-y.shape[0]:]  = y


def get_synrad_train_generator(sevir_catalog,
                                sevir_location,
                                x_types=['ir069','ir107','lght'],
                                y_types=['vil'],
                                batch_size=32,
                                start_date=None,
                                end_date=datetime.datetime(2019,6,1) ):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing data
    return SEVIRGenerator(catalog=sevir_catalog,
                         sevir_data_home=sevir_location,
                         x_img_types=x_types,
                         y_img_types=y_types,
                         batch_size=batch_size,
                         start_date=start_date,
                         end_date=end_date,
                         catalog_filter=filt,
                         unwrap_time=True)

def get_synrad_test_generator(sevir_catalog,
                               sevir_location,
                               batch_size=32,
                               start_date=datetime.datetime(2019,6,1),
                               end_date=None):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return SEVIRGenerator(catalog=sevir_catalog,
                         sevir_data_home=sevir_location,
                         x_types=['ir069','ir107','lght'],
                         y_types=['vil'],
                         batch_size=batch_size,
                         start_date=start_date,
                         end_date=end_date,
                         catalog_filter=filt,
                         unwrap_time=True)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    args=parse_args()
    main(args)
