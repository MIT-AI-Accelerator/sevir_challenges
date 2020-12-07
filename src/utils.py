"""
Misc util functions
"""
import os
import sys
import datetime

def make_log_dir(exp_dir,prefix='',symlink_name='latest'):
    """
    Creates a dated directory for an experiement, and also creates a symlink 
    """
    linked_dir=exp_dir+'/%s' % symlink_name
    dated_dir=prefix+'%s' % datetime.datetime.now().strftime('%y%m%d%H%M%S')
    os.mkdir(exp_dir+'/'+dated_dir)
    if os.path.islink(linked_dir):
        os.unlink(linked_dir)
    os.symlink(dated_dir,linked_dir)
    return exp_dir+'/'+dated_dir

def make_callback_dirs(logdir):
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    imgs_dir = os.path.join(logdir, 'images')
    if not os.path.isdir(imgs_dir):
        os.makedirs(imgs_dir)
    
    weights_dir = os.path.join(logdir, 'weights')
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    return tensorboard_dir, imgs_dir, weights_dir






