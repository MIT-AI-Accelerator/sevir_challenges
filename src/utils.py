"""
Misc util functions
"""

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