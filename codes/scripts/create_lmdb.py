import sys
import os
import glob
import pickle
import lmdb
import cv2
from memCheck import using

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar

PATH = os.path.abspath(os.path.dirname(__file__))
# raw_data_path = os.path.abspath(os.path.join(PATH, "..", "..", "data"))
raw_data_path = '/home/stoplinux/workspace/STSRGAN/data'
raw_subfolder = "rgbd"
data_path = os.path.abspath(os.path.join(PATH, "..", "..", "data"))
subfolder = raw_subfolder + "2"

# configurations
pathList = []

# img_folder = '../../data/raw/val_hr/*'  # glob matching pattern
# lmdb_save_path = '../../data/fast/val_hr.lmdb'  # must end with .lmdb

pathList.append((
    os.path.join(raw_data_path, raw_subfolder, 'val_lr/*'),
    os.path.join(data_path, subfolder, 'val_lr.lmdb')
))
pathList.append((
    os.path.join(raw_data_path, raw_subfolder, 'val_hr/*'),
    os.path.join(data_path, subfolder, 'val_hr.lmdb')
))
pathList.append((
    os.path.join(raw_data_path, raw_subfolder, 'train_lr/*'), 
    os.path.join(data_path, subfolder, 'lr.lmdb')
))
pathList.append((
    os.path.join(raw_data_path, raw_subfolder, 'train_hr/*'),
    os.path.join(data_path, subfolder, 'hr.lmdb')
))

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

if __name__ == "__main__":
    for paths in pathList:
        img_folder = paths[0]
        lmdb_save_path = paths[1]
        print("Saving {}".format(os.path.basename(lmdb_save_path)))

        if not os.path.exists(os.path.dirname(lmdb_save_path)):
            os.makedirs(os.path.dirname(lmdb_save_path))

        img_list = sorted(glob.glob(img_folder))
        data_size = get_size(os.path.dirname(img_folder)) * 25
        env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

        print('Process images...')
        pbar = ProgressBar(len(img_list))
        with env.begin(write=True) as txn:
            for i, v in enumerate(img_list):
                pbar.update('Process {}, {}'.format(v, using("Memory")))
                img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
                base_name = os.path.splitext(os.path.basename(v))[0]
                key = base_name.encode('ascii')
                if img.ndim == 2:
                    H, W = img.shape
                    C = 1
                else:
                    H, W, C = img.shape
                meta_key = (base_name + '.meta').encode('ascii')
                meta = '{:d}, {:d}, {:d}'.format(H, W, C)
                txn.put(key, img)
                txn.put(meta_key, meta.encode('ascii'))
        print('Finish processing {} images.'.format(len(img_list)))

        # create keys cache
        keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
        env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            print('Create lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
            pickle.dump(keys, open(keys_cache_file, "wb"))
        print('Finish creating lmdb keys cache.')
