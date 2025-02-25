import os
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage import transform
import pandas as pd


root_dir = os.path.expanduser('~/data/ISIC')

images = { 'train': 'ISIC2018_Task3_Training_Input',
            'val': 'ISIC2018_Task3_Validation_Input',
            'test': 'ISIC2018_Task3_Test_Input'}

annotations = { 'train': 'ISIC2018_Task3_Training_GroundTruth.csv',
                'val': 'ISIC2018_Task3_Validation_GroundTruth.csv',
                'test': 'ISIC2018_Task3_Test_GroundTruth.csv' }

class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
# partitions = ['train', 'val', 'test']
partitions = ['test']

raw_dir = os.path.join(root_dir, 'raw')


if __name__ == '__main__':
    for part in partitions:
        img_dir = os.path.join(raw_dir, images[part])
        anno_dir = os.path.join(raw_dir, annotations[part])
        save_dir = os.path.join(root_dir, part)
        os.makedirs(save_dir, exist_ok=False)
        df = pd.read_csv(anno_dir)
        heads = df.head()
        for head in heads:
            if head != 'image':
                os.makedirs(os.path.join(save_dir, head), exist_ok=False)
        img_names = df.loc[:, 'image']
        for index, row in df.iterrows():
            #print(row)
            img_name = f"{row['image']}.jpg"
            anno_name = [k for k,v in row.items() if v == 1]
            save_path = os.path.join(save_dir, anno_name[0], img_name)
            img_fname = os.path.join(img_dir, img_name)
            if not os.path.exists(img_fname):
                raise FileNotFoundError('img %s not found' % img_fname)
            image = io.imread(img_fname)
            image = transform.resize(image, (240, 240),
                                     order=1, mode='constant',
                                     cval=0, clip=True,
                                     preserve_range=True,
                                     anti_aliasing=True)
            io.imsave(save_path, image)
            print(f'save image: {img_name} to {save_path}')
