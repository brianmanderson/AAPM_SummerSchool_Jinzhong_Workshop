from keras.utils import Sequence, to_categorical
import glob, os, copy, sys
import numpy as np
from scipy import misc


class data_generator(Sequence):
    def __init__(self, path, batch_size=1, lower_threshold=-75, upper_threshold=100):
        self.path = path
        self.batch_size = batch_size
        self.lower = lower_threshold
        self.upper = upper_threshold
        self.get_training_data()
        self.shuffle()

    def get_training_data(self):
        self.train_vol_names = glob.glob(os.path.join(self.path, '*image*'))

    def shuffle(self):
        self.load_file_list = copy.deepcopy(self.train_vol_names)
        perm = np.arange(len(self.train_vol_names))
        np.random.shuffle(perm)
        self.load_file_list = list(np.asarray(self.load_file_list)[perm])

    def __getitem__(self, item):
        image = misc.imread(self.load_file_list[item])
        annotation = misc.imread(self.load_file_list[item].replace('image','mask'))[None,...]
        annotation[annotation>0] = 1
        annotation = to_categorical(annotation,2)
        return image[None,...,None], annotation

    def on_epoch_end(self):
        self.shuffle()

    def __len__(self):
        return len(self.load_file_list)


    def normalize(self, X):
        X[X<self.lower] = self.lower
        X[X > self.upper] = self.upper
        X = (X - self.lower)/(self.upper - self.lower)
        return X


if __name__ == '__main__':
    xxx = 1