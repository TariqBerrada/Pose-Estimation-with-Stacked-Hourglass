import sys
import os

from blocks import *
from data_process import normalize
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import mean_squared_error
from mpii_datagen import *

import scipy.misc
import numpy as np
import datetime

class HourglassNet(object):
    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres

    def build_model(self, mobile = False, show = False):
        if mobile:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks, self.num_channels, self.inres, self.outres, bottleneck_mobile)
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks, self.num_channels, self.inres, self.outres, bottleneck_block)
        if show:
            self.model.summary()

    def train(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen('./data/mpii/mpii_annotations.json', './data/mpii/images', inres = self.inres, outres = self.outres, is_train = True)
        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma = 1, is_shuffle = True, rot_flag = True, scale_flag = True, flip_flag = True)

        #checkpoint = EvalCallBack(model_path, self.inres, self.outres)

        self.model.fit(x = train_gen, steps_per_epoch = train_dataset.get_dataset_size()//batch_size, epochs = epochs)

    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):
        self.load_model(model_json, model_weights)
        self.model.compile(optimizer = RMSprop(lr=5e-4), loss = mean_squared_error, metrics = ['accuracy'])

        train_dataset = MPIIDataGen('./data/mpii/mpii_annotations.json', './data/mpii/images', inres = self.inres, outres = self.outres, is_train = True)

        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma = 1, is_shuffle = True, rot_flag = True, scale_flag = True, flip_flag = True)

        model_dir = os.path.dirname(os.path.Absapth(model_json))
        print('Resuming training from : %s, %s'%(model_dir, model_json))
        #checkpoint = EvalCallBack(model_dir, self.inres, self.outres)

        self.model.fit(x = train_gen, steps_per_epoch = train_dataset.get_dataset_size()//batch_size, initial_epoch = init_epoch, epochs = epochs)

    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)

    def inference_rgb(self, rgbdata, orgshape, mean = None):
        scale = (orgshape[0]/self.inres[0], orgshape[1]/self.inres[1])
        imgdata = scipy.misc.imresize(rgbdata, self.inres)

        if mean is None:
            mean = np.array([0.4404, 0.444, 0.4327], dtype = np.float32)
        
        imgdata = normalize(imgdata, mean)

        input = imgdata[np.newaxis, :, :, :]

        out = self.model.predict(input)
        return out[-1], scale

    def inference_file(self, imgfile, mean = None):
        imgdata = scipy.misc.imread(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)

        return ret