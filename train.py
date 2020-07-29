from tensorflow.keras import backend as K
from hourglass import *


mobile = False

batch_size = 8
num_stack = 2
epochs = 2

model_path = './trained_models/hg_model'

resume = False
resume_model = None
resume_model_json = None
init_epoch = 0

# Work with tiny network for speed.
tiny = False

# Initialize model architecture.
if tiny:
    xnet = HourglassNet(num_classes = 16, num_stacks = num_stack, num_channels = 128, inres = (192, 192), outres = (48, 48))
else:
    xnet = HourglassNet(num_classes = 16, num_stacks = num_stack, num_channels = 256, inres = (256, 256), outres = (64, 64))

if resume:
    x_net.resume_train(batch_size = batch_size, model_json = model_json, model_weights = resume_model, init_epoch = init_epoch, epochs = epochs)
else:
    xnet.build_model(mobile = mobile, show = True)
    xnet.train(epochs = epochs, model_path = model_path, batch_size = batch_size)