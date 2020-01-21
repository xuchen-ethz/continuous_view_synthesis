import time
import os
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.base_model import BaseModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CustomDatasetDataLoader(opt)
dataset = data_loader.load_data()
model = BaseModel(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
for i, data in enumerate(dataset):
    if i >= opt.how_many: break
    model.set_input(data)
    print('%04d: process image...' % (i))
    visualizer.save_current_anim(webpage, model.get_current_anim(), "%04d"%i)
