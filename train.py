import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from util.visualizer import Visualizer
import copy
from collections import OrderedDict
import tqdm
import torch
from models.base_model import BaseModel
torch.manual_seed(0)

opt = TrainOptions().parse()
data_loader = CustomDatasetDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training samples = %d' % dataset_size)

model = BaseModel(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    iter_start_time = 0
    for i, data in enumerate(dataset):
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:

            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        iter_start_time = time.time()

    model.switch_mode('eval')
    visualizer.display_current_anim(model.get_current_anim(), epoch)
    model.switch_mode('train')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()


