import torch.utils.data

def CreateDataset(opt):

    if opt.category in ['car', 'chair']:
        from data.shapenet_data_loader import ShapeNetDataLoader
        dataset = ShapeNetDataLoader()
    elif opt.category in ['kitti']:
        from data.kitti_data_loader import KITTIDataLoader
        dataset = KITTIDataLoader()
        print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                yield self.dataloader[i%self.opt.max_dataset_size]
                # break
            else:
                yield data
