from trainer import Trainer
# from tester import Tester
from torch.backends import cudnn
from utils import make_folder
from parameter import get_parameters

import os


##### Import libary for dataloader #####
##### https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/main.py
from Dataloader.dataloader import load_data
from Dataloader.mean import get_mean


def main(config):
    # For fast training
    cudnn.benchmark = True

    ##### Dataloader #####
    config.video_path = os.path.join(config.root_path, config.video_path)
    config.annotation_path = os.path.join(config.root_path, config.annotation_path)
    config.mean = get_mean(config.norm_value, dataset=config.mean_dataset)

    config.scales = [config.initial_scale]
    for i in range(1, config.n_scales):
        config.scales.append(config.scales[-1] * config.scale_step)

    train_loader, val_loader, test_loader, data_mean, data_std = load_data(
        config.dataname,
        config.batch_size,
        config.val_batch_size,
        config.data_root,
        config.num_workers
    )
    val_loader = test_loader if val_loader is None else val_loader

    ##### End dataloader #####

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    # make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)

    if config.train:
        if config.model=='dvd-gan':
            trainer = Trainer(train_loader, test_loader, config) 
        else:
            trainer = None

        trainer.train()
    else:
        # TODO: implement
        tester = Tester(val_loader, config)
        tester.test()


if __name__ == '__main__':
    config = get_parameters()

    for key in config.__dict__.keys():
        print(key, "=", config.__dict__[key])

    main(config)