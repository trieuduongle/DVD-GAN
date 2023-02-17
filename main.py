from Dataloader.dataloader_kth import load_test_data, load_train_data
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

    file_names = ['train_data_0_sample_2000_gzip.hkl', 'train_data_1_sample_2000_gzip.hkl', 'train_data_2_sample_1200_gzip.hkl']
    
    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    # make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)

    ##### Dataloader #####
    config.video_path = os.path.join(config.root_path, config.video_path)
    config.annotation_path = os.path.join(config.root_path, config.annotation_path)
    config.mean = get_mean(config.norm_value, dataset=config.mean_dataset)

    config.scales = [config.initial_scale]
    for i in range(1, config.n_scales):
        config.scales.append(config.scales[-1] * config.scale_step)

    train_loader = load_train_data(
        config.batch_size,
        config.val_batch_size,
        config.data_root,
        config.num_workers
    )

    val_loader, test_loader = load_test_data(
        config.dataname,
        config.batch_size,
        config.val_batch_size,
        config.data_root,
        config.num_workers
    )
    val_loader = test_loader if val_loader is None else val_loader

    ##### End dataloader #####

    if config.train:
        start = config.pretrained_model if config.pretrained_model else 0
        runner = (config.total_epoch - start)  // 10

        if config.pretrained_model is not None:
            config.total_epoch = config.pretrained_model + 10
        else:
            config.total_epoch = 10

        for _ in range(runner):
            file_counter = 0
            
            for name in file_names:
                file_counter = file_counter + 1
                train_loader = load_train_data(
                    config.batch_size,
                    config.data_root,
                    name,
                    config.num_workers
                )
                trainer = Trainer(train_loader, val_loader, test_loader, config)

                trainer.train()

                train_loader = None
                trainer = None
                config.pretrained_model = config.total_epoch
                config.total_epoch = config.total_epoch + 10
            


    else:
        # TODO: implement
        tester = Tester(val_loader, config)
        tester.test()


if __name__ == '__main__':
    config = get_parameters()

    for key in config.__dict__.keys():
        print(key, "=", config.__dict__[key])

    main(config)