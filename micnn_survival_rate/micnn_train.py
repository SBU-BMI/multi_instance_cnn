import os
import argparse
import json
import torch

from scheduler import Scheduler


def train(args, config, device):
    sch = Scheduler(args, config, device)
    sch.train_mil()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MICNN')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-g', '--gpu_ids', default='0', type=str,
                           help='indices of GPUs to enable (default: 0)')
    parser.add_argument('--fg', action='store_true', default=False,
                           help='compute foreground')      
    parser.add_argument('--ncores', type=int, default=1, help='number of cores')
    
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda:{0}".format(0) if torch.cuda.is_available() else "cpu")

    train(args, config, device)
    

