#!/usr/bin/python3

import argparse
import os
from trainer import UNet_Trainer
import yaml
os.environ['CUDA_VISIBLE_DEVICES']='1'

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.full_load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./Yaml/UNet.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    print("======config=======")
    print(config)

    trainer = UNet_Trainer(config)

    trainer.train()
    

###################################
if __name__ == '__main__':
    main()