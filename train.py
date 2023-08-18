#!/usr/bin/python3

import argparse
import os
from trainer import P2p_Trainer_v6
import yaml

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./Yaml/mrgan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    print("=========config=========")
    print(config)
    print("======config end =======")
    trainer = P2p_Trainer_v6(config)

    trainer.train()
    
    



###################################
if __name__ == '__main__':
    main()