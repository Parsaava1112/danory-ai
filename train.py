import argparse
from danory.trainer import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Danory GPT v8 trainer')
    p.add_argument('--config',default='configs/pretrain.json');p.add_argument('--stage',choices=['pretrain','sft'],default='pretrain')
    p.add_argument('--resume',default=None);p.add_argument('--init-from',default=None)
    p.add_argument('--steps-this-run',type=int,default=None)
    a=p.parse_args();run(a.config,a.stage,a.resume,a.init_from,a.steps_this_run)
