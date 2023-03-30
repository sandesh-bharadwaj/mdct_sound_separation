from datetime import datetime
import torch
from torch.utils.data import DataLoader

from train import Solver
from dataset import SDXDataset, CLIPSDXDataset
import wandb
import os
import argparse
from tqdm import tqdm
from glob import glob
from dataset import aug_from_str


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bsrnn', 'demucs'])
    parser.add_argument('--target-stem', type=str, default='vocals', choices=['drums', 'bass', 'other', 'vocals'])
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--hop-length', type=int, default=512)
    parser.add_argument('--channels', type=int, default=2)
    # parser.add_argument('--bands', type=int, nargs='+', default=[1000, 4000, 8000, 16000, 20000])
    # parser.add_argument('--num-subbands', type=int, nargs='+', default=[10, 12, 8, 8, 2, 1])
    parser.add_argument('--fc-dim', type=int, default=128)
    parser.add_argument('--num-band-seq-module', type=int, default=12)

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', type=float, default=0.98)
    parser.add_argument('--lambda-spec', type=float, default=0.5)
    parser.add_argument('--valid-per', type=int, default=5)
    parser.add_argument('--ckpt-per', type=int, default=5)
    parser.add_argument('--wandb-per', type=int, default=3)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--num-workers', type=int, default=16)

    # Dataset
    parser.add_argument('--data-dir', type=str,
                        default="/mnt/BigChonk/LinuxDownloads/moisesdb23_bleeding_v1.0_prepared")
    parser.add_argument('--seq-dur', type=int,
                        default=6)
    parser.add_argument('--random-chunks', type=bool,
                        default=True)
    parser.add_argument('--source-augmentations', type=str, nargs="+")

    # CLIP Parameters
    parser.add_argument('--num-mixtures', type=int,
                        default=2)
    parser.add_argument("--use-weighted-loss", action=argparse.BooleanOptionalAction,
                        default=True,
                        )

    # Resume training
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--nowstr', type=str, default=None)
    parser.add_argument('--start-epoch', type=int, default=5)

    # Debug params
    parser.add_argument('--base-dir', type=str, default='./result')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--debug-wandb', action='store_true', default=False)

    # WANDB params
    parser.add_argument('--project', type=str, default='mdct_sound_separation')
    parser.add_argument('--wandb-id', type=str, default=None)
    parser.add_argument('--memo', type=str, default='')

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.run_name = args.model + '_' + args.target_stem
    if args.nowstr is None:
        if args.debug:
            save_dir = 'debug'
        else:
            args.nowstr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            save_dir = '_'.join([args.nowstr, args.run_name]) if args.run_name is not None else args.nowstr
        args.save_dir = os.path.join(args.base_dir, save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        for folder in ['train', 'valid', 'test', 'ckpt', 'wandb']:
            os.makedirs(os.path.join(args.save_dir, folder), exist_ok=True)
    else:
        # RESUME
        args.resume = True
        ckpt_dirs = sorted(glob(os.path.join(args.base_dir, args.nowstr + '*')))
        assert len(ckpt_dirs) == 1
        args.ckpt_dir = ckpt_dirs[0]
        args.save_dir = args.ckpt_dir
        # save_dir = '_'.join([args.nowstr, args.run_name]) if args.run_name is not None else args.nowstr
        ckpt = torch.load(os.path.join(args.ckpt_dir, 'ckpt', str(args.start_epoch).zfill(4) + '.pt'))

    # SETUP WANDB
    if not args.debug:
        wandb_data = wandb.init(
            settings=wandb.Settings(start_method='thread'), project=args.project,
            dir=args.save_dir, resume=False if args.nowstr is None else True,
            config=args
        )

    model = load_model(args)
    print("model loaded")

    if args.resume:
        model.load_state_dict(ckpt['net'])
        print('Pretrained model loaded in epoch: ', args.start_epoch)

    train_dataset, valid_dataset = get_sdx_dataset(args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers)

    print("Train set length: ", len(train_dataset))
    print("Valid set length: ", len(valid_dataset))

    # Optimizer/Scheduler setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=lambda epoch: args.lr_decay ** (epoch // 2))

    # Instantiate Solver
    solver = Solver(
        args=args,
        model=model,
        device=device,
        optimizer=optimizer,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        test_loader=None,
        scheduler=scheduler,
    )
    solver.train()

def get_sdx_dataset(args):
    interferer_files = ['vocals.wav', 'other.wav', 'bass.wav', 'drums.wav']
    interferer_files.remove(args.target_stem + '.wav')
    source_augmentations = aug_from_str(args.source_augmentations)
    train_dataset = CLIPSDXDataset(root=args.data_dir, interferer_files=interferer_files, random_chunks=True,
                                   seq_duration=args.seq_dur, source_augmentations=source_augmentations)
    valid_dataset = CLIPSDXDataset(root=args.data_dir, split="valid", interferer_files=interferer_files,
                                   seq_duration=None)
    return train_dataset, valid_dataset


def load_model(args):
    if args.model == "bsrnn":
        from model import BSRNN
        model = BSRNN(target_stem=args.target_stem,
                      sample_rate=args.sample_rate,
                      n_fft=args.n_fft,
                      hop_length=args.hop_length,
                      channels=args.channels,
                      fc_dim=args.fc_dim,
                      num_band_seq_module=args.num_band_seq_module,
                      num_mixtures=args.num_mixtures
                      )
    else:
        NotImplementedError

    return model


if __name__ == "__main__":
    args = parse_args()
    main(args)
