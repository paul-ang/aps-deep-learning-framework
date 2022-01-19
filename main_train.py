import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.aps import APS


def main():
    # Args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # Training
    parser.add_argument('--seed', default=555, type=int,
                        help='Set the random seed.')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate for the optimizers.')
    parser.add_argument('--debug', default=0, type=int,
                        help='Activate debug mode.')
    parser.add_argument('--name', default='Default-name', type=str,
                        help="Name of the experiment folder.")
    parser.add_argument('--monitor_loss', default='val_mae', type=str,
                        help="For early stopping and save best model.")
    parser.add_argument('--image_size', default=(288, 288), type=int,
                        help="image size. (Default = 288x288). ", nargs="+")
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size for the train and val dataloaders. '
                             'Test dataloader always use batch size of 1 for '
                             'visualization compatibility.')
    parser.add_argument('--num_workers', default=6, type=int,
                        help='Num workers for the dataloaders.')
    parser.add_argument('--saved_weight', default='', type=str,
                        help="path to the saved weight.")
    # APS model specific
    parser.add_argument('--lambda_adv', default=1.0, type=float,
                        help="The hyperparameter L_adv for APS.")
    parser.add_argument('--lambda_pix', default=100.0, type=float,
                        help="The hyperparameter L_pix for APS.")
    parser.add_argument('--lambda_str', default=10.0, type=float,
                        help="The hyperparameter L_str for APS.")
    args = parser.parse_args()

    # Setup experiment dir
    if args.debug == 1:
        # A debug save_dir
        save_dir = 'experiments/test-debug'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        # Create an experiment folder for logging and saving model's weights
        save_dir = 'experiments/{}-{}'.format(
            args.name,
            time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(save_dir, visual_folder=True)

    # Some args logic
    if args.debug == 1:
        print("Debug mode on.")
        args.fast_dev_run = True
        args.num_workers = 0

    seed_everything(args.seed)

    # Setup dataloaders
    train_loader, val_loader, test_loader = get_training_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size)

    # Setup model
    assert args.batch_size % 2 == 0, "APS model only works with even number batch size"
    print("APS model")
    model = APS(lr=args.lr, input_size=args.image_size,
                lambda_adv=args.lambda_adv, lambda_pix=args.lambda_pix,
                lambda_str=args.lambda_str)

    # Setup checkpoint callbacks
    save_best_model = ModelCheckpoint(monitor=args.monitor_loss, dirpath=save_dir,
                                      filename='best_model', save_top_k=1,
                                      mode='min', save_last=True, verbose=True)

    early_stop = EarlyStopping(
        monitor='val_mae',
        patience=50,
        mode='min',
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=save_dir,
                                            callbacks=[save_best_model, early_stop])

    if len(args.saved_weight) == 0:  # train
        print("Train and test the model.")
        trainer.fit(model, train_loader, val_loader)

        trainer.test(test_dataloaders=test_loader, ckpt_path='best')
    else:  # test
        print("Test the model using the saved weight.")
        print(f"Using {args.saved_weight}.")
        model = model.load_from_checkpoint(args.saved_weight)

        trainer.test(model=model, test_dataloaders=test_loader)


def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already exists.")
    print('Experiment dir : {}'.format(path))


def get_training_dataloaders(batch_size=32, num_workers:int =6, **kwargs):
    # Integrate your custom dataset class here.
    train_dataset, val_dataset, test_dataset = your_custom_dataset(**kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    main()