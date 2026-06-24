import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting
from custom_dataset import TurboClips, TurboClipsTesting
from model import ContextAwareModel
from train import trainer, test
from loss import ContextAwareLoss, SpottingLoss

# Fixing seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def _resolve_pretrained_kwargs(args):
    if not args.load_weights:
        return dict(
            weights=None,
            load_mode="full",
            class_map=None,
            source_num_classes=17,
        )
    class_map = None
    source_num_classes = args.pretrained_source_classes
    if args.load_mode != "full" and args.class_set == "foul_ball_2":
        from config import foul_ball_classes as class_cfg

        class_map = class_cfg.PRETRAINED_V2_CLASS_MAP
        source_num_classes = class_cfg.PRETRAINED_SOURCE_NUM_CLASSES
    return dict(
        weights=args.load_weights,
        load_mode=args.load_mode,
        class_map=class_map,
        source_num_classes=source_num_classes,
    )


def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))


    # Create Train Validation and Test datasets
    if args.custom_dataset:
        dataset_cls = TurboClips
        dataset_test_cls = TurboClipsTesting
        ds_kwargs = dict(
            path=args.SoccerNet_path,
            splits_path=args.splits_path,
            class_set=args.class_set,
        )
        train_kwargs = dict(
            **ds_kwargs,
            background_weight=args.background_weight,
            event_class_weights=args.event_class_weights,
        )
    else:
        dataset_cls = SoccerNetClips
        dataset_test_cls = SoccerNetClipsTesting
        ds_kwargs = dict(path=args.SoccerNet_path, features=args.features)

    chunk_frames = args.chunk_size * args.framerate
    rf_frames = args.receptive_field * args.framerate

    if not args.test_only:
        dataset_Train = dataset_cls(**train_kwargs, split="train", framerate=args.framerate, chunk_size=chunk_frames, receptive_field=rf_frames, chunks_per_epoch=args.chunks_per_epoch)
        dataset_Valid = dataset_cls(**train_kwargs, split="valid", framerate=args.framerate, chunk_size=chunk_frames, receptive_field=rf_frames, chunks_per_epoch=args.chunks_per_epoch)
        dataset_Valid_metric  = dataset_test_cls(**ds_kwargs, split="valid", framerate=args.framerate, chunk_size=chunk_frames, receptive_field=rf_frames)
    
    split_to_test = "test"
    if args.challenge:
        split_to_test="challenge"
    dataset_Test  = dataset_test_cls(**ds_kwargs, split=split_to_test, framerate=args.framerate, chunk_size=chunk_frames, receptive_field=rf_frames)


    # Create the deep learning model
    pretrained_kwargs = _resolve_pretrained_kwargs(args)
    model = ContextAwareModel(
        input_size=args.num_features,
        num_classes=dataset_Test.num_classes,
        chunk_size=args.chunk_size * args.framerate,
        dim_capsule=args.dim_capsule,
        receptive_field=args.receptive_field * args.framerate,
        num_detections=dataset_Test.num_detections,
        framerate=args.framerate,
        **pretrained_kwargs,
    ).cuda()
    # Logging information about the model
    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # Create the dataloaders for train validation and test datasets
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset_Test,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    # Training parameters
    if not args.test_only:
        criterion_segmentation = ContextAwareLoss(K=dataset_Train.K_parameters)
        criterion_spotting = SpottingLoss(lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                                    betas=(0.9, 0.999), eps=1e-07, 
                                    weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # Start training
        trainer(train_loader, val_loader, val_metric_loader, test_loader, 
                model, optimizer, scheduler, [criterion_segmentation, criterion_spotting], [args.loss_weight_segmentation, args.loss_weight_detection],
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

    # Load the best model and compute its performance
    checkpoint = torch.load(os.path.join("models", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = test(test_loader, model=model, model_name=args.model_name, save_predictions=True)
    logging.info("Best performance at end of training ")
    logging.info("Average mAP: " +  str(a_mAP))
    logging.info("Average mAP visible: " +  str( a_mAP_visible))
    logging.info("Average mAP unshown: " +  str( a_mAP_unshown))
    logging.info("Average mAP per class: " +  str( a_mAP_per_class))
    logging.info("Average mAP visible per class: " +  str( a_mAP_per_class_visible))
    logging.info("Average mAP unshown per class: " +  str( a_mAP_per_class_unshown))

    return a_mAP







if __name__ == '__main__':

    # Load the arguments
    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=True, type=str, help='Path to the SoccerNet-V2 dataset folder' )
    parser.add_argument('--custom_dataset', required=False, action='store_true', help='Use turbo clip dataset (ground_truth.json + 1_ResNET_TF2_PCA512.npy)' )
    parser.add_argument('--splits_path', required=False, type=str, default=None, help='Path to splits.json for custom dataset' )
    parser.add_argument('--class_set', required=False, type=str, default='turbo_8', choices=('goal_1', 'foul_1', 'foul_ball_2', 'turbo_8', 'v2_17'), help='Label vocabulary for custom dataset (goal_1, foul_1, foul_ball_2, turbo_8, or v2_17)' )
    parser.add_argument(
        '--background_weight',
        required=False,
        type=float,
        default=1.0,
        help='Background sampling weight vs event classes (1:1 when 1.0; 1:5 when 5.0). Custom dataset only.',
    )
    parser.add_argument(
        '--event_class_weights',
        required=False,
        type=str,
        default=None,
        help='Comma-separated per-class sampling weights when an event chunk is drawn (e.g. 4,1 for Foul,Ball out of play). Custom dataset only.',
    )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_PCA512.npy",     help='Video features' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='Path to CALF checkpoint (full or partial init)' )
    parser.add_argument(
        '--load_mode',
        required=False,
        type=str,
        default='full',
        choices=('full', 'backbone', 'backbone_seg'),
        help='full: strict load; backbone: conv_1+conv_2 only; backbone_seg: backbone + mapped conv_seg (foul_ball_2)',
    )
    parser.add_argument(
        '--pretrained_source_classes',
        required=False,
        type=int,
        default=17,
        help='Source num_classes when using backbone_seg (SoccerNet-V2 CALF_benchmark=17)',
    )
    parser.add_argument('--model_name',   required=False, type=str,   default="CALF",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--challenge',   required=False, action='store_true',  help='Perform evaluations on the challenge set to produce json files' )

    parser.add_argument('--K_params', required=False, type=type(torch.FloatTensor),   default=None,     help='K_parameters' )
    parser.add_argument('--num_features', required=False, type=int,   default=512,     help='Number of input features' )
    parser.add_argument('--chunks_per_epoch', required=False, type=int,   default=18000,     help='Number of chunks per epoch' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=20,     help='Number of chunks per epoch' )
    parser.add_argument('--dim_capsule', required=False, type=int,   default=16,     help='Dimension of the capsule network' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=120,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--receptive_field', required=False, type=int,   default=40,     help='Temporal receptive field of the network (in seconds)' )
    parser.add_argument("--lambda_coord", required=False, type=float, default=5.0, help="Weight of the coordinates of the event in the detection loss")
    parser.add_argument("--lambda_noobj", required=False, type=float, default=0.5, help="Weight of the no object detection in the detection loss")
    parser.add_argument("--loss_weight_segmentation", required=False, type=float, default=0.000367, help="Weight of the segmentation loss compared to the detection loss")
    parser.add_argument("--loss_weight_detection", required=False, type=float, default=1.0, help="Weight of the detection loss")

    parser.add_argument('--batch_size', required=False, type=int,   default=32,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--patience', required=False, type=int,   default=25,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')

    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    if args.custom_dataset and not args.splits_path:
        parser.error("--splits_path is required when --custom_dataset is set")


    # Logging information
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # Start the main training function
    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
