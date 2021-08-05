#!/bin/bash

CUDA_DEVICE=0                     # DEFINE CUDA DEVICE TO BE USED
BLUR=9                            # DEFINE NUMBER OF FRAMES TO BLUR
DATASET=~/datasets/enhancement/adobe_imgs_new       # PATH TO THE DATASET CREATED USING SCRIPT: create_dataset.py
TRAIN_BATCH=2                     # NUMBER OF TRAINING BATCHES
TEST_BATCH=1                      # NUMBER OF TEST BATCHES
INIT_LR=0.0001                    # INITIAL LEAR RATE
DECODE_MODE='both'                # MODE FOR THE MODEL. OPTION: 1.  deblurring   : 'deblur',
                                  #                             2. interpolation : 'interpolate'
                                  #                             3. both          : 'both'


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --dataset_root $DATASET                                   \
                                                       --checkpoint_dir ./checkpoint/${DECODE_MODE}_${BLUR} \
                                                       --train_batch_size $TRAIN_BATCH                      \
                                                       --test_batch_size $TEST_BATCH                        \
                                                       --num_frame_blur $BLUR                               \
                                                       --init_learning_rate $INIT_LR                        \
                                                       --decode_mode $DECODE_MODE \
                                                       --checkpoint /storage/home/agupt013/projects/video/joint_deblurring/final_codes/deployment/ablation/checkpoint/both_9/epoch-68-test-psnr-31pt6005-ssim-0pt0535.ckpt
