#!/bin/bash

CUDA_DEVICE=0                     # DEFINE CUDA DEVICE TO BE USED
BLUR=9                            # DEFINE NUMBER OF FRAMES TO BLUR
DATASET=<dataset_directory>       # PATH TO THE DATASET CREATED USING SCRIPT: create_dataset.py
TRAIN_BATCH=2                     # NUMBER OF TRAINING BATCHES
TEST_BATCH=2                      # NUMBER OF TEST BATCHES
INIT_LR=0.0001                    # INITIAL LEAR RATE
DECODE_MODE='both'                # MODE FOR THE MODEL. OPTION: 1.  deblurring   : 'deblur',
                                  #                             2. interpolation : 'interpolate'
                                  #                             3. both          : 'both'

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python deblur_and_interpolate.py --dataset_root $DATASET                              \
                                                                   --checkpoint_dir ./checkpoint/${DECODE_MODE}_${BLUR} \
                                                                   --test_batch_size $TEST_BATCH                        \
                                                                   --num_frame_blur $BLUR                               \
                                                                   --decode_mode $DECODE_MODE
