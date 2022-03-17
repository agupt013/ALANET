# ALANET: Adaptive Latent Attention Network for Joint Video Deblurring and Interpolation

Implementation of our paper titled "[ALANET: Adaptive Latent Attention Network for Joint Video Deblurring and Interpolation](https://arxiv.org/abs/2009.01005)" accepted to [ACM-MM 2020](https://2020.acmmm.org/).
Please refer [project page](https://akashagupta.com/publication/acm2020_alanet/project.html) for more details. This is old version of the code. Some files may be outdated. I have also attached one checkpoint for the model.

## Dataset Processing
Download data video dataset and copy it to './dataset/ ' directory.

## Create Dataset
Run the following script to generate data that can be used for training and testing.
```
cd data

python create_dataset.py --ffmpeg_dir <path-to-ffmpeg-dir>            \
                         --dataset_folder <path-to-store-video-data>  \
                         --videos_folder ./dataset
```


## Training ALANET
Execute run.sh bash script to train the network.

NOTE: You will have to specify parameter before running run.sh script.
       For more details look at run.sh in the provided codes.

## Citation
```
@inproceedings{gupta2020alanet,
  title={ALANET: Adaptive Latent Attention Network for Joint Video Deblurring and Interpolation},
  author={Gupta, Akash and Aich, Abhishek and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={256--264},
  year={2020}
}
```
    
## Contact
Please contact the first author Akash Gupta ([agupt013@ucr.edu](agupt013@ucr.edu)) for any questions.

