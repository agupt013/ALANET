#### Dataset Processing ####
Download data video dataset and copy it to './dataset/ ' directory.

#### Create Dataset ####
Run the following script to generate data that can be used for training and testing.

cd data;
python create_dataset.py --ffmpeg_dir <path-to-ffmpeg-dir>              \
                          --dataset_folder <path-to-store-video-data>    \
                          --videos_folder ./dataset


#### Training ALANET ####
Execute run.sh bash script to train the network.
NOTE: You will have to specify parameter before running run.sh script.
       For more details look at run.sh in the provided codes.

