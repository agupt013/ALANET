import os
import numpy as np

from skimage import io
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


class BlurDataset(Dataset):
    """ Video custom dataset for generating blurry frames online and corresponding sharpe frames """

    def __init__(self, root_dir, seq_len=3, tau=1, delta=1, channel_first=True, transform=None, return_path=False):
        """
        :param root_dir     : path to the data directory.
        :param seq_len      : number of frames to return.
        :param tau          : number of sharpe frames to average to generate a blurry frames.
        :param delta        : number of sharpe frames generated when the aperture is closed.
        :param transform    : any transform that needs to be applied on each frame.

        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.tau = tau
        self.delta = delta
        self.channel_first = channel_first
        self.transform = transform
        self.return_path = return_path
        # assert self.tau > 0
        self.videos, self.videos_info, self.total_frames = self.get_video_info
        # import pdb; pdb.set_trace()
        self.dataset_len = len(self.videos)

        # TODO: Dataset length based on frames
        # self.dataset_len = int(self.total_frames / self.seq_len)
        self.frame_size = self.get_frame_size()

        # TODO : Frame buffer with channel first or batch index first
        if self.channel_first:
            self.blurry_buffer = np.zeros((3, self.seq_len, self.frame_size[0], self.frame_size[1]))
            self.sharp_buffer = np.zeros((3, 2*self.seq_len - 1, self.frame_size[0], self.frame_size[1]))
        else:
            self.blurry_buffer = np.zeros((self.seq_len, 3, self.frame_size[0], self.frame_size[1]))
            self.sharp_buffer = np.zeros((self.seq_len, 3, 2*self.delta + 1, self.frame_size[0], self.frame_size[1]))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        frames = self.videos_info[idx]
        # input(frames)
        # import pdb;
        # pdb.set_trace()
        try:
            start_idx = 5
            # assert len(frame) > self.seq_len*(self.tau-1) + self.delta
            #start_idx = np.random.randint(int(len(frames)/(self.tau-1)) - self.seq_len + 1)*(self.tau-1) + self.delta
        except:
            import pdb;
            pdb.set_trace()
        end_idx = 24 # start_idx + self.seq_len*self.tau
        #end_idx = start_idx + self.seq_len*(self.tau-1) - self.delta + 2
        jump_idx = self.tau - 1

        for f_idx, frame_name in enumerate(frames[start_idx: end_idx: jump_idx]):
            # frame_path = os.path.join(self.root_dir, self.videos[idx], frame_name)
            # frame = self.get_blurry_frame(idx, frames[start_idx + f_idx*jump_idx:
            #                                          start_idx + (f_idx + 1)*jump_idx - self.delta])

            frame = self.get_blurry_frame(idx, frames[start_idx + f_idx*jump_idx - self.delta:
                                                      start_idx + (f_idx)*jump_idx + self.delta + 1
                                               ]
                                          )
            # input(frames[start_idx + f_idx*jump_idx - self.delta:
            #                                           start_idx + (f_idx)*jump_idx + self.delta + 1
            #                                    ])
            # print(frames[start_idx + f_idx*jump_idx - self.delta:
            #                                           start_idx + (f_idx)*jump_idx + self.delta + 1
            #                                    ][5:10:4])
            if self.transform:
                frame = self.transform(frame)
                #sharp_frames = self.transform(sharp_frames)

            if self.channel_first:
                self.blurry_buffer[:, f_idx, :, :] = np.rollaxis(frame, 2, 0)
                #self.sharp_buffer[:, f_idx, :, :, :] = sharp_frames
            else:
                self.blurry_buffer[f_idx, :, :, :] = np.rollaxis(frame, 2, 0)
                #self.sharp_buffer[f_idx, :, :, :, :] = sharp_frames
        self.sharp_buffer = self.get_sharp_frames(idx, frames[
                                                  start_idx:
                                                  end_idx:4
                                                  ]
                                                  )
        if self.return_path:
            return torch.from_numpy(self.blurry_buffer).float(), torch.from_numpy(self.sharp_buffer).float(), frames[
                                                  start_idx:
                                                  end_idx:4
                                                  ]
        else:
            return torch.from_numpy(self.blurry_buffer).float(), torch.from_numpy(self.sharp_buffer).float()

    def get_frame_size(self):
        temp_frame = io.imread(os.path.join(self.root_dir, self.videos[0], self.videos_info[0][0]))
        return temp_frame.shape

    def get_blurry_frame(self, idx, all_frame_names):
        frame_path = os.path.join(self.root_dir, self.videos[idx], all_frame_names[0])
        blur_frame = np.array(io.imread(frame_path), dtype=np.float32)
        for f_name in all_frame_names[1:]:
            frame_path = os.path.join(self.root_dir, self.videos[idx], f_name)
            blur_frame += io.imread(frame_path)
        return blur_frame / float(len(all_frame_names)) #/ 255.0

    def get_sharp_frames(self, idx, all_frame_names):
        sharp_frames = np.zeros((3, len(all_frame_names), self.frame_size[0], self.frame_size[1]))
        for s_idx, f_name in enumerate(all_frame_names):
            frame_path = os.path.join(self.root_dir, self.videos[idx], f_name)
            sharp_frames[:, s_idx, :, :] = np.rollaxis(io.imread(frame_path), 2, 0) #/ 255.0
        return sharp_frames

    @property
    def get_video_info(self):
        videos = sorted(os.listdir(self.root_dir))
        videos_info = {v_idx: sorted(os.listdir(os.path.join(self.root_dir, video_path)))
                       for v_idx, video_path in enumerate(videos)}
        total_frames = np.sum([len(videos_info[frames]) for frames in videos_info.keys()])
        return videos, videos_info, total_frames


if __name__ == '__main__':

    rootdir = '/home/agupt013/datasets/enhancement/adobe_imgs_new/train'
    seqlen = 3
    T = 9
    delta = 5
    trans = None
    channel_first = True
    vid_data = BlurDataset(rootdir, seqlen, T, delta, transform = trans, channel_first=channel_first)

    b_videos, s_videos = vid_data.__getitem__(0)
    print("Blurry Video sequence shape: ", b_videos.shape)
    print("Sharp Video sequence shape: ", s_videos.shape)

    # Visualize the image
    if channel_first:
        b_vid = (1/255.0) * b_videos[:, 1, :, :].numpy()
        s_vid = (1/255.0) * s_videos[:, 1, :, :].numpy()
    else:
        b_vid = (1/255.0) * b_videos[1, :, :, :].numpy()
        s_vid = (1/255.0) * s_videos[1, :, :, :].numpy()

    plt.imshow(np.rollaxis(b_vid, 0, 3))
    plt.figure('Sharp')
    plt.imshow(np.rollaxis(s_vid, 0, 3))
    plt.show()
