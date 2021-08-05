import matplotlib.pyplot as plt
import numpy as np


def plot(tensor, channel_location=0):
    import pdb; pdb.set_trace()
    img = tensor.permute(0, 2, 4, 3, 1)[0, 1, :, :, :].detach().cpu().numpy()
    plt.imshow(img/255.0)
    plt.show()
