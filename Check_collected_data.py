import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
def show_state(image):
    plt.figure('Image sample', figsize=(8, 8))
    # plt.clf()
    plt.subplot(3, 3, 1)
    plt.imshow(image[0])
    plt.axis(False)

    plt.subplot(3, 3, 2)
    plt.imshow(image[1])
    plt.axis(False)

    plt.subplot(3, 3, 3)
    plt.imshow(image[2])
    plt.axis(False)

    plt.subplot(3, 3, 4)
    plt.imshow(image[3])
    plt.axis(False)

    plt.subplot(3, 3, 5)
    plt.imshow(image[4])
    plt.axis(False)

    plt.subplot(3, 3, 6)
    plt.imshow(image[5])
    plt.axis(False)

    plt.subplot(3, 3, 7)
    plt.imshow(image[6])
    plt.axis(False)

    plt.subplot(3, 3, 8)
    plt.imshow(image[7])
    plt.axis(False)

    plt.subplot(3, 3, 9)
    plt.imshow(image[8])
    plt.axis(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图间距
    plt.savefig('Dynamic_samples' + '.png',
                bbox_inches="tight",
                pad_inches=0.3,
                transparent=True,
                facecolor="w",
                edgecolor='w',
                orientation='landscape')
    plt.show()
    # plt.pause(1)


if __name__ == '__main__':

    image = []
    # State = np.load(base_path + "Static_State_" + data_label + ".npy")
    State = np.load('Dynamic_State_1000_win_5_1_2_45.npy')
    H = np.load("H_value_1000_win_5_1_2_45.npy")
    for i in range(9):
        print('Image number:', i + 1)
        index = np.random.randint(1, 50)
        State32 = np.array(State[index], dtype=np.float32)
        State_ = np.transpose(State32, (1, 2, 0))
        print(H[index])
        image.append(State_)  # Do not divide by 100 as the previous version

    show_state(image)
