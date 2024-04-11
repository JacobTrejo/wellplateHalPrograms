import numpy as np
import math

#from programs.f_fitmodel_intrinsic import f_fitmodel_intrinsic
from programs.f_fitmodel_intrinsic_hc_constant import f_fitmodel_intrinsic

from programs.f_x_to_model_intrinsic import f_x_to_model_intrinsic
import cv2 as cv

# These are what the x values correspond to
#d_eye = x(3) * seglen;
#c_eyes = x(4);
#c_belly = x(5);
#c_head = x(6);
#eyes_br = x(7);
#belly_br = x(8);
#head_br = x(9);
#eye_w = x(10) * seglen;
#eye_l = x(11) * seglen;
#belly_w = x(12) * seglen;
#belly_l = x(13) * seglen;
#head_w = x(14) * seglen;
#head_l = x(15) * seglen;


def calculate_intrinsic_parameters(image, position_array):
    approx_centroid = (3 * position_array[0, :] + 1 * position_array[1, :]) / (3 + 1)
    heading_vec = np.array([[- (position_array[1, 1] - position_array[0, 1])],
                            [position_array[1, 0] - position_array[0, 0]]])

    theta_0 = math.atan2(heading_vec[0], heading_vec[1])
    fishlen = np.sqrt(np.sum(heading_vec ** 2))
    seglen = 0.09 * fishlen
    x = np.zeros((19))
    x[0] = approx_centroid[0]
    x[1] = approx_centroid[1]
    x[2] = -theta_0
    x[3] = 0.95
    x[4] = 1.0541  # 1.8
    x[5] = 1.1771  # 1.2
    x[6] = 1.4230  # 1.3
    #x[7] = 350 / 1000
    #x[8] = 200 / 1000
    #x[9] = 120 / 1000
    x[7] = 200 / 1000
    x[8] = 200 / 1000
    x[9] = 200 / 1000
    x[10] = .5
    x[11] = .5
    x[12] = .5
    x[13] = .8
    x[14] = .3
    x[15] = .7
    x[16] = .5
    x[17] = .6
    x[18] = 1

    # seglen = 1.9261
    #
    # im_gray, pt = f_x_to_model_intrinsic(x, seglen, 0, 38, 43)
    # cv.imwrite('test3.png', im_gray)
    # exit()

    x0 = np.copy(x)
    # In case the image is read as RGB
    if len(image.shape) > 2: image = image[..., 0]
    x2, fval = f_fitmodel_intrinsic(x0, seglen, image)

    im_gray, pt = f_x_to_model_intrinsic(x2, seglen, 0, image.shape[1], image.shape[0])
    #cv.imwrite('test.png', im_gray)

    return x2, seglen, im_gray



def calculate_seglen(image, position_array):
    approx_centroid = (3 * position_array[0, :] + 1 * position_array[1, :]) / (3 + 1)
    heading_vec = np.array([[- (position_array[1, 1] - position_array[0, 1])],
                            [position_array[1, 0] - position_array[0, 0]]])

    theta_0 = math.atan2(heading_vec[0], heading_vec[1])
    fishlen = np.sqrt(np.sum(heading_vec ** 2))
    seglen = 0.09 * fishlen
    
    return seglen




