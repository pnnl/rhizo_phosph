# -*- coding: utf-8 -*-
"""
Module to assist with DDAO/SYPRO/microscope images

@author: nune558
"""

# Imports
import cv2
from math import ceil
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from time import time


# Register cividis with matplotlib
rgb_cividis = np.loadtxt('cividis.txt').T
cmap = colors.ListedColormap(rgb_cividis.T, name='cividis')
cm.register_cmap(name='cividis', cmap=cmap)


# Calculations
# Load image directly to grayscale
def load_img(fname):
    img = cv2.imread(fname)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255.
    return img


def get_vals(fname, w=1, max_val=1):
    """ Takes in the filename of an image, returns an array (vals).
    """
    t = time()
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Organic material (set1) is red in the mask
    # Quartz (set2) is blue
    set1 = [255, 0, 0]
    set2 = [0, 0, 255]

    # Initialize array to place image
    # Higher w -> lower res -> faster run time
    vals = np.ones((int(img.shape[1] / w), int(img.shape[0] / w)))
    for i in range(int(img.shape[0] / w)):
        for j in range(int(img.shape[1] / w)):

            if max_val is not None:
                # Square to be converted into 1 pixel
                c = img[w * i: w * (i + 1), w * j: w * (j + 1), :]
                # Choose average color as the new pixel color
                c = np.mean(c, dtype=int)
                c = c / 255. * max_val
                vals[j, i] = c

            else:
                # Means this is the image containing the markers
                # Organic material will be labeled as 0
                # Quartz will be labeled as -1
                # All else (soil/root/etc) will be labeled as 1
                # Judges based on the middle of the square
                # (rather than average)
                c = list(img[w * i + int(w / 2), w * j + int(w / 2), :])
                if c == set1:
                    vals[j, i] = 0
                elif c == set2:
                    vals[j, i] = -1

    # Print time taken to load this image
    print('%.2f' % (time() - t))
    return vals


# Res given in mm
def _calc_dist(img, res=0.05):
    img = np.array(np.copy(img), dtype=np.uint8)
    temp = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    return temp * res


# Calc and apply root mask
def calc_dists(img, root):

    # Dist to Set1 (organic material)
    temp_img = np.copy(img)
    # make everything, including quartz, look like soil/root
    temp_img[np.where(temp_img == -1)] = 1
    res1 = _calc_dist(temp_img)

    # Dist to Set2
    temp_img = np.copy(img)
    temp_img[np.where(temp_img == 0)] = 1
    temp_img[np.where(temp_img == -1)] = 0
    res2 = _calc_dist(temp_img)

    # Dist to root
    temp_img = np.copy(img)
    res3 = _calc_dist(root)

    return res1, res2, res3


# Calculate distance from each pixel to root, line by line
def calc_dist_single_line(img, res=0.05):
    img = np.array(np.copy(img), dtype=np.uint8)
    temp = []

    for i in range(len(img)):
        temp_img = cv2.distanceTransform(img[i], cv2.DIST_L2, 5)
        temp_img = temp_img.flatten()
        temp = np.append(temp, temp_img)
    return temp * res


# Calculate distance from each pixel to root, move down in chunks of 8
def calc_dist_chunks(img, res=0.05):
    img = np.array(np.copy(img), dtype=np.uint8)
    temp = []
    count = 0
    while count < len(img):
        temp_img = cv2.distanceTransform(img[count:count + 8], cv2.DIST_L2, 5)
        temp = np.append(temp, temp_img)
        count += 8
    return temp * res


# Chop used to remove arbitrary pixels at edges (caused by image transform)
def create_root_mask(fname):  # , chop):
    img_g = cv2.imread(fname)
    img_g = 1 - np.array(cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)) / 255.
#     img_g = img_g[:, chop:]
    root_mask = img_g
    root_mask = root_mask < 0.001
#     return (1 - root_mask).T
    return (1 - root_mask)


def mat_mask(fname):
    img_g = cv2.imread(fname)
    img_g = 1 - np.array(cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)) / 255.
    mat_mask = img_g
    mat_mask = img_g > 0.999
    return (1 - mat_mask)


# Create a list of colors to use for plotting
def gen_colors(l):
    m = max(l)
    return [tuple(rgb_cividis[:, int(x / m * 255)]) for x in l]


# There are different ways of normalizing
# Set min of array to 0 and max to 1
def normalize(m):
    m = np.copy(m)
    m[m > np.percentile(m, 99)] = np.percentile(m, 99)
    m -= np.nanmin(m)
    m = m / np.nanmax(m)
    return m


# Removes edge pixels in these images that should be ignored
def normalize_ddao_sypro(ddao, sypro, cutoff):
    ddao_norm = np.copy(ddao)
    sypro_norm = np.copy(sypro)
    ind = np.where(sypro <= cutoff)
    ddao_norm[ind] = np.nan
    sypro_norm[ind] = np.nan
    ddao_norm = normalize(ddao_norm)
    sypro_norm = normalize(sypro_norm)
    return ddao_norm, sypro_norm


# Finds how thick the edge of pixels to ignore is
def calc_chop(ddao, sypro):
    chop1 = min(np.where(sypro[:, 0] == 1)[0][-1],
                np.where(ddao[:, 0] == 1)[0][-1])
    ddao = ddao[chop1:, :]
    sypro = sypro[chop1:, :]
    chop2 = min(np.where(sypro[:, 0] < 0.05)[0][-1],
                np.where(ddao[:, 0] < 0.05)[0][-1])
    chop = chop1 + chop2
    return chop


# Plotting
# Quickly plot images
def show(img, name=None, vmax=None, vmin=None, cmap='viridis', axis='off',
         origin='upper', norm=None):
    plt.imshow(img, cmap=cmap, vmax=vmax, vmin=vmin, interpolation='none',
               origin=origin, norm=norm)
    plt.axis(axis)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if name is not None:
        plt.savefig(name, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.show()


# Same as show() but adds a colorbar. Useful for distance plots
def plot_dist(dist, name=None):
    show(dist)
    plt.colorbar()
    plt.axis('off')
    if name is not None:
        plt.savefig(name, dpi=500, bbox_inches='tight')
    plt.show()


# Distance/trend plotting
def plot_hex(dist, x, l, x_name=None, y_name=None, f_name=None, dist_max=4):
    dist = np.reshape(np.copy(dist), (l))
    x = np.reshape(np.copy(x), (l))

    joint_kws = dict(gridsize=50)
    g = sns.jointplot(dist, x, kind='hex', joint_kws=joint_kws, xlim=(0, 100))

    if x_name is not None and y_name is not None:
        g.set_axis_labels(x_name + ' (mm)', y_name)

        if f_name is not None:
            g.savefig('../Analysis/distance_plots/hex_%s_%s_%s' %
                      (f_name, y_name, x_name), dpi=500)
        plt.show()

    return g


def plot_kde(dist, x, l, x_name=None, y_name=None, f_name=None, dist_max=4):
    dist = np.reshape(np.copy(dist), (l))
    x = np.reshape(np.copy(x), (l))

    ind = np.where(np.logical_and(dist <= dist_max, dist > 0))

    # Data decimation
    dist, x = dist[ind], x[ind]
    final_dist = []
    final_x = []

    w = 0.5
    minx = 0
    ind = np.where(np.logical_and(dist >= minx, dist < minx + w))[0]
    n = len(ind)
    print(n)
    final_dist.extend(list(dist[ind]))
    final_x.extend(list(x[ind]))

    for minx in np.arange(w, dist_max, w):
        ind = np.where(np.logical_and(dist >= minx, dist < minx + w))[0]
        rand_ind = np.random.permutation(ind)[:n]
        final_dist.extend(list(dist[rand_ind]))
        final_x.extend(list(x[rand_ind]))

    if dist_max == 2:
        if y_name == 'DDAO':
            ylim = (0.05, 0.7)
        elif y_name == 'SYPRO':
            ylim = (0.25, 0.55)
    elif dist_max == 4:
        if y_name == 'SYPRO':
            ylim = (0.27, 0.6)
        elif y_name == 'DDAO':
            ylim = (0.1, 0.8)
    final_dist = np.array(final_dist)
    final_x = np.array(final_x)
    # Break into two plots
#    g = sns.jointplot(final_dist, final_x, kind='kde', ylim=ylim)
    sns.set_style('white')
    plt.scatter(final_dist, final_x, c='k', alpha=0.1)
    plt.xlabel(x_name + ' (mm)')
    plt.ylabel(y_name)
    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.xlim(left=0, right=dist_max)
#    plt.show()
#    return
    if x_name is not None and y_name is not None:
        # g.set_axis_labels(x_name + ' (mm)', y_name)
        if f_name is not None:
            plt.savefig('../Analysis/distance_plots/zoom_%s_%s_%s' %
                        (f_name, y_name, x_name), dpi=500)
        plt.show()

    return


# These functions (f1, f2, f3) are arbitrary.
# Trying to find a trend between distances and stain intensity.
def f1(x, y):
    x = np.copy(x)
    y = np.copy(y)
    ind = np.where(np.logical_and(x == 0, y == 0))
    x[ind] = np.nan
    y[ind] = np.nan
    return x + y


def f2(x, y):
    x = np.copy(x)
    y = np.copy(y)
    ind = np.where(np.logical_and(x == 0, y == 0))
    x[ind] = np.nan
    y[ind] = np.nan
    return np.square(x) + np.square(y)


def f3(x, y):
    x = np.copy(x)
    y = np.copy(y)
    ind = np.where(np.logical_and(x == 0, y == 0))
    x[ind] = np.nan
    y[ind] = np.nan
    return x * y
