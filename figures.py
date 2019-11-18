"""
Module to generate plots for DDAO stain images

@author: Monee McGrady
"""

import rhizoblot_module_v2 as mod
import numpy as np
import scipy.stats
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
import seaborn as sns

# global variables for figure cosmetics
c = "cividis"


def img_dist_plot(mask, overlay):
    """Takes in image of root mask or material mask and
    returns a plot of how far every pixel in the image is
    from the nearest root/material.

    Arguments:
    mask    -- image with masks
    overlay -- image with mask of what you want to impose
    over the plot
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (12, 10)
    # use if overlay is material
#     overlay2 = mod.mat_mask(mask)
#     just_overlay = np.uint8(overlay2)
#     just_overlay = cv2.Canny(just_overlay, 0, 1)
#     just_overlay = np.ma.masked_where(just_overlay == 0, just_overlay)
    # create the root mask, 1=not root 0=root
#     root_mask = mod.create_root_mask(mask)
    mat_mask = mod.mat_mask(mask)
    # calculate distance of each pixel from closest root/material
    distance = mod._calc_dist(mat_mask)
    plt.imshow(distance, cmap=c)
    # use if overlay is roots
    root = mod.create_root_mask(overlay)
    root = 1 - root
    just_root_mask = np.ma.masked_where(root == 0, root)
    plt.xticks([0, 300, 600, 800, 1100, 1400, 1700],
               ("0", "15", "30", "0", "15", "30", "45"))
    plt.yticks([0, 200], ("0", "10"))
    clbar = plt.colorbar()
#     plt.clim(0, 20)
#     clbar.set_ticks([0, 5, 10, 15, 20])
    plt.imshow(just_root_mask, cmap="Greys")
#     plt.imshow(just_overlay, cmap="Greys")
#     clbar.set_label('Distance from root (mm)')
    # turn off axis unless trying to scale
    plt.axis("off")
    plt.show()


def avg_near_single_root(root_fname, om_fname, ddao):
    """Create image of average pixel intensity alongside a single root
    with organic material displayed.

    Arguments:
    root_fname -- image with root mask
    om_fname   -- image with organic material mask
    ddao       -- image of ddao stain
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (12, 9)
    root_mask = mod.create_root_mask(root_fname)
    # detect organic material and display it in the image
    om = mod.get_vals(om_fname, w=1, max_val=None)
    om = om.T
    just_om = np.uint8(om)
    just_om = cv2.Canny(just_om, 0, 1)
    mod.show(just_om, cmap="Greys")
    # create mask of organic material so you don't include it
    # in calculation of average pixel intensity (only want soil)
    om_mask = cv2.imread(om_fname)
    om_mask = 1 - np.array(cv2.cvtColor(om_mask, cv2.COLOR_BGR2GRAY)) / 255.
    # get pixel intensity everywhere except root
    new_array = root_mask*ddao
    # calculate average pixel intensity for pixels on either side
    #  of a single root
    count = -1
    # only go out 200 pixels from either side of the root
    to_plot = np.zeros([root_mask.shape[0], 40])
    no_added_values = []
    for i in range(len(new_array)):
        root = []
        for j in range(len(new_array[i])):
            pixel = new_array[i][j]
            org = om_mask[i][j]
            # keep track of where the root is
            if pixel == 0:
                root.append(j)
            else:
                pass
            if org == 1:
                new_array[i][j] = 0
        # wherever there is root, look on either side and average
        # the pixel intensities together
        if root != []:
            count += 1
            # find center pixel of root
            root_point = root[int(len(root)/2)]
            if (root[0]-41) <= 0:
                start = 0
            else:
                start = (root[0] - 41)
            begin = new_array[i][start:root[0] - 1]
            if (root[-1]+21) >= len(new_array[i]-1):
                stop = len(new_array[i])-1
            else:
                stop = root[-1]+41
            end = new_array[i][(root[-1]+1):stop]
            smaller_new = []
            for k in range(len(begin)):
                b = begin[len(begin)-k-1]
                if k >= (len(end) - 1) and b != 0:
                    smaller_new.append(b)
                else:
                    e = end[k]
                    if b != 0 and e != 0:
                        avg_pix = (b + e) / 2
                    elif b != 0 and e == 0:
                        avg_pix = b
                    elif b == 0 and e != 0:
                        avg_pix = e
                    smaller_new.append(avg_pix)
            # add extra pixels as placeholders if area around root doesn't
            # extend 200 pixels out
            if len(smaller_new) < 40:
                to_add = 40-len(smaller_new)
                for l in range(to_add):
                        smaller_new.append(0)
        else:
            count += 1
            smaller_new = []
            for d in range(40):
                smaller_new.append(0)
        for k in range(len(smaller_new)):
            to_plot[count][k] = smaller_new[k]
    mod.show(to_plot, cmap=c)
    clbar = plt.colorbar()
    plt.clim(0.0, 0.85)
    clbar.set_ticks([0.0, 0.20, 0.40, 0.60, 0.80])
    return to_plot


# helper functions for avg_near_multiple_root()
def find_one_root(root):
    """Find a single root.
    """
    one_root = []
    rest = []
    for i in range(len(root)-1):
        if root[i+1] != root[i]+1:
            rest = root[:i+1]
            one_root = root[i+1:]
    return rest, one_root


def find_all_roots(root, all_ones):
    """Recursive function that will go through and find all
    of the separate roots in the image.

    Arguments:
    root     -- a list of positions for one root
    all_ones -- list of roots
    """
    all_r, one_root = find_one_root(root)
    all_ones.append(one_root)
    if all_r == []:
        return all_ones
    else:
        all_r = find_all_roots(all_r, all_ones)
        return all_ones


def either_side(root, i, new_array):
    """Average pixels on either side of one root.

    Arguments:
    root      -- a list of positions for one root
    i         -- line you are currently on in the image
    new_array -- array of pixel intensities everywhere
    except for the root
    """
    root_point = root[int(len(root)/2)]
    if (root[0]-41) <= 0:
        start = 0
    else:
        start = (root[0] - 41)
    begin = new_array[i][start:root[0] - 1]
    if (root[-1]+21) >= len(new_array[i])-1:
        stop = len(new_array[i])-1
    else:
        stop = root[-1]+41
    end = new_array[i][(root[-1]+1):stop]
    smaller_new = []
    for j in range(len(begin)):
        b = begin[len(begin)-j-1]
        if j >= (len(end) - 1):
            smaller_new.append(b)
        else:
            e = end[j]
            if b != 0 and e != 0:
                avg_pix = (b + e) / 2
            elif b != 0 and e == 0:
                avg_pix = b
            elif b == 0 and e != 0:
                avg_pix = e
            smaller_new.append(avg_pix)
    if len(smaller_new) < 40:
        to_add = 40 - len(smaller_new)
        for l in range(to_add):
            smaller_new.append(0)
    return smaller_new


def root_checker(root1, root2):
    """Average the average pixel values of two roots together.
    """
    new_list = []
    for i in range(len(root1)):
        if root1[i] == 0 and root2[i] == 0:
            new_list.append(0)
        elif root1[i] == 0 and root2[i] != 0:
            new_list.append(root2[i])
        elif root1[i] != 0 and root2[i] == 0:
            new_list.append(root1[i])
        else:
            new_list.append((root1[i]+root2[i]) / 2)
    return new_list


def avg_near_multiple_root(root_fname, q_fname, ddao):
    """Create image of average pixel intensity alongside multiple roots
    with quartz displayed.

    Arguments:
    root_fname -- image with root mask
    q_fname    -- image with quartz material mask
    ddao       -- image of ddao stain
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (12, 9)
    root_mask = mod.create_root_mask(root_fname)
    # put the quartz in the image
    qu = mod.get_vals(q_fname, w=1, max_val=None)
    qu = qu.T
    quartz = np.uint8(qu)
    show_quartz = cv2.Canny(quartz, 0, 1)
    mod.show(show_quartz, cmap="Greys")
    # create mask of quartz so you can avoid it
    q_mask = cv2.imread(q_fname)
    q_mask = 1 - np.array(cv2.cvtColor(q_mask, cv2.COLOR_BGR2GRAY)) / 255.
    # get pixel intensity everywhere except root
    new_array = root_mask * ddao
    count = -1
    to_plot = np.zeros([root_mask.shape[0], 40])
    for i in range(len(new_array)):
        root = []
        for j in range(len(new_array[i])):
            pixel = new_array[i][j]
            q = q_mask[i][j]
            if pixel == 0:
                root.append(j)
            else:
                pass
            if q == 1:
                new_array[i][j] = 0
        sep_root = find_all_roots(root, [])
        sep_roots = [x for x in sep_root if x != []]
        if sep_roots != []:
            count += 1
            averaged = []
            either = [either_side(y, i, new_array) for y in sep_roots]
            either = [s for s in either if len(s) == 40]
            if len(either) == 1:
                averaged = either[0]
            elif len(either) == 2:
                averaged = root_checker(either[0], either[1])
            elif len(either) == 3:
                new_avg = root_checker(either[0], either[1])
                averaged = root_checker(new_avg, either[2])
            elif len(either) == 4:
                new_avg = root_checker(either[0], either[1])
                new_new_avg = root_checker(new_avg, either[2])
                averaged = root_checker(new_new_avg, either[3])
            else:
                new_avg = root_checker(either[0], either[1])
                new_new_avg = root_checker(new_avg, either[2])
                newest_avg = root_checker(new_new_avg, either[3])
                averaged = root_checker(newest_avg, either[4])
        else:
            count += 1
            averaged = []
            for d in range(40):
                averaged.append(0)
        for k in range(len(averaged)):
            to_plot[count][k] = averaged[k]
    mod.show(to_plot, cmap=c)
    clbar = plt.colorbar()
    plt.clim(0.0, 0.85)
    clbar.set_ticks([0.0, 0.20, 0.40, 0.60, 0.80])
    return to_plot


def just_root(root_fname, ddao, mat_fname):
    """Create image that shows just the pixel intensity along
    the roots.

    Arguments:
    root_fname -- Image with root mask.
    ddao       -- Image of DDAO stain.
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (12, 9)
    # overlay organic material (only for root 1)
    om = mod.get_vals(mat_fname, w=1, max_val=None)
    om = om.T
    just_om = np.uint8(om)
    just_om = cv2.Canny(just_om, 0, 1)
    root_mask = mod.create_root_mask(root_fname)
    just_root = np.zeros([1870, 880])
    for i in range(len(ddao)):
        for j in range(len(ddao[i])):
            if root_mask[i][j] == 0:
                just_root[i][j] = ddao[i][j]
            else:
                just_root[i][j] = 0
    just_om = np.ma.masked_where(just_om == 0, just_om)
    just_root_mask = np.ma.masked_where(just_root == 0, just_root)
    plt.imshow(just_root_mask, cmap=c)
    clbar = plt.colorbar()
    plt.clim(0.0, 0.85)
    clbar.set_ticks([0.0, 0.20, 0.40, 0.60, 0.80])
#     plt.imshow(just_om)
    plt.axis('off')
    return just_root


# helper function for avg_distance()
def get_avgs(array):
    """Calculate the average pixel intensity for each line of
    pixels.
    """
    avg_depth_r = []
    for i in range(len(array)):
        single_line = []
        for j in range(len(array[i])):
            if array[i][j] == 0:
                pass
            else:
                single_line.append(array[i][j])
        if len(single_line) != 0:
            avg_pix_r = sum(single_line)/len(single_line)
            avg_depth_r.append(avg_pix_r)
    return avg_depth_r


def get_dev(array):
    """Calculate standard deviation for each row of pixels.
    """
    sd = np.zeros([len(array)])
    for i in range(len(array)):
        sd[i] = np.std(array[i])
    return sd


def avg_distance(array1, array2):
    """Generate a distance plot of the average pixel intensities
    going down the root.

    Arguments:
    array1 -- 2D array of pixels of either just the root or the
    area around the root (from just_root() or avg_near_multiple_root())
    array2 -- 2D array of pixels of either just the root or the
    area around the root (from just_root() or avg_near_single_root()
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (9, 12)
    # generate array of pixel averages going down the root
    avg_depth_r1 = get_avgs(array1)
#     sd1 = np.std(avg_depth_r1)
    sd1 = get_dev(array1[:len(avg_depth_r1)])
    avg_depth_r2 = get_avgs(array2)
    # make sure the lengths match
    avg_depth_r2 = avg_depth_r2[:len(avg_depth_r1)]
#     sd2 = np.std(avg_depth_r2)
    sd2 = get_dev(array2[:len(avg_depth_r2)])
    # convert pixels to mm
    # the length of yax needs to be changed to match the length
    # of the average depth arrays
    yax = []
    for y in range(len(avg_depth_r1)):
        yax.append(y*0.05)
    # generate distance plot
    fig = plt.figure()
    ax = plt.axes()
    # add error shading
    ax.errorbar(avg_depth_r1, yax, xerr=sd1, ecolor='skyblue', elinewidth=1,
                zorder=1, alpha=0.25)
    ax.errorbar(avg_depth_r2, yax, xerr=sd2, ecolor='navajowhite',
                elinewidth=1, zorder=1, alpha=0.3)
    ax.plot(avg_depth_r1, yax, color='tab:blue', linewidth=2.0, zorder=2)
    ax.plot(avg_depth_r2, yax, color='tab:orange', linewidth=2.0, zorder=2)
    plt.gca().invert_yaxis()
    root1 = mpatches.Patch(color='skyblue', label='Root 1')
    root2 = mpatches.Patch(color='navajowhite', label='Root 2')
    # add in patches for organic material and quartz
    om1 = mpatches.Rectangle((0.01, 11.9), 0.02, 12.65, color="black")
    om2 = mpatches.Rectangle((0.01, 55.85), 0.02, 12.7, color="black")
    om3 = mpatches.Rectangle((0.01, 91.05), 0.02, 2.45, color="black")
    q1 = mpatches.Rectangle((0.76, 16.85), 0.02, 4.2, color="black",
                            linestyle='dashed', linewidth=2.0, fill=False)
    q2 = mpatches.Rectangle((0.76, 59.65), 0.02, 5.6, color="black",
                            linestyle='dashed', linewidth=2.0, fill=False)
    q3 = mpatches.Rectangle((0.76, 91.35), 0.02, 2.15, color="black",
                            linestyle='dashed', linewidth=2.0, fill=False)
    rectangles = [om1, om2, om3, q1, q2, q3]
    for r in range(len(rectangles)):
        plt.gca().add_patch(rectangles[r])
    plt.legend(handles=[root1, root2],  prop={'size': 20})
    plt.xlim(0.0, 0.8)
    plt.ylim(93.5, 0)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)


def twoD_hist(array1, array2):
    """Generate a 2D array of pixel intensity and distance from the root.

    Arguments:
    array1 -- 2D array of pixels of either just the root or the
    area around the root (from just_root() or avg_near_multiple_root())
    array2 -- 2D array of pixels of either just the root or the
    area around the root (from just_root() or avg_near_single_root()
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (12, 7)
    # initialize array that will be the x axis
    # this needs to change based on what arrays you are using and
    # the shape of them
    # these are for outputs from just_root()
#     x = np.zeros([array1.shape[1], array1.shape[0]])
#     x2 = np.zeros([array2.shape[1], array2.shape[0]])
    # these are for outputs from avg_near_multiple_root() and
    # avg_near_single_root()
    x = np.zeros([array1.shape[0], array1.shape[1]])
    x2 = np.zeros([array2.shape[0], array1.shape[1]])
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = j + 1
    for i in range(len(x2)):
        for j in range(len(x2[i])):
            x2[i][j] = j + 1

    # take the transpose if the array is from just_root
    # so you can look at vertical distance instead of horizontal
    # (when setting for_hist)
#     array1 = array1.T
    for_hist1 = array1.flatten()
    x = x.flatten()
    new_hist1 = []
    new_x = []
    for m in range(len(for_hist1)):
        if for_hist1[m] != 0:
            new_hist1.append(for_hist1[m])
            new_x.append(x[m])
        else:
            pass
#     array2 = array2.T
    for_hist2 = array2.flatten()
    x2 = x2.flatten()
    new_hist2 = []
    new_x2 = []
    for m in range(len(for_hist2)):
        if for_hist2[m] != 0:
            new_hist2.append(for_hist2[m])
            new_x2.append(x2[m])
    # use range=[[0.0, 1.0], [0, 93.5]] for the root itself
    # use range=[[0, 200], [0.0, 1.0]] for along the root
    fig, ax = plt.subplots()
    # new_hist = x, new_x = y for root itself
    # new_x = x, new_hist = y for along the roots
    cax = ax.hist2d(new_x, new_hist1, bins=50, range=[[0, 200],
                    [0.0, 1.0]], normed=True, cmap=c)
    # invert axis for root only
#     plt.gca().invert_yaxis()
    cax[3].set_clim(vmin=0, vmax=0.35)
    cbar = fig.colorbar(cax[3], cmap=c, ticks=[0, 0.35])
    cbar.ax.set_yticklabels(['low', 'high'])
    # use this x-axis option for along the root, otherwise default is fine
    plt.xticks([0, 50, 100, 150, 200], ("0", "2.5", "5", "7.5", "10"))
    plt.show()
    return new_hist1, new_hist2


def twoD_hist_mat(root_fname, mat_fname, just_root, ddao):
    """Generate a 2D histogram of pixel intensity and distance from
    material.

    Arguments:
    root_fname -- Image with root mask.
    mat_fname  -- Image with material mask.
    just_root  -- Output from just_root().
    ddao       -- DDAO stain image.
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (12, 7)
    # calculate distance from material to any pixel
    mat = mod.mat_mask(mat_fname)
    dist_m = mod._calc_dist(mat)
    # translate this to only distance to the root or soil
    dist = np.zeros([mat.shape[0], mat.shape[1]])
    pix_int = np.zeros([mat.shape[0], mat.shape[1]])
    root_mask = mod.create_root_mask(root_fname)
    for i in range(len(root_mask)):
        for j in range(len(root_mask[i])):
            # == 0 for looking at distance from material to
            # roots, == 1 for looking at distance from
            # material to soil
            if root_mask[i][j] == 1:
                dist[i][j] = dist_m[i][j]
                pix_int[i][j] = ddao[i][j]
            else:
                dist[i][j] = 0
                pix_int[i][j] = 0
    # create the 2D histogram
    x = dist.flatten()
    y = pix_int.flatten()
    plot_x = []
    plot_y = []
    for k in range(len(x)):
        if x[k] != 0 and x[k] < 2.0:
            plot_x.append(x[k])
            plot_y.append(y[k])
        else:
            pass
    fig, ax = plt.subplots()
    cax = ax.hist2d(plot_x, plot_y, bins=50, range=[[0, 2], [0, 1.0]],
                    normed=True, cmap=c)
    plt.locator_params(axis='x', nbins=5)
    cax[3].set_clim(vmin=0, vmax=2.5)
    cbar = fig.colorbar(cax[3], cmap=c, ticks=[0, 2.5])
    cbar.ax.set_yticklabels(['low', 'high'])
    plt.show()
    return plot_x, plot_y


# helper function for avg_mat
def avg_bin(dist, pix_int):
    """Average pixel intensities based on bin of distance from
    material to root/soil.

    Arguments:
    dist -- array of distances from material to root/soil
    (output one from twoD_hist_mat)
    pix_int -- array of pixel intensities that corresponds to dist
    (output two from twoD_hist_mat)
    """
    # create a dictionary to sort distances with corresponding pixel
    # intensities
    dict = {}
    for i in range(len(dist)):
        dict[dist[i]] = pix_int[i]
    bins = []
    b = 0.1
    start = 0
    while start < 2:
        if start == 0:
            bins.append([0, b])
        else:
            bins.append([start, start + b])
        start += 0.001
    full_bins = []
    for j in range(len(bins)):
        one_bin = []
        for k in dict.keys():
            if bins[j][0] <= k <= bins[j][1]:
                one_bin.append(dict[k])
            else:
                pass
        full_bins.append(one_bin)
    # calculate average pixel intensity for each bin
    avgs = []
    sd = []
    yax = []
    for a in range(len(full_bins)):
        if len(full_bins[a]) > 1:
            avgs.append(sum(full_bins[a]) / len(full_bins[a]))
            sd.append(np.std(full_bins[a]))
            yax.append(bins[a])
    return avgs, yax, sd


def avg_mat(dist1, pix_int1, dist2, pix_int2):
    """Plot average pixel intensities against distance from
    material.

    Arguments:
    dist1 -- array of distances from material to root/soil
    (output one from twoD_hist_mat for root1)
    pix_int1 -- array of pixel intensities that corresponds to dist1
    (output two from twoD_hist_mat for root1)
    dist2 -- array of distances from material to root/soil
    (output one from twoD_hist_mat for root2)
    pix_int2 -- array of pixel intensities that corresponds to dist2
    (output two from twoD_hist_mat for root2)
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (9, 12)
    avg1, bins1, sd1 = avg_bin(dist1, pix_int1)
    avg2, bins2, sd2 = avg_bin(dist2, pix_int2)
    avg2 = avg2[:len(avg1)]
    sd2 = sd2[:len(sd1)]
    # create y axis
    yax = []
    for i in range(len(bins1)):
        yax.append(bins1[i][1])
    # create a line plot of avg pixel intensities
    # generate distance plot
    fig = plt.figure()
    ax = plt.axes()
    # add error shading
    ax.errorbar(avg1, yax, xerr=sd1, ecolor='skyblue', elinewidth=1,
                zorder=1, alpha=0.25)
    ax.errorbar(avg2, yax, xerr=sd2, ecolor='navajowhite', elinewidth=1,
                zorder=1, alpha=0.3)
    ax.plot(avg1, yax, color='tab:blue', linewidth=2.0, zorder=2)
    ax.plot(avg2, yax, color='tab:orange', linewidth=2.0, zorder=2)
    plt.gca().invert_yaxis()
    root1 = mpatches.Patch(color='skyblue', label='Root 1')
    root2 = mpatches.Patch(color='navajowhite', label='Root 2')
    plt.legend(handles=[root1, root2],  prop={'size': 20})
    plt.xlim(0.0, 0.8)
    plt.ylim(2, 0)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='x', nbins=5)
    plt.xticks(ha='left')


def oneD_hist(array1, array2):
    """Generate a 1D histogram of pixel intensity.

    Arguments:
    array1 -- 1D array of pixel intensities (first output from
    twoD_hist)
    array2 -- 1D array of pixel intensities (second output from
    twoD_hist)
    """
    plt.rcParams.update({'font.size': 24})
    plt.rcParams["figure.figsize"] = (9, 12)
    sns.distplot(array2, color="navajowhite", kde=False, norm_hist=True,
                 label="Root 2", hist_kws=dict(alpha=0.7))
    sns.distplot(array1, color="skyblue", kde=False, norm_hist=True,
                 label="Root 1", hist_kws=dict(alpha=0.7))
    root1 = mpatches.Patch(color='skyblue', label='Root 1')
    root2 = mpatches.Patch(color='navajowhite', label='Root 2')
    plt.legend(handles=[root1, root2],  prop={'size': 20})

    # overlay a normal Gaussian distribution on both histograms
    mean2 = np.mean(array2)
    variance2 = np.var(array2)
    sigma2 = np.sqrt(variance2)
    x2 = np.linspace(0.0, 0.85, 100)
    plt.plot(x2, scipy.stats.norm.pdf(x2, mean2, sigma2), color='tab:orange',
             linewidth=5.0)
    mean1 = np.mean(array1)
    variance1 = np.var(array1)
    sigma1 = np.sqrt(variance1)
    x1 = np.linspace(0.0, 0.85, 100)
    plt.plot(x1, scipy.stats.norm.pdf(x1, mean1, sigma1), color='tab:blue',
             linewidth=5.0)
    plt.locator_params(axis='y', nbins=5)
    plt.xlim(0, 1)
    plt.ylim(0, 8)
    plt.xticks(ha='left')
    plt.show()
