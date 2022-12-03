'''
This file will contain miscellaneous functions that might be useful elsewhere in the package.
'''

from tkinter import Tk
import numpy as np
import scipy.interpolate
import matplotlib
import os

def copy_text_to_clipboard(text : str):
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(text)
    r.update()
    r.destroy()

def get_cmap_from_digitalizer_file(red_file : str, green_file : str, blue_file : str, alpha_file : str=None, num_pts : int=200, saturated_bounds : bool=True) -> matplotlib.colors.ListedColormap:
    '''
    Creates a matplotlib colormap linearly interpolated from three .csv files 
    containing the RGB values. The .csv files should contain ordered pairs with (x, y) 
    where x and y are both between 0 and 1. This function is meant to be used for 
    digitalized colormap plots.

    Parameters
    ----------
    red_file : str
        .csv containing the R values
    green_file : str
        .csv containing the G values
    blue_file : str
        .csv containing the B values
    alpha_file : str, optional
        .csv containing the alpha values. Defaults is None.
    num_pts : int, optional
        The number of points in the colormap. Default is 200.
    saturated_bounds : bool, optional
        If true, 0 and 1 will be added to the ends of all color channels to make the
        bounds black and white. Default is True.

    Returns
    -------
    cmap : matplotlib.colors.ListedColorMap

    '''

    r = np.loadtxt(red_file, delimiter=',')
    g = np.loadtxt(green_file, delimiter=',')
    b = np.loadtxt(blue_file, delimiter=',')

    r_x = r[:, 0]
    r_y = r[:, 1]
    g_x = g[:, 0]
    g_y = g[:, 1]
    b_x = b[:, 0]
    b_y = b[:, 1]

    r_x = np.insert(r_x, 0, 0)
    g_x = np.insert(g_x, 0, 0)
    b_x = np.insert(b_x, 0, 0)
    r_x = np.append(r_x, 1)
    g_x = np.append(g_x, 1)
    b_x = np.append(b_x, 1)

    if saturated_bounds:
        r_y = np.insert(r_y, 0, 0)
        g_y = np.insert(g_y, 0, 0)
        b_y = np.insert(b_y, 0, 0)
        r_y = np.append(r_y, 1)
        g_y = np.append(g_y, 1)
        b_y = np.append(b_y, 1)
    else:
        r_y = np.insert(r_y, 0, np.amin(r_y))
        g_y = np.insert(g_y, 0, np.amin(g_y))
        b_y = np.insert(b_y, 0, np.amin(b_y))
        r_y = np.append(r_y, np.amax(r_y))
        g_y = np.append(g_y, np.amax(g_y))
        b_y = np.append(b_y, np.amax(b_y))

    R_interp = scipy.interpolate.interp1d(r_x, r_y)
    G_interp = scipy.interpolate.interp1d(g_x, g_y)
    B_interp = scipy.interpolate.interp1d(b_x, b_y)

    X = np.linspace(0, 1, num_pts)

    R = R_interp(X)
    G = G_interp(X)
    B = B_interp(X)

    return matplotlib.colors.ListedColormap(np.c_[R, G, B])

def get_w_cmap():
    '''
    Returns the secret sauce cmap
    '''
    path_to_this_file = os.path.dirname(__file__)
    r_file = os.path.join(path_to_this_file, 'cmaps/w_r.csv')
    g_file = os.path.join(path_to_this_file, 'cmaps/w_g.csv')
    b_file = os.path.join(path_to_this_file, 'cmaps/w_b.csv')
    return get_cmap_from_digitalizer_file(r_file, g_file, b_file)