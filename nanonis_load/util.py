"""
This file will contain miscellaneous functions that might be useful elsewhere in the package.
"""

import os
from tkinter import Tk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


def copy_text_to_clipboard(text: str):
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(text)
    r.update()
    r.destroy()


def get_cmap_from_digitalizer_file(
    red_file: str,
    green_file: str,
    blue_file: str,
    alpha_file: str = None,
    num_pts: int = 200,
    saturated_bounds: bool = True,
) -> matplotlib.colors.ListedColormap:
    """
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

    """

    r = np.loadtxt(red_file, delimiter=",")
    g = np.loadtxt(green_file, delimiter=",")
    b = np.loadtxt(blue_file, delimiter=",")

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
    """
    Returns the secret sauce cmap
    """
    path_to_this_file = os.path.dirname(__file__)
    r_file = os.path.join(path_to_this_file, "cmaps/w_r.csv")
    g_file = os.path.join(path_to_this_file, "cmaps/w_g.csv")
    b_file = os.path.join(path_to_this_file, "cmaps/w_b.csv")
    return get_cmap_from_digitalizer_file(r_file, g_file, b_file)


def linecut(arr: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
    num_pts = int(np.hypot(i0 - i1, j0 - j1))
    i, j = np.linspace(i0, i1, num_pts), np.linspace(j0, j1, num_pts)
    return arr[i.astype(int), j.astype(int)]


class LinecutPlot:
    """A class for plotting 2D arrays and linecuts through them interactively."""

    def __init__(
        self,
        arr,
        x_size=None,
        y_size=None,
        fig=None,
        im_ax=None,
        linecut_ax=None,
        im_xlabel="x",
        im_ylabel="y",
        linecut_xlabel="d",
        linecut_ylabel="z",
        autoscale_x = False,
        autoscale_y = True
    ):
        self.arr = arr
        if im_ax is None or linecut_ax is None:
            self.fig, axs = plt.subplots(1, 2)
            self.im_ax, self.linecut_ax = axs

        self.im_ax.set_xlabel(im_xlabel)
        self.im_ax.set_ylabel(im_ylabel)
        self.linecut_ax.set_xlabel(linecut_xlabel)
        self.linecut_ax.set_ylabel(linecut_ylabel)

        self.x_size = x_size if x_size is not None else arr.shape[-1]
        self.y_size = y_size if y_size is not None else arr.shape[0]

        self.autoscale_x = autoscale_x
        self.autoscale_y = autoscale_y

        self.im = self.im_ax.imshow(arr, origin="lower", extent=(0, self.x_size, 0, self.y_size))

        self.p0 = np.array([0, 0])
        self.p1 = (
            (np.array(self.arr.shape[::-1]) - np.array([1, 1]))
            / np.array(self.arr.shape[::-1])
            * np.array([self.x_size, self.y_size])
        )

        self.click = None
        self.linecut_line = self.im_ax.plot(self.p0, self.p1, color="red")[0]
        self.fig.canvas.draw()
        self.linecut_plot = None

        def on_press(event):
            if event.inaxes == self.im_ax and event.button == 1:
                self.click = (event.xdata, event.ydata)
                self.p0 = np.array(self.click)
                self.linecut_line.set_xdata([event.xdata, event.xdata])
                self.linecut_line.set_ydata([event.ydata, event.ydata])

        def on_motion(event):
            if event.inaxes == self.im_ax and event.button == 1:
                self.linecut_line.set_xdata([self.click[0], event.xdata])
                self.linecut_line.set_ydata([self.click[1], event.ydata])
                self.p1 = np.array([event.xdata, event.ydata])
                update_linecut()
                self.fig.canvas.draw()

        def update_linecut():
            j0, i0 = (
                self.p0
                / np.array([self.x_size, self.y_size])
                * np.array(self.arr.shape[::-1])
            ).astype(int)
            j1, i1 = (
                self.p1
                / np.array([self.x_size, self.y_size])
                * np.array(self.arr.shape[::-1])
            ).astype(int)

            y_data = linecut(self.arr, i0, i1, j0, j1)
            x_data = np.linspace(0, np.hypot(*(self.p1 - self.p0)), len(y_data))

            if self.linecut_plot is not None:
                for line in self.linecut_plot:
                    line.remove()
            self.linecut_plot = self.linecut_ax.plot(x_data, y_data, color="#1f77b4")
            if self.autoscale_x:
                self.linecut_ax.set_xlim(0, x_data.max())
            if self.autoscale_y:
                range = abs(y_data.min() - y_data.max()) 
                self.linecut_ax.set_ylim(y_data.min() - 0.1*range, y_data.max() + 0.1*range)

        update_linecut()
        self.fig.canvas.mpl_connect("button_press_event", on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", on_motion)

