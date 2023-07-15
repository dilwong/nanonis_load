r"""
Loads Nanonis dI/dV spectroscopy files.

Usage:

spec = didv.spectrum(FILENAME) loads a single Bias Spectroscopy .dat file into a variable named spec.
spec.header contains ancillary information, spec.data is array of data.

specs = didv.batch_load(BASENAME) loads multiple Bias Spectroscopy .dat files into a list named specs.
didv.batch_load searches for all files with filename BASENAMEXXXXX.dat, where XXXXX is a five-digit number.

didv.plot(spectra, channel = NameOfChannel) plots sample bias vs. the channel NameOfChannel.
spectra is either a single spectrum loaded via didv.spectrum or a list of spectra loaded via didv.batch_load.

didv.waterfall(spectra_list, vertical_shift = NUMBER, reverse = False) makes a waterfall plot.
spectra_list is a series of either lists of didv.spectrum objects or BASENAME strings.

p = didv.colorplot(spectra_list) plots dI/dV(Vs, Vg).
This defaults to channel = 'Input 2 (V)' or 'Input 2 [AVG] (V)'. Use double_lockin = True to average with 'Input 3 (V)'.
p.drag_bar(direction = 'v' or 'h', locator = False).

"""

import numpy as np
import scipy.integrate
import pandas as pd
import time
import sys
import os
import ast
from .util import copy_text_to_clipboard
from .util import get_cmap_from_digitalizer_file

import glob
import re

from . import sxm

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

# BUG: interactive_colorplot.drag_bar appears to be broken in the MacOSX backend on matplotlib 3.5.1
# To fix this issue, either use a different backend (such as Qt5Agg) or downgrade to matplotlib 3.4.3
try:
    from . import interactive_colorplot
except (ImportError, ValueError):
    import interactive_colorplot

class spectrum():

    r"""
    didv.spectra(filename) loads one Nanonis spectrum file (extension .dat).
    
    Args:
        filename : str
            Name of the Nanonis spectrum file to be loaded.
    
    Attributes:
        header : dict
            Dictionary containing the header of the spectrum file.
        data : pandas.DataFrame
            Pandas DataFrame containing different measurement channels as columns.

    Methods:
        plot(channel = 'Input 2 (V)') : matplotlib.axes._subplots.AxesSubplot
            Plot 'Bias calc (V)' vs channel.
        to_clipboard(self, channel = None) : None
            Copy the data to clipboard.
    """

    def __init__(self, filename = None, attribute = None):

        #Read the header, build the header
        self.header = {}
        if filename is None:
            return
        self._filename = filename
        with open(filename,'r') as file_id:
            header_lines = 1
            while True:
                file_line=file_id.readline().strip()
                if '[DATA]' in file_line:
                    break
                header_lines += 1
                file_line=file_line.split('\t')
                if len(file_line) > 1:
                    self.header[file_line[0]]=file_line[1]
            self._fix_header()
            if attribute is not None:
                self.header['attribute'] = attribute

        self.data = pd.read_csv(filename, sep = '\t', header = header_lines, skip_blank_lines = False)

    @property
    def zero_bias_index(self):
        '''
        Returns the index of smallest bias magnitude.
        '''
        return np.argmin(np.abs(self.data['Bias calc (V)']))

    def get_integrated_data(self, channel, scale_factor : float=1.0) -> pd.DataFrame:
        '''
        Returns a channel of the data integrated from zero bias. I wouldn't use this if your spectrum doesn't
        cross zero bias. I'll probably add that later on when I feel less lazy :^)

        Parameters
        ----------
        channel : string
            The channel to integrate
        scale_factor : float, optional
            Scale factor applied to the data after integration. Default is 1.

        Returns
        -------
        data : pd.DataFrame
            A pandas DataFrame with 'Bias calc (V)' as the first column and 'Integrated {channel}' as the second column.
        '''
        integrated_data = np.empty(len(self.data))
        i0 = self.zero_bias_index
        for i in range(i0):
            integrated_data[i] = -scipy.integrate.trapezoid(self.data[channel][i:i0], self.data['Bias calc (V)'][i:i0])
        for i in range(i0, len(self.data)):
            integrated_data[i] = scipy.integrate.trapezoid(self.data[channel][i0:i], self.data['Bias calc (V)'][i0:i])
        data = pd.DataFrame({'Bias calc (V)' : self.data['Bias calc (V)'], f'Integrated {channel}' : integrated_data*scale_factor})
        return data

    def get_lockin_calibration_factor(self, lockin_channel : str='Input 2 (V)', method : str='derivative') -> float:
        '''
        Gets the conversion factor from lockin output to conductance. Has units of siemens/volt.
        Least squares is used to find the best fit scale factor.

        Parameters
        ----------
        lockin_channel : str, optional
            The name of the lockin channel. Default is 'Input 2 (V)'.
        method : str, optional
            The method to find the calibration factor with. If 'derivative', the derivative of the current will be
            used for least squares. If 'integral' or 'integrate', the integral of the lockin signal will be used for
            least squares. Default is 'derivative'.

        Returns
        -------
        calibration_factor : float
            The calibration factor in siemens/volt.
        '''
        current_channel_name = 'Current (A)'
        if current_channel_name not in self.data.keys():
            current_channel_name = 'Current [AVG] (A)'
        if method == 'derivative':
            return np.linalg.lstsq(self.data[lockin_channel].to_numpy()[:,np.newaxis], np.gradient(self.data[current_channel_name], self.data['Bias calc (V)']), rcond=None)[0]
        elif method == 'integral' or method == 'integrate':
            return np.linalg.lstsq(self.get_integrated_data(channel=lockin_channel)[f'Integrated {lockin_channel}'].to_numpy()[:,np.newaxis], self.data[current_channel_name], rcond=None)[0]


    def get_bias_offset(self):
        '''
        Returns the bias offset by finding the point with minimum current magnitude.
        '''
        return self.data['Bias calc (V)'][np.argmin(np.abs(self.data['Current (A)']))]

    def _fix_header(self):
        if 'X (m)' in self.header:
            self.header['x (nm)'] = float(self.header['X (m)'])*1e9
        if 'Y (m)' in self.header:
            self.header['y (nm)'] = float(self.header['Y (m)'])*1e9
        if 'Z (m)' in self.header:
            self.header['z (nm)'] = float(self.header['Z (m)'])*1e9
        if 'Gate Voltage (V)' in self.header:
            self.header['Gate (V)'] = float(self.header['Gate Voltage (V)'])
            self.gate = self.header['Gate (V)']
    
    def to_clipboard(self, channel = None):
        
        r'''
        Copies the data to clipboard.

        Args:
            channel : Optional[str]
                A string specifying which channel to copy to clipboard. If None, copy the entire DataFrame to clipboard.

        Returns:
            None
        '''
        
        if channel is None:
            self.data.to_clipboard()
        else:
            self.data[channel].to_clipboard(header = True)

    def plot(self, channel = 'Input 2 (V)', label = 'gate', multiply = 1, add = 0, plot_on_previous = False, ax=None, **kwargs):
        
        '''
        Plots 'Bias calc (V)' against the data in the column named channel.
            
        Args:
            channel : str (defaults to 'Input 2 (V)')
                A string specifying which channel to plot on the y-axis.
            label : str (defaults to 'gate')
                A string label for the data. If label = 'gate', then set the label to header['Gate (V)'] if it exists.
            multiply : float (defaults to 1)
                Scale the data by a multiplicative factor.
            add : float (defaults to 0)
                Shifts the data vertically by a constant.
        
        Returns:
            matplotlib.axes._subplots.AxesSubplot
        '''
        if ax is None:
            ax = plt.gca()

        if label == 'gate':
            try:
                legend_label = self.header['Gate (V)']
            except AttributeError:
                legend_label = None
        else:
            legend_label = label
        dat = self.data.copy()
        if multiply != 1:
            dat[channel] = dat[channel] * multiply
        if add != 0:
            dat[channel] = dat[channel] + add
        if plot_on_previous:
            return dat.plot(x = 'Bias calc (V)', y = channel, label = legend_label, ax = ax, **kwargs)
        return dat.plot(x = 'Bias calc (V)', y = channel, label = legend_label, **kwargs)

    def get_gap_bounds(self, mode : str='fwhm', channel : str='Input 2 (V)', prominence=0.001,
                                blur_width=0, second_derivative_threshold=0.001, 
                                min_search_window=(-0.01, 0.01), max_search_window=(-0.01, 0.01), 
                                correct_zero_bias=True, verbose=False):
        '''
        Documentation
        '''
        import scipy.signal
        import scipy.ndimage
        import scipy.interpolate

        x = self.data['Bias calc (V)']
        if blur_width == 0:
            y = self.data[channel].to_numpy()
        else:
            y = scipy.ndimage.gaussian_filter1d(self.data[channel], blur_width)

        zero_bias_index = np.argmin(np.abs(self.data['Bias calc (V)']))
        if np.gradient(np.gradient(y))[zero_bias_index] < second_derivative_threshold:
            return np.array([0, 0])

        if mode == 'fwhm' or mode == 'full width half max':
            max = np.amax(y[(x > max_search_window[0]) & (x < max_search_window[1])])
            min = np.amin(y[(x > min_search_window[0]) & (x < min_search_window[1])])
            interp = scipy.interpolate.UnivariateSpline(x, y - (max + min)/2, s=0)
            roots = interp.roots()
            if correct_zero_bias:
                roots -= x[np.argmin(np.abs(self.data['Current (A)']))]
            positive_roots = roots[roots >= 0]
            negative_roots = roots[roots < 0]
            if len(positive_roots) == 0 or len(negative_roots) == 0:
                return 0, 0
            return negative_roots[np.argmin(np.abs(negative_roots))], np.amin(positive_roots)
        elif mode == 'derivative':
            peaks, peak_properties = scipy.signal.find_peaks(np.gradient(y), prominence=prominence)
            dips, dip_properties = scipy.signal.find_peaks(-np.gradient(y), prominence=prominence)
            if verbose:
                print(f"peak_properties = {peak_properties}")
                print(f"dip_properties = {dip_properties}")
            if len(peaks) == 0 or len(dips) == 0:
                return np.array([0, 0])
            # Take the largest peak and dip
            return x[sorted([dips[np.argmax(dip_properties['prominences'])], peaks[np.argmax(peak_properties['prominences'])]])]
        elif mode == 'peaks' or mode == 'peak':
            peaks, properties = scipy.signal.find_peaks(y, prominence=prominence)
            sorted_indices = np.argsort(properties['prominences'])
            if verbose:
                print(f"properties = {properties}")
            if len(peaks) == 0:
                return np.array([0, 0])
            return x[sorted([peaks[sorted_indices[-2]], peaks[sorted_indices[-1]]])] # Indices of the two biggest peaks


    def get_gap_size(self, mode : str='derivative', channel : str='Input 2 (V)', prominence=0.001, blur_width=0, 
                        second_derivative_threshold=0.001, 
                        min_search_window=(-0.01, 0.01), max_search_window=(-0.01, 0.01), verbose=False) -> float:
        bounds = self.get_gap_bounds(mode=mode, channel=channel, prominence=prominence, 
                                            blur_width=blur_width, verbose=verbose, 
                                            second_derivative_threshold=second_derivative_threshold, 
                                            min_search_window=min_search_window, max_search_window=max_search_window)
        return np.abs(bounds[0] - bounds[1])

# Plot a spectrum
class plot():

    r"""
    Plots a list of didv.spectra. Each spectrum is plotted a separate line.
    
    Args:
        spectra : List[didv.spectra]
            List of didv.spectra to be plotted.
        channel: str (defaults to 'Input 2 (V)')
            The x-axis is 'Bias calc (V)'. The y-axis is channel.
        waterfall: float (default is 0)
            Offset each curve vertically by waterfall.
            To use waterfall, you must also specify increment.
        increment: Optional[float]
            The sign of increment determines whether waterfall shifts the spectra
            in ascending order or descending order.
        multiply : Optional[float]
            If not None, scale the data by a multiplicative factor.
        color : list of colors or a cmap-like object that matplotlib will accept
            Determines the color of each spectrum line.

    Attributes:
        fig : the matplotlib figure object
        ax : the matplotlib axes object

    Methods:
        xlim(x_min : float, x_max : float) : None
            Set the x-axis limits. x_min < x_max
        ylim(y_min : float, y_max : float) : None
            Set the y-axis limits. y_min < y_max
    """

    def __init__(self, spectra, channel = 'Input 2 (V)',  \
                                names = None, \
                                use_attributes = False, \
                                start = None, increment = None, \
                                waterfall = 0.0, \
                                dark = False, \
                                multiply = None, \
                                plot_on_previous = False, \
                                axes = None, \
                                color = None, \
                                bias_shift = 0, \
                                gate_as_index = True, \
                                legend = True, **kwargs):

        if waterfall != 0: # Does not work if spectra is a non-list iterator
            if dark:
                plt.style.use('dark_background')
                cmap = cm.get_cmap('RdYlBu')(np.linspace(0.1,0.8,len(spectra)))
            else:
                plt.style.use('default')
                cmap = cm.get_cmap('brg')(np.linspace(0,0.6,len(spectra)))
            cmap=cmap[::-1]

        if plot_on_previous:
            self.ax = plt.gca()
            self.fig = self.ax.figure
        elif axes is not None:
            self.ax = axes
            self.fig = self.ax.figure
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        name_list = names

        if (start is not None) and (increment is not None):
            name_list = np.arange(len(spectra)) * increment + start # Does not work if spectra is a non-list iterator
        try:
            spectra_iterator = iter(spectra)
        except TypeError:
            spectra_iterator = iter([spectra])
        for idx, spectrum_inst in enumerate(spectra_iterator):
            try:
                if use_attributes:
                    spectrum_label = str(spectrum_inst.header['attribute'])
                else:
                    spectrum_label = str(name_list[idx])
            except (TypeError, IndexError):
                spectrum_label = str(idx)
            if ('Gate (V)' in spectrum_inst.header) and (gate_as_index):
                    spectrum_label = str(spectrum_inst.header['Gate (V)'])
            spec_data = spectrum_inst.data.copy()
            if multiply is not None:
                spec_data[channel] = multiply * spec_data[channel]
            plot_args = dict(x = spec_data.columns[0], y = channel, ax = self.ax, legend = False, label = spectrum_label)
            if waterfall != 0:
                spec_data[channel] = spec_data[channel] + waterfall * idx * np.sign(increment) + 0.5 * (-np.sign(increment) + 1) * waterfall * (len(spectra) - 1)
                if color is None:
                    plot_args['color'] = tuple(cmap[idx])
                else:
                    if isinstance(color, list) and isinstance(spectra, list) and (len(color) == len(spectra)):
                        plot_args['color'] = color[idx]
                    else:
                        plot_args['color'] = color
            if bias_shift != 0:
                spec_data.iloc[:,0] -= bias_shift
            spec_data.plot(**plot_args)

        #Make a legend
        if legend:
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            if (waterfall == 0) or (np.sign(increment) < 0):
                self.legend = self.ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
                plot_lines = self.ax.get_lines()
            else:
                handles, labels = self.ax.get_legend_handles_labels()
                self.legend = self.ax.legend(handles[::-1], labels[::-1], loc = 'center left', bbox_to_anchor=(1, 0.5))
                plot_lines = self.ax.get_lines()
                plot_lines.reverse()
            legend_lines = self.legend.get_lines()
            line_map = dict()
            for legend_line, plot_line in zip(legend_lines,plot_lines):
                legend_line.set_picker(True)
                legend_line.set_pickradius(5)
                line_map[legend_line] = plot_line

            def pick_line(event):
                legend_line = event.artist
                plot_line = line_map[legend_line]
                visibility = not plot_line.get_visible()
                plot_line.set_visible(visibility)
                if visibility:
                    legend_line.set_alpha(1)
                else:
                    legend_line.set_alpha(0.2)
                self.fig.canvas.draw()

            self.pick_line = pick_line
            self.fig.canvas.mpl_connect('pick_event', pick_line)

        if dark:
            plt.style.use('default')

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

class colorplot(interactive_colorplot.colorplot):

    r"""
    Plots a channel('Bias calc (V)', 'Gate (V)') colorplot.
    didv.colorplot first takes an arbitrary number of non-keyword arguments.
    These non-keyword arguments are either strings or lists of didv.spectrum.
    If a string is 'BASENAME', search for all files in the current directory with a filename
    that matches 'BASENAMEXXXXX.dat', where XXXXX is a five-digit number.
    These files are loaded as didv.spectrum objects.
    Then didv.colorplot takes a set of keyword arguments defined below.
    
    Optional Keyword Arguments:
        channel : str
            The channel to be plotted on the color axis.
            If channel is not specified, channel will default to 'Input 2 (V)'.
            Also search for 'Input 2 [AVG] (V) in the pandas DataFrame for all
            didv.spectrum objects and rename that column to 'Input 2 (V)'.
        transform : function
            A function to element-wise transform the data.
            Alternative, specify 'diff' or 'derivative' to plot the derivative
            (via finite difference).
        diff_axis :
            If transform is 'diff' or 'derivative', diff_axis specifies the axis
            (bias or gate) to take the derivative along.
        dark : bool
            If true, plot with dark background.
        axes : matplotlib.axes._subplots.AxesSubplot
            Plot the colorplot on axes. If None, create a new axes object.
        over_iv: Tuple[float, float]
            If given a tuple of two floats (e.g. (XX, XX)), plot dI/dV/(I/V),
            where dI/dV is 'Input 2 (V)' and I/V is 'Current (A)' or 'Current [AVG] (A)'.
            The two floats vertically offset dI/dV and horizontally shift the bias
            in order to minimize the divide by zero issue.
        multiply : float
            If not None, scale the data by a multiplicative factor.
        ping_remove : float
            If not None, remove pings from data if there are 3 or more averages
            per spectrum. Deletes data that differs from the mean value by more than
            ping_remove standard deviations.
        rasterized : bool
            If True, then rasterize the colorplot. This is important when exporting the
            plot to a vector graphics format (e.g. SVG).
        colorbar : bool
            If False, do not display the colorbar.
        tilt_by_bias : bool
            If True, plot gate voltage - sample bias on the y-axis instead of just the
            gate voltage.

    Attributes:
        fig : matplotlib.figure.Figure
            The matplotlib Figure object that contains the colorplot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object that contains the colorplot.
        data : numpy.ndarray
            A numpy array with shape (# of biases, # of gates) that contains the numeric data.
        bias : numpy.ndarray
            A numpy array containing the bias values.
        gate : numpy.ndarray
            A numpy array containing the gate voltages. Also aliased as index_list.
        spectra_list : List[didv.spectrum]
            A list of all of the didv.spectrum objects.
        image_data_markers : list[list[float], list[float]]

    Methods:
        xlim(x_min : float, x_max : float) : None
            Set the x-axis limits. x_min < x_max
        ylim(y_min : float, y_max : float) : None
            Set the y-axis limits. y_min < y_max
        clim(c_min : float, c_max : float) : None
            Set the color axis limits. c_min < c_max
        refresh(wait_time : float) : None
            Reload the data every wait_time seconds.
        drag_bar(direction = 'horizontal', locator = False, axes = None, color = None) : interactive_colorplot.drag_bar
            Creates a "drag_bar" that allows the user to interact with the data.
            The drag_bar is a mouse-movable line on the colorplot that generates a plot of the line cut of the data.
            The drag_bar can also be moved by the keyboard arrow keys.
        colormap(cmap) : None
            Change the colormap to cmap, where cmap is an acceptable matplotlib colormap.
    """

    def __init__(self, *spectra_list, **kwargs):

        interactive_colorplot.colorplot.__init__(self)
        # Python 2 compatibility
        channel = kwargs['channel'] if ('channel' in kwargs) else None
        index_range = kwargs['index_range'] if ('index_range' in kwargs) else None
        index_label = kwargs['index_label'] if ('index_label' in kwargs) else 'Gate Voltage (V)'
        start = kwargs['start'] if ('start' in kwargs) else None
        increment = kwargs['increment'] if ('increment' in kwargs) else None
        transform = kwargs['transform'] if ('transform' in kwargs) else None
        diff_axis = kwargs['diff_axis'] if ('diff_axis' in kwargs) else 0
        dark = kwargs['dark'] if ('dark' in kwargs) else False
        axes = kwargs['axes'] if ('axes' in kwargs) else None
        over_iv = kwargs['over_iv'] if ('over_iv' in kwargs) else None
        multiply = kwargs['multiply'] if ('multiply' in kwargs) else None
        gate_as_index = kwargs['gate_as_index'] if ('gate_as_index' in kwargs) else True
        double_lockin = kwargs['double_lockin'] if ('double_lockin' in kwargs) else False
        ping_remove = kwargs['ping_remove'] if ('ping_remove' in kwargs) else False
        bias_shift = kwargs['bias_shift'] if ('bias_shift' in kwargs) else 0
        rasterized = kwargs['rasterized'] if ('rasterized' in kwargs) else False
        colorbar = kwargs['colorbar'] if ('colorbar' in kwargs) else True
        over_current = kwargs['over_current'] if ('over_current' in kwargs) else None
        post_transform = kwargs['post_transform'] if ('post_transform' in kwargs) else None
        running_index = kwargs['running_index'] if ('running_index' in kwargs) else False
        tilt_by_bias = kwargs['tilt_by_bias'] if ('tilt_by_bias' in kwargs) else False
        constraint = kwargs['constraint'] if ('constraint' in kwargs) else None
        cache = kwargs['cache'] if ('cache' in kwargs) else None
        yaxis = kwargs['yaxis'] if ('yaxis' in kwargs) else None

        self.arg_list = spectra_list
        self.state_for_update = {}
        self.initial_kwarg_state = kwargs
        self._bshift = bias_shift
        self._linecut_event_handlers = []

        self.spectra_list = parse_arguments(*spectra_list, cache = cache, constraint = constraint)
        if not self.spectra_list:
            return

        if channel is None:
            for spec in self.spectra_list:
                spec.data.rename(columns = {'Input 2 [AVG] (V)' : 'Input 2 (V)'}, inplace = True)
                if double_lockin:
                    spec.data.rename(columns = {'Input 3 [AVG] (V)' : 'Input 3 (V)'}, inplace = True)
                    spec.data['Input 2 (V)'] = (spec.data['Input 2 (V)'] + spec.data['Input 3 (V)']) * 0.5
                    self.state_for_update['double_lockin'] = True
            channel = 'Input 2 (V)'
        self.channel = channel
        if ping_remove:
            self.state_for_update['ping_remove'] = True
            for spec in self.spectra_list:
                std_ping_remove(spec, ping_remove)

        pcolor_cm = 'RdYlBu_r'

        if dark:
            plt.style.use('dark_background')

        try:
            bias_shift[0]
            bias = self.spectra_list[0].data.iloc[:,0].values
        except TypeError:
            try:
                if bias_shift == 0:
                    bias = self.spectra_list[0].data.iloc[:,0].values
                else:
                    bias = self.spectra_list[0].data.iloc[:,0].values - bias_shift
            except Exception:
                bias = self.spectra_list[0].data.iloc[:,0].values
        if transform is None:
            if multiply is None:
                self.data = pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values
            else:
                # TO DO: Implement keyword multiply that accepts an iterator of length len(self.spectra_list)
                self.data = pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values * multiply
            if over_iv is not None:
                self.current = pd.concat(((spec.data.get('Current (A)', 0) + spec.data.get('Current [AVG] (A)', 0)) for spec in self.spectra_list),axis=1).values - over_iv[0]
            self.bias = bias
        else:
            if (transform == 'diff') or (transform == 'derivative'):
                self.data = np.diff(pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values, axis = diff_axis)
                self.bias = bias[:-1]
                pcolor_cm = 'seismic'
            elif (transform == 'second_derivative'):
                self.data = np.diff(pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values, axis = diff_axis)
                self.data = np.diff(self.data, axis = diff_axis)
                bias = bias[:-2]
                self.bias = bias
                pcolor_cm = 'seismic'
            else:
                self.data = transform(pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values)
                self.bias = bias
        if axes is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = axes
            self.fig = axes.figure
        if running_index:
            self.index_list = np.array([int(s._filename.split('.')[0].split('_')[-1]) for s in self.spectra_list])
            self.state_for_update['running_index'] = True
        else:
            self.state_for_update['running_index'] = False
            if index_range is None:
                if start is None:
                    index_range = [-1000, 1000]
                else:
                    if increment is None:
                        index_range = [start, 1000]
                    else:
                        index_range = np.arange(len(self.spectra_list)) * increment + start # increment must be signed
            if len(index_range) == 2:
                self.index_list = np.linspace(index_range[0],index_range[1],len(self.spectra_list))
            if len(index_range) == len(self.spectra_list):
                self.index_list = np.array(index_range)
            if yaxis is None:
                if gate_as_index and (start is None) and (increment is None):
                    self.index_list = np.array([spec.gate for spec in self.spectra_list])
            else:
                self.state_for_update['yaxis'] = yaxis
                if isinstance(yaxis, str):
                    self.ylist = np.array([getattr(spec, yaxis) for spec in self.spectra_list])
                elif callable(yaxis):
                    self.ylist = np.array([yaxis(spec) for spec in self.spectra_list])
                else:
                    raise TypeError('yaxis is of unrecognized type')

        if over_iv is not None:
            self.data = self.data/self.current*(self.bias[:,np.newaxis] - over_iv[1])
        if over_current is not None: # Do not use over_iv and over_current at the same time!!
            self.current = pd.concat(((spec.data.get('Current (A)', 0) + spec.data.get('Current [AVG] (A)', 0)) for spec in self.spectra_list),axis=1).values - over_current
            self.data = self.data/self.current
        if post_transform is not None:
            self.data = post_transform(self.data)

        # TO DO: This program assumes all spectra have the same bias list.
        #        Implement the ability to handle spectra with different bias lists.
        deriv = (transform == 'diff') or (transform == 'derivative')
        try:
            len(bias_shift)
            xshift = True
        except TypeError:
            xshift = False
        self.state_for_update['tilt_by_bias'] = tilt_by_bias
        x, y = self.mesh(tilt = tilt_by_bias, xshift = xshift, derivative = deriv)

        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = pcolor_cm, rasterized = rasterized)
        self.original_cmap = self.pcolor.cmap
        if colorbar:
            self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
        self.ax.set_xlabel('Sample Bias (V)')
        self.ax.set_ylabel(index_label)
        self._x_axes_limits = list(self.ax.get_xlim())
        self._y_axes_limits = list(self.ax.get_ylim())

        if dark:
            plt.style.use('default')

        # Image data markers
        self.img_data_points = {} # Keys should be the tuple (V_s, V_g) and values should be filenames
        self.marker_annot = self.ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"),
                    zorder=2000)
        self.marker_annot.set_visible(False)
        self.img_data_scatter = self.ax.scatter([], [])
        plt.ion()
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.real_cscale = 3.5
        self.fft_cscale = 4
        
        #Set default colormap
        path_to_this_file = os.path.dirname(__file__)
        r_file = os.path.join(path_to_this_file, 'cmaps/w_r.csv')
        g_file = os.path.join(path_to_this_file, 'cmaps/w_g.csv')
        b_file = os.path.join(path_to_this_file, 'cmaps/w_b.csv')
        self.sxm_cmap = get_cmap_from_digitalizer_file(r_file, g_file, b_file)

    def get_header(self) -> dict:
        '''
        Returns the header of the first spectrum without the gate entry.
        '''
        header = self.spectra_list[0].header
        try:
            header.pop('Gate (V)')
        except:
            pass
        return header

    def get_spectra_gate_range(self, start_gate : float, end_gate : float) -> list[spectrum]:
        spectra = []
        for spectrum in self.spectra_list:
            if start_gate <= spectrum.gate <= end_gate:
                spectra.append(spectrum)
        
        return spectra

    def get_gap_size(self, start_gate : float=None, end_gate : float=None, 
                channel='Input 2 (V)', mode='derivative', prominence=0.001, blur_width=0, 
                second_derivative_threshold=0.001, 
                min_search_window=(-0.01, 0.01), max_search_window=(-0.01, 0.01), verbose=False) -> tuple[list[float], list[float]]:
        spectra = self.get_spectra_gate_range(start_gate, end_gate)
        gates = [spectrum.gate for spectrum in spectra]
        gap_sizes = [spectrum.get_gap_size(mode=mode, channel=channel, 
                                            prominence=prominence, blur_width=blur_width, 
                                            second_derivative_threshold=second_derivative_threshold, 
                                            min_search_window=min_search_window, max_search_window=max_search_window, 
                                            verbose=verbose) for spectrum in spectra]
        return gates, gap_sizes

    def get_spectrum_from_gate(self, gate : float) -> spectrum:
        closest_index = np.argmin(np.abs(self.gate - gate))
        return self.spectra_list[closest_index]

    @property
    def gate(self):
        return self.ylist

    @gate.setter
    def gate(self, value):
        self.ylist = value

    @property
    def index_list(self):
        return self.ylist

    @index_list.setter
    def index_list(self, value):
        self.ylist = value

    @property
    def bias(self):
        return self.xlist

    @bias.setter
    def bias(self, value):
        self.xlist = value
    
    @property
    def _bshift(self):
        return self._xshift

    @_bshift.setter
    def _bshift(self, value):
        self._xshift = value

    def load_data(self, cache = None):
        constraint = self.initial_kwarg_state['constraint'] if ('constraint' in self.initial_kwarg_state) else None
        cache = self.initial_kwarg_state['cache'] if ('cache' in self.initial_kwarg_state) else None
        self.spectra_list = parse_arguments(*self.arg_list, cache = cache, constraint = constraint)
        # Only works for Input 2 (V) or similar types of data
        for spec in self.spectra_list:
            spec.data.rename(columns = {'Input 2 [AVG] (V)' : 'Input 2 (V)'}, inplace = True)
            if 'double_lockin' in self.state_for_update:
                spec.data.rename(columns = {'Input 3 [AVG] (V)' : 'Input 3 (V)'}, inplace = True)
                spec.data['Input 2 (V)'] = (spec.data['Input 2 (V)'] + spec.data['Input 3 (V)']) * 0.5
        if 'ping_remove' in self.state_for_update:
            for spec in self.spectra_list:
                std_ping_remove(spec, self.initial_kwarg_state['ping_remove'])
        bias = self.spectra_list[0].data.iloc[:,0].values # No bias_shift
        self.bias = bias
        if self.state_for_update['running_index']:
            self.index_list = np.array([int(s._filename.split('.')[0].split('_')[-1]) for s in self.spectra_list])
        else:
            if 'yaxis' in self.state_for_update:
                yaxis = self.state_for_update['yaxis']
                if isinstance(yaxis, str):
                    self.ylist = np.array([getattr(spec, yaxis) for spec in self.spectra_list])
                elif callable(yaxis):
                    self.ylist = np.array([yaxis(spec) for spec in self.spectra_list])
            else:
                self.ylist = np.array([spec.gate for spec in self.spectra_list])
        self.data = pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values  # No transform, multiply, over_iv
    
    def update(self):
        colorbar = self.initial_kwarg_state['colorbar'] if ('colorbar' in self.initial_kwarg_state) else True
        super(colorplot, self).update(colorbar = colorbar, tilt = self.state_for_update['tilt_by_bias'])

    # Finds peaks and clusters them according to DBSCAN algorithm with parameters eps and min_samples
    # sigma controls the Gaussian smoothing of each index line before finding the local maxima
    #
    # TO DO: Implement a function that shifts data by np.convolve or np.correlate
    def peak_identify(self, sigma = 1.5, eps = 3, min_samples = 5):

        from sklearn.cluster import DBSCAN
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import argrelextrema

        lpts_bias = []
        lpts_gate = []
        pairs = []
        mod_pairs = []
        for idx in range(self.data.shape[1]):
            smoothed = gaussian_filter1d(self.data[:, idx], sigma)
            maxargs = argrelextrema(smoothed, np.greater)[0]
            for maxarg in maxargs:
                lpts_bias.append(self.bias[maxarg])
                lpts_gate.append(self.gate[idx])
                pairs.append([maxarg, idx])
                mod_pairs.append([1.0*maxarg, idx/1.0])
        clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(mod_pairs)
        label_list = clustering.labels_
        labels = set(label_list)
        for label in labels:
            points_list = []
            for index, point in enumerate(label_list):
                if point == label:
                    xb = self.bias[pairs[index][0]]
                    yg = self.gate[pairs[index][1]]
                    points_list.append([xb, yg])
                    x, y = zip(*points_list)
                    self.ax.scatter(x, y, s = .2)
        self._peak_pairs = pairs
        self._peak_cluster_labels = label_list

    def get_peak_cluster(self, bias, gate):
        gate_index = min(range(len(self.gate)), key = lambda idx: abs(self.gate[idx] - gate))
        bias_index = min(range(len(self.bias)), key = lambda idx: abs(self.bias[idx] - bias))
        nearest_pair_index = min(range(len(self._peak_pairs)), key = lambda idx: (bias_index - self._peak_pairs[idx][0])**2 + (gate_index - self._peak_pairs[idx][1])**2)
        cluster_index = self._peak_cluster_labels[nearest_pair_index]
        cluster_list = [idx for idx, val in enumerate(self._peak_cluster_labels) if val == cluster_index]
        relevant_pairs = [self._peak_pairs[idx] for idx in cluster_list]
        x, y = list(zip(*relevant_pairs))
        x = list(x)
        y = list(y)
        biases = self.bias[x]
        gates = self.gate[y]
        self.ax.scatter(biases, gates, marker = "*")
        return [x, y, biases, gates]

    def peak_follower(self, start_bias, start_gate, end_gate, cluster_filter = False):
        start_gate_index = min(range(len(self.gate)), key = lambda idx: abs(self.gate[idx] - start_gate))
        end_gate_index = min(range(len(self.gate)), key = lambda idx: abs(self.gate[idx] - end_gate))
        start_bias_index = min(range(len(self.bias)), key = lambda idx: abs(self.bias[idx] - start_bias))
        if start_gate_index < end_gate_index:
            gate_index_increment = 1
        elif start_gate_index > end_gate_index:
            gate_index_increment = -1
        else:
            print("Error: Start Gate == End Gate")
            return
        all_pairs = list(zip(self._peak_pairs, self._peak_cluster_labels))
        starting_pairs = [pair for pair in all_pairs if pair[0][1] == start_gate_index]
        index_into_pair = min(range(len(starting_pairs)), key = lambda idx: abs(starting_pairs[idx][0][0] - start_bias_index))
        curr_bias_index = starting_pairs[index_into_pair][0][0]
        curr_gate_index = start_gate_index
        if cluster_filter:
            cluster_index = starting_pairs[index_into_pair][1]
            relevant_pairs = [[pair[0][0], pair[0][1]] for pair in all_pairs if pair[1] == cluster_index]
        else:
            relevant_pairs = [[pair[0][0], pair[0][1]] for pair in all_pairs]
        b_list = [ curr_bias_index ]
        g_list = [ curr_gate_index ]
        while curr_gate_index != end_gate_index:
            curr_gate_index += gate_index_increment
            searched_biases = np.array([pair[0] for pair in relevant_pairs if pair[1] == curr_gate_index])
            if searched_biases.size == 0:
                #break
                continue
            index_into_bias = np.argmin(np.abs(searched_biases - curr_bias_index))
            curr_bias_index = searched_biases[index_into_bias]
            b_list.append(curr_bias_index)
            g_list.append(curr_gate_index)
        biases = self.bias[b_list]
        gates = self.gate[g_list]
        x = np.array(b_list)
        y = np.array(g_list)
        self.ax.scatter(biases, gates, marker = "*")
        return [x, y, biases, gates]

    def linecut(self, startPoint = None, endPoint = None, ninterp = 200, axes = None):

        from matplotlib.collections import LineCollection

        x, y = np.meshgrid(self.bias, self.index_list)
        x = x.T
        y = y.T
        try:
            bias_shift_len = len(self._bshift)
            if bias_shift_len == len(self.index_list):
                for idx, shift_val in enumerate(self._bshift):
                    x[:,idx] = x[:,idx] + shift_val
        except TypeError:
            pass
        x = x.flatten()
        y = y.flatten()
        data = self.data.flatten()

        fig = self.fig
        ax = self.ax

        first_pt = [None, None]
        last_pt = [None, None]
        line = [None, None]
        linecut_points = [] #linecut_points = set()
        try:
            self.linecut_points.append(linecut_points)
        except Exception: #AttributeError
            self.linecut_points = []
            self.linecut_points.append(linecut_points)

        nState = [0, None, None]
        pickrad = 1.0

        event_handlers = [None, None, None]
        self._linecut_event_handlers.append(event_handlers)

        if axes is None:
            linecut_fig = plt.figure()
            linecut_ax = linecut_fig.add_subplot(111)
        else:
            linecut_ax = axes
            linecut_fig = linecut_ax.figure

        def nearest_pt(bias_val, gate_val):
            # The Euclidean distance is used here.
            # Perhaps this is not desirable because bias and gate are on very different scales.
            # Consider normalizing each axis.
            dist_matrix = (x - bias_val)**2 + (y - gate_val)**2
            return np.argmin(dist_matrix)

        def build_line():
            line[0][0].set_xdata([first_pt[0], last_pt[0]])
            line[0][0].set_ydata([first_pt[1], last_pt[1]])
            fig.canvas.draw()
            x_list = np.linspace(first_pt[0], last_pt[0], ninterp)
            y_list = np.linspace(first_pt[1], last_pt[1], ninterp)
            linecut_points[:] = []
            for idx in range(ninterp):
                closest_pt_idx = nearest_pt(x_list[idx], y_list[idx])
                closest_pt = (x[closest_pt_idx], y[closest_pt_idx], data[closest_pt_idx])
                if closest_pt not in linecut_points:
                    linecut_points.append(closest_pt)
            linecut_points[:] = sorted(linecut_points, key = lambda itm: itm[0])
            points = np.array(linecut_points)
            bias = points[:,0]
            gate = points[:,1]
            conductance = points[:,2]
            points_tmp = points[:, [0, 2]].reshape(-1, 1, 2)
            segments = np.concatenate([points_tmp[:-1], points_tmp[1:]], axis = 1)
            norm = plt.Normalize(gate.min(), gate.max())
            lc = LineCollection(segments, cmap = 'plasma', norm = norm)
            lc.set_array(gate[:-1])
            lc.set_linewidth(2)
            if line[1] is not None:
                line[1].remove()
            line[1] = linecut_ax.add_collection(lc)
            bias_min = bias.min()
            bias_max = bias.max()
            conductance_min = conductance.min()
            conductance_max = conductance.max()
            if bias_min != bias_max:
                linecut_ax.set_xlim(bias_min, bias_max)
            if conductance_min != conductance_max:
                linecut_ax.set_ylim(conductance_min, conductance_max)
            linecut_fig.canvas.draw()

        def on_click(event):
            if (event.xdata is not None) and (event.ydata is not None):
                if first_pt[0] is None :
                    if startPoint is None:
                        nearest_idx = nearest_pt(event.xdata, event.ydata)
                    else:
                        nearest_idx = nearest_pt(startPoint[0], startPoint[1])
                    first_pt[0] = x[nearest_idx]
                    first_pt[1] = y[nearest_idx]
                    line[0] = ax.plot([first_pt[0], first_pt[0]], [first_pt[1], first_pt[1]], pickradius = pickrad)
                    event_handlers[1] = fig.canvas.mpl_connect('motion_notify_event', on_motion)
                    event_handlers[2] = fig.canvas.mpl_connect('button_release_event', on_release)
                else:
                    try:
                        first_dist = (event.xdata - first_pt[0])**2 + (event.ydata - first_pt[1])**2
                        last_dist = (event.xdata - last_pt[0])**2 + (event.ydata - last_pt[1])**2
                        distSens = 1
                        if first_dist < last_dist:
                            if first_dist < distSens:
                                nState[1] = 0
                                return
                        elif first_dist > last_dist:
                            if last_dist < distSens:
                                nState[1] = 1
                                return
                        else:
                            pass
                        contains, _ = line[0][0].contains(event)
                        if not contains:
                            return
                        else:
                            nState[1] = 2
                            nState[2] = [first_pt[0], first_pt[1], last_pt[0], last_pt[1], event.xdata, event.ydata]
                    except TypeError:
                        pass

        def on_motion(event):
            if event.inaxes != ax:
                return
            if nState[0] == 0:
                if endPoint is None:
                    nearest_idx = nearest_pt(event.xdata, event.ydata)
                else:
                    nearest_idx = nearest_pt(endPoint[0], endPoint[1])
                last_pt[0] = x[nearest_idx]
                last_pt[1] = y[nearest_idx]
                build_line()
            else:
                nearest_idx = nearest_pt(event.xdata, event.ydata)
                if nState[1] == 0:
                    first_pt[0] = x[nearest_idx]
                    first_pt[1] = y[nearest_idx]
                    build_line()
                elif nState[1] == 1:
                    last_pt[0] = x[nearest_idx]
                    last_pt[1] = y[nearest_idx]
                    build_line()
                elif nState[1] == 2:
                    vec_x = event.xdata - nState[2][4]
                    vec_y = event.ydata - nState[2][5]
                    first_pt[0] = nState[2][0] + vec_x
                    first_pt[1] = nState[2][1] + vec_y
                    last_pt[0] = nState[2][2] + vec_x
                    last_pt[1] = nState[2][3] + vec_y
                    build_line()
                else:
                    pass

        def on_release(event):
            nState[0] += 1
            nState[1] = None
            nState[2] = None

        if (startPoint is not None) and (endPoint is not None):
            nearest_idx = nearest_pt(startPoint[0], startPoint[1])
            first_pt[0] = x[nearest_idx]
            first_pt[1] = y[nearest_idx]
            nearest_idx = nearest_pt(endPoint[0], endPoint[1])
            last_pt[0] = x[nearest_idx]
            last_pt[1] = y[nearest_idx]
            line[0] = ax.plot([first_pt[0], last_pt[0]], [first_pt[1], last_pt[1]], pickradius = pickrad)
            build_line()
            event_handlers[1] = fig.canvas.mpl_connect('motion_notify_event', on_motion)
            event_handlers[2] = fig.canvas.mpl_connect('button_release_event', on_release)
            nState[0] += 1

        event_handlers[0] = fig.canvas.mpl_connect('button_press_event', on_click)

    def to_html(self, filename, library = 'altair'):

        r'''
        Save the colorplot in an interactive HTML file, using Vega/Vega-Lite.

        Note that performance may be poor for large datasets for certain browsers.

        Arguments:
            filename : str
                Output filename
            library : str (defaults to 'altair')
                Plotting library used to generate HTML file.
                So far, only 'altair' and 'hvplot' are implemented.
        '''

        if filename[-5:] != '.html':
            filename = filename + '.html'
        
        if library.lower() == 'altair':
            import altair as alt
            from bs4 import BeautifulSoup

            x, y = self.mesh()
            x = np.repeat(x, 2, axis = 0)[1:-1,:-1].reshape(self.data.shape[0], 2, -1).transpose((1,0,2)).reshape(2,-1)
            y = np.repeat(y, 2, axis = 1)[:-1,1:-1].reshape(-1, 2).T

            epsilonX = 1e-3
            epsilonY = 1e-3

            df = pd.DataFrame({
                'didv' : self.data.ravel(),
                'x' : x[0, :] - epsilonX,
                'x2' : x[1, :] + epsilonX,
                'y' : y[0, :] - epsilonX,
                'y2' : y[1, :] + epsilonY
            })

            mindidv = np.min(df['didv'])
            maxdidv = np.max(df['didv'])
            nRes = 100
            didvresolution = (maxdidv - mindidv) / nRes
            # selectorMin = alt.selection_single(
            #     name = "minSelect",
            #     fields = ['min'],
            #     bind = alt.binding_range(min = mindidv, max = maxdidv, step = didvresolution, name = 'min'), init = {'min': mindidv}
            # )
            selectorMin = alt.selection_single(
                name = "minSelect",
                fields = ['minimum'],
                bind = alt.binding_range(min = mindidv, max = maxdidv, step = didvresolution, name = 'Color Minimum'), init = {'minimum': mindidv}
            )
            selectorMax = alt.selection_single(
                name = "maxSelect",
                fields = ['maximum'],
                bind = alt.binding_range(min = mindidv, max = maxdidv, step = didvresolution, name = 'Color Maximum'), init = {'maximum': maxdidv}
            )

            base = alt.Chart(df)

            minX = np.min(x)
            minY = np.min(y)
            maxX = np.max(x)
            maxY = np.max(y)

            try:
                colorScheme = alt.Color('didv_clamped', type = 'quantitative', scale = alt.Scale(scheme = 'redyellowblue', reverse = True), title = 'dI/dV')
            except:
                colorScheme = alt.Color('didv_clamped', type = 'quantitative', sort = 'descending', scale = alt.Scale(scheme = 'redyellowblue'), title = 'dI/dV')

            cplot = base.properties(
                height = 500,
                width = 300
            ).mark_rect().transform_calculate(
                didv_clamped = 'clamp(datum.didv, minSelect.minimum, maxSelect.maximum)'
            ).encode(
                x = alt.X(
                    'x',
                    type = 'quantitative',
                    scale = alt.Scale(domain = (minX, maxX), nice = False),
                    title = 'Sample Bias (V)',
                    axis = alt.Axis(grid = False)
                ),
                y = alt.Y(
                    'y',
                    type = 'quantitative',
                    scale = alt.Scale(domain = (minY, maxY), nice = False),
                    title = 'Gate Voltage (V)',
                    axis = alt.Axis(grid = False)
                ),
                x2 = alt.X2('x2'),
                y2 = alt.Y2('y2'),
                fill = colorScheme,
                #stroke = colorScheme,
            ).interactive().add_selection(selectorMin).add_selection(selectorMax)

            mouseSelection = alt.selection_single(
                name = 'mouseSelect', fields = ['y'], nearest = True, on = 'mouseover', empty = 'none', clear = 'mouseout'
            )

            rule = base.properties(
                height = 500,
                width = 300
            ).mark_rule().transform_calculate(
                gate = '(datum.y + datum.y2) / 2.0'
            ).encode(
                y = 'y:Q',
                opacity = alt.condition(mouseSelection, alt.value(0.5), alt.value(0)),
                tooltip=[alt.Tooltip('gate', type='quantitative')]
            ).add_selection(mouseSelection)

            lineplot = base.properties(
                height = 200,
                width = 300
            ).transform_filter('datum.y == mouseSelect.y').mark_line().transform_calculate(
                centered_x = (alt.datum.x2 + alt.datum.x) / 2.0
            ).encode(
                x = alt.X('centered_x:Q', title = 'Sample Bias (V)'),
                y = alt.Y('didv:Q', title = 'dI/dV')
            )

            background = alt.Chart(
                {"values": [{"x": minX, "y": mindidv}, {"x": maxX, "y": maxdidv}]}
            ).transform_filter(
                '!isValid(mouseSelect.y)'
            ).mark_point(opacity = 0).encode(
                x='x:Q',
                y='y:Q'
            ).properties(
                height = 200,
                width = 300
            ).add_selection(mouseSelection)

            alt.vconcat((cplot + rule), (lineplot + background)).save(filename)

            with open(filename, 'r') as f:
                soup = BeautifulSoup(f.read(), features = 'html.parser')
            soup.find('style').append(
                """form.vega-bindings {
                position: absolute;
                top: 0px;
                }
                """ +
                # """
                # div.chart-wrapper {
                #   margin-top: 1.5cm;
                # }
                # """
                """
                canvas {
                margin-top: 1.5cm;
                }
                """
            )
            with open(filename, 'w') as f:
                f.write(str(soup))
        elif (library.lower() == 'hvplot') or (library.lower() == 'holoviews'):
            if sys.version_info.major == 2:
                raise NotImplementedError(r"Not compatible with Python 2.")
            elif sys.version_info.major == 3:
                if sys.version_info.minor < 6:
                    raise NotImplementedError(r"Python version must be at least 3.6.")
            import holoviews as hv
            import hvplot
            hv.extension('bokeh')
            qmesh=hv.QuadMesh((self.xlist, self.ylist, self.data.T))
            qmesh.opts(
                clim = (np.min(self.data), np.max(self.data)),
                frame_height = 800,
                frame_width = 600,
                cmap = 'RdYlBu_r',
                colorbar = True,
                xlabel = 'Sample Bias (V)',
                ylabel = 'Gate Voltage (V)')
            hvplot.save(qmesh, filename)
        elif library.lower() == 'bokeh':
            raise NotImplementedError(r"Use library = 'hvplot' option instead.")
        elif library.lower() == 'plotly':
            raise NotImplementedError(r"plotly backend not yet implemented.")
        else:
            raise NotImplementedError(r"Unknown plotting library.")

    def bias_and_gate_in_range(self, sample_bias, gate_voltage, tolerance=100):
        '''
        Returns True if sample_bias and gate_voltage are within the bounds of the gate sweep
        '''
        gate_min = np.amin(self.ylist)
        gate_max = np.amax(self.ylist)
        gate_tolerance = np.abs(gate_max - gate_min) / tolerance
        bias_min = np.amin(self.xlist)
        bias_max = np.amax(self.xlist)
        bias_tolerance = np.abs(bias_max - bias_min) / tolerance

        return (sample_bias > bias_min - bias_tolerance) & (sample_bias < bias_max + bias_tolerance) & (gate_voltage > gate_min - gate_tolerance) & (gate_voltage < gate_max + gate_tolerance)

    def add_img_data_marker(self, filename):
        '''
        Loads an sxm file "filename", grabs its sample bias and gate voltage, and adds 
        it to self.image_data_marker

        If the gate voltage is not stored in the file, nothing is added to 
        self.image_data_marker.
        '''
        header = sxm.sxm_header(filename)

        # Try getting gate voltage
        try:
            gate_voltage = round(float(header[':Ext. VI 1>Gate voltage (V):'][0]), 2)
        except:
            print("Warning: " + filename + " does not have the gate voltage stored in it")
            return

        if 'multipass_biases' in header.keys():
            sample_biases = header['multipass_biases']
            # Only add the marker if it's within the bounds of the spectrum
            for sample_bias in sample_biases:
                self.img_data_points[(round(sample_bias, 5), gate_voltage)] = filename
        else:
            sample_bias = float(header[':BIAS:'][0]) # If this throws an exception, then the header is probably fucked up
            self.img_data_points[(round(sample_bias, 5), gate_voltage)] = filename

    def add_img_data_marker_manual(self, filename, sample_bias, gate_voltage):
        '''
        Manually adds a sample bias and gate voltage to the image data markers.
        '''
        if (sample_bias > np.amin(self.xlist)) & (sample_bias < np.amax(self.xlist)) & (gate_voltage > np.amin(self.ylist)) & (gate_voltage < np.amax(self.ylist)):
            self.img_data_points['filename'].append(filename)
            self.img_data_points['V_s'].append(sample_bias)
            self.img_data_points['V_g'].append(gate_voltage)

    # Should this clear add_img_data_marker first to avoid duplicates?
    #
    # Sometimes multiple images have the same (V_s, V_g) because the data was retaken
    # or because some images are incomplete. How should this be handled?
    # Fixed using dicts
    def auto_add_img_data_markers(self, basename = None, start = 0, end = 99999, index_list = None):
        '''
        Automatically loops over all files in current working directory and adds image data markers using them.
        '''
        for filename in os.listdir('.'):
            if filename.endswith('.sxm'):
                if basename is None:
                    self.add_img_data_marker(filename)
                else:
                    regex = re.compile(basename + '_?([0-9]+).sxm')
                    match = regex.match(filename)
                    if (match is not None) and (start <= int(match.group(1)) <= end):
                        if index_list is not None:
                            if int(match.group(1)) not in index_list: # Should check if index_list is a list, tuple, set, etc...
                                continue
                        self.add_img_data_marker(filename)
        self.plot_img_data_markers()
        self.axes_reset()

    def auto_add_image_data_markers(self, *args, **kwargs):
        '''
        For name convenience
        '''
        self.auto_add_img_data_markers(*args, **kwargs)

    def add_img_data_markers_by_time(self, start_time, end_time):
        '''
        Adds files in current working directory if their last modified time is between 
        start_time and end_time.
        '''


    def plot_img_data_markers(self, s=100, color=(0, 0, 0, 1), zorder=1000, **kwargs):
        '''
        Plots the image data markers.
        '''
        sample_biases = [bias for bias, _ in self.img_data_points.keys()]
        gate_voltages = [gate for _, gate in self.img_data_points.keys()]
        self.img_data_scatter = self.ax.scatter(sample_biases, gate_voltages, s=s, color=color, zorder=zorder, **kwargs)
        self.ax.zorder = 100

    def plot_image_data_markers(self, *args, **kwargs):
        self.plot_img_data_markers(*args, **kwargs)

    def update_annotation(self, ind):
        index = ind["ind"][0]
        pos = self.img_data_scatter.get_offsets()[index]
        self.marker_annot.xy = pos
        text = self.img_data_points[list(self.img_data_points.keys())[index]] + '\nVs = ' + str(list(self.img_data_points.keys())[index][0]*1000) + ' mV' + '\nVg = ' + str(list(self.img_data_points.keys())[index][1]) + ' V'
        self.marker_annot.set_text(text)
        self.marker_annot.get_bbox_patch().set_facecolor((0, 1, 0, 1))
        self.marker_annot.get_bbox_patch().set_alpha(0.6)

    def get_onenote_info_string(self) -> str:
        '''
        Returns an info string to paste into your notes
        '''
        filename = self.arg_list[0]
        
        header = self.get_header()

        gate_range = f"{(round(np.amin(self.ylist), 2), round(np.amax(self.ylist), 2))} V"
        gate_increment = f"{round(np.abs(self.ylist[1] - self.ylist[0]), 2)} V"

        bias_range_float = (np.amin(self.xlist), np.amax(self.xlist))
        if np.abs(bias_range_float[0]) < 1 or np.abs(bias_range_float[1]) < 1:
            bias_range = f"{(round(bias_range_float[0]*1000, 2), round(bias_range_float[1]*1000, 2))} mV"
        else:
            bias_range = f"{(round(bias_range_float[0], 2), round(bias_range_float[1], 2))} V"

        try:
            setpoint_bias_float = float(header['Bias (V)'])
            if np.abs(setpoint_bias_float) < 1:
                setpoint_bias = f"{round(setpoint_bias_float * 1000, 2)} mV"
            else:
                setpoint_bias = f"{round(setpoint_bias_float, 2)} V"
        except:
            setpoint_bias = "Not recorded"
        try:
            setpoint_current_float = float(header['Setpoint current (pA)'])
            if np.abs(setpoint_current_float) < 1e3:
                setpoint_current = f"{round(setpoint_current_float, 2)} pA"
            elif 1e3 < np.abs(setpoint_current_float) < 1e6:
                setpoint_current = f"{round(setpoint_current_float / 1e3, 2)} nA"
            else:
                setpoint_current = f"{round(setpoint_current_float / 1e6, 2)} uA"
        except:
            setpoint_current = "Not recorded"
        try:
            lockin_amplitude_float = float(header['Lockin Amplitude'])
            bias_calibration_factor = float(header['Bias Calibration Factor'])
            if 0.3 < bias_calibration_factor < 0.7:
                lockin_amplitude = f"{lockin_amplitude_float / 100 * 1000} mV"
            elif bias_calibration_factor < 0.3:
                lockin_amplitude = f"{lockin_amplitude_float} mV"
            else:
                lockin_amplitude = "Weird bias calibration value?"
        except:
            lockin_amplitude = "Not recorded"
        try:
            lockin_frequency = f"{header['Lockin Frequency']} Hz"
        except:
            lockin_frequency = "Not recorded"
        try:
            lockin_sensitivity = header['Lockin Sensitivity']
        except:
            lockin_sensitivity = "Not recorded"
        try: 
            lockin_time_constant = header['Lockin Time Constant']
        except:
            lockin_time_constant = "Not recorded"
        
        return f"{filename}\n\nGate range = {gate_range}\nGate increment = {gate_increment}\nBias range = {bias_range}\nSetpoint bias = {setpoint_bias}\nCurrent = {setpoint_current}\nLockin amplitude = {lockin_amplitude}\nLockin frequency = {lockin_frequency}\nLockin sensitivity = {lockin_sensitivity}\nLockin time constant = {lockin_time_constant}"
        
    def copy_onenote_info_string(self):
        '''
        Copies the string returned by get_onenote_info_string() onto the clipboard.
        '''
        copy_text_to_clipboard(self.get_onenote_info_string())

    def on_hover(self, event):
        '''
        Handles mouse hover events over scatter plot points.
        '''
        visible = self.marker_annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.img_data_scatter.contains(event)
            if cont:
                self.update_annotation(ind)
                self.marker_annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if visible:
                    self.marker_annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def on_click(self, event):
        '''
        Handles click events (namely the image data markers)
        '''
        if event.inaxes == self.ax:
            cont, ind = self.img_data_scatter.contains(event)
            if cont:
                if sys.version_info[1] < 7:
                    print("Error: Python version is below 3.7.")
                    return
                index = ind["ind"][0]
                image_sxm = sxm.sxm(self.img_data_points[list(self.img_data_points.keys())[index]])
                x_range = image_sxm.header['x_range (nm)']
                y_range = image_sxm.header['y_range (nm)']
                x_pixels = image_sxm.header['x_pixels']
                y_pixels = image_sxm.header['y_pixels']
                try:
                    multipass_index = np.argmin(np.abs(image_sxm.header["multipass_biases"] - event.xdata))
                    data = image_sxm.process_data(image_sxm.data["Z (m)"][multipass_index], process='subtract plane')
                except KeyError:
                    data = image_sxm.process_data(image_sxm.data["Z (m)"][0], process='subtract plane')

                data_fft = np.abs(np.fft.fftshift(np.fft.fft2(data)))
                
                data_vmin = np.mean(data) - self.real_cscale*np.std(data)
                data_vmax = np.mean(data) + self.real_cscale*np.std(data)
                
                fig, ax = plt.subplots(1, 2)
                
                copy_text_to_clipboard(image_sxm.get_onenote_info_string())

                ax[0].imshow(data,
                            cmap=self.sxm_cmap,
                            vmin=data_vmin,
                            vmax=data_vmax,
                            origin='lower',
                            extent=(0, x_range, 0, y_range)
                )
                ax[1].imshow(data_fft,
                            cmap=self.sxm_cmap,
                            vmin=0,
                            vmax=self.fft_cscale*np.std(data_fft),
                            origin='lower',
                            extent = (-np.pi/(x_range/x_pixels), np.pi/(x_range/x_pixels), -np.pi/(y_range/y_pixels), np.pi/(y_range/y_pixels))
                )
                fig.canvas.draw()
                plt.show()
                


        

def batch_load(basename, file_range = None, attribute_list = None, cache = None, constraint = None):

    if cache is not None:
        spectrum_array = cache
        file_list = [c._filename for c in cache]
    else:
        file_list = []
        spectrum_array = []
    if ('.h5' in basename) or ('.hdf5' in basename):
        cachedNames = set(file_list) if cache else None
        return HDF5Tospecs(basename, cachedNames = cachedNames, returnNames = True, constraint = constraint)
    else:
        if file_range is None:
            file_range = range(9999)
        file_string = basename + '*.dat'
        file_exist = glob.glob(file_string)
        if not file_exist:
            return (spectrum_array, file_list)
        for idx, file_number in enumerate(file_range):
            filename = basename + '%0*d' % (5, file_number) + '.dat'
            if filename in file_exist:
                if cache is not None:
                    if filename in file_list: # Maybe use a set
                        continue
                try:
                    spectrum_inst = spectrum(filename)
                    file_list.append(filename)
                    if attribute_list:
                        spectrum_inst.header['attribute'] = attribute_list[idx]
                    spectrum_array.append(spectrum_inst)
                except IOError:
                    continue
        return (spectrum_array, file_list)

def parse_arguments(*spectra_arguments, **kwargs) -> list[spectrum]:

    cache = kwargs['cache'] if ('cache' in kwargs) else None
    constraint = kwargs['constraint'] if ('constraint' in kwargs) else None

    if cache is None:
        spectra = []
    else:
        spectra = cache
    for arg in spectra_arguments:
        if type(arg) == str:
            s, f = batch_load(arg, cache = cache, constraint = constraint)
            if f == []:
                s, f = batch_load(arg + '.h5', cache = cache, constraint = constraint)
                if f == []:
                    print('WARNING: NO FILES WITH BASENAME ' + arg)
            # monotonic keyword depreciated
            spectra.extend(s)
        elif type(arg) == list:
            spectra.extend(arg) # Does not check if list contains only didv.spectrum
        elif type(arg) == spectrum:
            spectra.append(arg)
        else:
            print('INCORRECT TYPE ERROR IN didv.parse_arguments')
    if not spectra:
        print('ERROR: NO FILES!')
    return spectra

def specsToHDF5(spectrumList, filename):
    r'''
    Saves a list of didv.spectrum objects as an HDF5 file.
    Does not overwrite an HDF5 if it already exists. Instead, append data to existing file.

    Args:
        filename : str
            A string specifying the filename of the HDF5 file.
    '''
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category = FutureWarning)
        import h5py
    if sys.version_info.major == 2:
        FileInUseError = IOError
    elif sys.version_info.major == 3:
        FileInUseError = BlockingIOError
    while True:
        try:
            with h5py.File(filename, 'a') as f:
                if 'data' not in f.keys():
                    f.create_group('data') # Unfortunately, track_order not available in older versions of h5py
                for spec in spectrumList:
                    f['data'].create_dataset(spec._filename, data = spec.data.values)
                    for key, item in spec.header.items():
                        f['data'][spec._filename].attrs[key] = item
                    f['data'][spec._filename].attrs['channels'] = '||'.join(spec.data.columns)
                size = len(f['data'])
                f.attrs['size'] = size
                while str(size) in f['data'].keys():
                    size += 1
                f.attrs['index'] = size
        except FileInUseError:
            time.sleep(0.1)
            continue
        break

def HDF5Tospecs(filename, cachedNames = None, returnNames = False, constraint = None):
    r'''
    Reads an HDF5 file containing Bias Spectroscopy data, and returns a list of didv.spectrum objects.

    Args:
        filename : str
            A string specifying the filename of the HDF5 file.
        constraint : function(h5py.AttributeManager) -> bool
            A function that filters which datasets are returned from the HDF5 file.
            The function takes as input an 'attrs' of a HDF5 dataset.
            The function returns True if the dataset should be included in the returned List[didv.spectrum], 
            and false otherwise.
    '''
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category = FutureWarning)
        import h5py
    if sys.version_info.major == 2:
        FileInUseError = IOError
    elif sys.version_info.major == 3:
        FileInUseError = BlockingIOError
    specList = []
    fileList = []
    while True:
        try:
            if not os.path.isfile(filename):
                break
            with h5py.File(filename, 'r') as f:
                for name, dataset in f['data'].items():
                    if (cachedNames is not None) and (name in cachedNames):
                        continue
                    if (constraint is not None) and (not constraint(dataset.attrs)):
                        continue
                    spec = spectrum()
                    for key, item in dataset.attrs.items():
                        spec.header[key] = item
                    spec.data = pd.DataFrame(dataset[()], columns = dataset.attrs['channels'].split('||'))
                    spec._filename = name
                    spec._fix_header()
                    specList.append(spec)
                    fileList.append(name)
        except FileInUseError: # If the file is in use by another process, wait.
            time.sleep(0.1)
            continue
        break
    if returnNames:
        return (specList, fileList)
    else:
        return specList

def datFilesToHDF5():
    r'''
    Convert .dat files in current working directory to .h5 HDF5 files.

    .dat files are grouped according to their basenames, i.e. data from .dat files
    named 'YYYYYYXXXXX.dat' (where XXXXX is any 5 digit number) are put into an HDF5
    file named 'YYYYYY.h5'.

    This function skips any files with basename 'Bias-Spectroscopy'.
    '''
    import re
    allFiles = glob.glob('*.dat')
    regex = re.compile(r'^(.*)\d{5}.dat')
    allBasenames = set()
    for file in allFiles:
        match = regex.match(file)
        if match is not None:
            basename = match.group(1)
            if basename != 'Bias-Spectroscopy': # Will still accept 'Bias-Spectroscopy_'
                allBasenames.add(basename)
    for basename in allBasenames:
        specs, _ = batch_load(basename)
        specsToHDF5(specs, basename + '.h5')

def quick_colorplot(*args, **kwargs):

    return colorplot(*args, **kwargs)

class transform_colorplot(interactive_colorplot.colorplot):
    
    r"""
    Plots a transformed colorplot based on other didv.colorplot objects.
    The first argument to didv.transform_colorplot is a function that takes n floats
    and returns a single float.
    Then the next n arguments to didv.transform_colorplot are didv.colorplot objects.
    The data from the n didv.colorplot objects are element-wise fed into the function,
    and the result is plotted as a colorplot.
    didv.transform_colorplot will only plot data for gate voltages that exist in all of
    the didv.colorplot objects. Gate voltages that do not exist in all didv.colorplot
    objects will be skipped.

    Attributes:
        fig : matplotlib.figure.Figure
            The matplotlib Figure object that contains the colorplot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object that contains the colorplot.
        data : numpy.ndarray
            A numpy array with shape (# of biases, # of gates) that contains the numeric data.
        bias : numpy.ndarray
            A numpy array containing the bias values.
        gate : numpy.ndarray
            A numpy array containing the gate voltages. Also aliased as index_list.

    Methods:
        xlim(x_min : float, x_max : float) : None
            Set the x-axis limits. x_min < x_max
        ylim(y_min : float, y_max : float) : None
            Set the y-axis limits. y_min < y_max
        clim(c_min : float, c_max : float) : None
            Set the color axis limits. c_min < c_max
        refresh(wait_time : float) : None
            Reload the data every wait_time seconds.
        drag_bar(direction = 'horizontal', locator = False, axes = None, color = None) : interactive_colorplot.drag_bar
            Creates a "drag_bar" that allows the user to interact with the data.
            The drag_bar is a mouse-movable line on the colorplot that generates a plot of the line cut of the data.
            The drag_bar can also be moved by the keyboard arrow keys.
        colormap(cmap) : None
            Change the colormap to cmap, where cmap is an acceptable matplotlib colormap.
    """

    def __init__(self, *args, **kwargs):

        interactive_colorplot.colorplot.__init__(self)
        self.kwargs = kwargs
        pcolor_cm = kwargs['cmap'] if ('cmap' in kwargs) else 'RdYlBu_r'
        rasterized = kwargs['rasterized'] if ('rasterized' in kwargs) else False

        if len(args) < 2:
            raise TypeError("didv.transform_colorplot takes at least two arguments")
        if not callable(args[0]):
            raise TypeError("First argument is not a callable function")
        self.func = args[0]
        self._cplots = args[1:]
        self.xlist = self._cplots[0].bias # Assumes bias for all self._cplots are the same
        self.load_data()
        pseudocoordX, pseudocoordY = self.mesh()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.pcolor = self.ax.pcolormesh(pseudocoordX, pseudocoordY, self.data, cmap = pcolor_cm, rasterized = rasterized)
        self.original_cmap = self.pcolor.cmap
        self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
        self.ax.set_xlabel('Sample Bias (V)')
        self.ax.set_ylabel('Gate (V)')
        self._x_axes_limits = list(self.ax.get_xlim())
        self._y_axes_limits = list(self.ax.get_ylim())

    def load_data(self):

        index_list_tmp = []
        data_tmp = []
        for y_val in self._cplots[0].ylist:
            curr_row = []
            try:
                for cplot in self._cplots:
                    new_ind = np.where(cplot.ylist == y_val)[0][-1]
                    curr_row.append(cplot.data[:, new_ind])
            except IndexError:
                continue
            data_tmp.append(self.func(*curr_row))
            index_list_tmp.append(y_val)
        self.ylist = np.array(index_list_tmp)
        self.data = np.array(data_tmp).T

    @property
    def gate(self):
        return self.ylist

    @gate.setter
    def gate(self, value):
        self.ylist = value

    @property
    def index_list(self):
        return self.ylist

    @index_list.setter
    def index_list(self, value):
        self.ylist = value

    @property
    def bias(self):
        return self.xlist

    @bias.setter
    def bias(self, value):
        self.xlist = value

class multi_colorplot():

    r"""
    Plots multiple didv.colorplots together on the same Figure.
    drag_bar objects on didv.multi_colorplot are jointly movable.
    
    Args:
        n : int
            The number of colorplots to plot side-by-side.

    Attributes:
        fig : matplotlib.figure.Figure
            The matplotlib Figure object that contains the colorplot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object that contains the colorplot.
        colorplots : List[didv.colorplot]
            A list of the didv.colorplot objects.
        drag_bars : List[interactive_colorplot.drag_bar]
            A list of the interactive_colorplot.drag_bar objects.

    Methods:
        add_data(*args, **kwargs) : None
            Adds data to the multi_colorplot by calling didv.colorplot(*args, **kwargs).
        xlim(x_min : float, x_max : float) : None
            Set the x-axis limits. x_min < x_max
        ylim(y_min : float, y_max : float) : None
            Set the y-axis limits. y_min < y_max
        clim(c_min : float, c_max : float) : None
            Set the color axis limits. c_min < c_max
        colormap(cmap) : None
            Change the colormap to cmap, where cmap is an acceptable matplotlib colormap.
        set_fast() : None
            Speed up interactions with the drag_bars by blitting.
    """

    def __init__(self, n, direction = 'h'):

        self.fig = plt.figure()
        self.axes = self.fig.subplots(1,n)
        self.colorplots = []
        self.drag_bars = []
        self.max = n
        self.direction = direction
        self.count = 0
        self.fast = False

        self.drag_fig = plt.figure()
        self.drag_ax = self.drag_fig.subplots()

        self._color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def add_data(self, *args, **kwargs):

        if self.count < self.max:
            new_colorplot = quick_colorplot(*args, axes = self.axes[len(self.colorplots)], **kwargs)
            self.colorplots.append(new_colorplot)
            self.drag_bars.append(new_colorplot.drag_bar(direction = self.direction, axes = self.drag_ax, color = self._color_cycle[self.count]))
            self.count +=1
        else:
            print('No more data can be added to the figure!')
            return

        if len(self.colorplots) == self.max:
            self.drag_bars[0].join_drag_bars(*self.drag_bars[1:])

    def fix_layout(self):
        self.fig.tight_layout()

    def clim(self, c_min, c_max):
        for cplot in self.colorplots:
            cplot.clim(c_min, c_max)

    def xlim(self, x_min, x_max):
        for cplot in self.colorplots:
            cplot.xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        for cplot in self.colorplots:
            cplot.ylim(y_min, y_max)

    def colormap(self, cmap):
        for cplot in self.colorplots:
            cplot.colormap(cmap)

    def set_fast(self):
        if self.fast == False:
            for bar in self.drag_bars:
                bar.colorplot_line.set_animated(True)
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            for bar in self.drag_bars:
                bar.fast = True
                bar.background = self.background
                bar.colorplot.ax.draw_artist(bar.colorplot_line)
            self.fig.canvas.blit(self.fig.bbox)
            self.fast = True

    def stop_fast(self):
        if self.fast:
            for bar in self.drag_bars:
                bar.colorplot_line.set_animated(False)
                bar.fast = False
            self.fig.canvas.draw()
            self.fast = False

def waterfall(*spectra_list, **kwargs):

    r'''
    waterfall(*spectra_list, **kwargs) is a convenience method for creating a waterfall plot.
    Just like didv.colorplot, waterfall takes an arbitrary number of non-keyword arguments.
    These non-keyword arguments are either strings or lists of didv.spectrum.
    If a string is 'BASENAME', search for all files in the current directory with a filename
    that matches 'BASENAMEXXXXX.dat', where XXXXX is a five-digit number.
    waterfall() also takes optional keyword arguments defined below.

    Optional Keyword Arguments:
        vertical_shift : float
            Offset each curve vertically by vertical_shift.
        reverse : bool
            If True, plot the spectra in reverse order.
    '''

    if 'vertical_shift' in kwargs:
        vertical_shift = kwargs['vertical_shift']
    else:
        vertical_shift = 0
    if 'reverse' in kwargs:
        reverse = kwargs['reverse']
    else:
        reverse = False

    spectra = parse_arguments(*spectra_list)
    if reverse:
        increment = -1
    else:
        increment = 1
    gate_list = np.array([spec.gate for spec in spectra])
    if gate_list.size != np.unique(gate_list).size:
        print('WARNING: DUPLICATE GATE VOLTAGES DETECTED')

    for spec in spectra:
        spec.data.rename(columns = {'Input 2 [AVG] (V)' : 'Input 2 (V)'}, inplace = True)
        if 'double_lockin' in kwargs:
            if kwargs['double_lockin']:
                spec.data.rename(columns = {'Input 3 [AVG] (V)' : 'Input 3 (V)'}, inplace = True)
                spec.data['Input 2 (V)'] = (spec.data['Input 2 (V)'] + spec.data['Input 3 (V)']) * 0.5
        if 'ping_remove' in kwargs:
            std_ping_remove(spec, kwargs['ping_remove'])

    w_plot = plot(spectra, waterfall = vertical_shift, increment = increment, **kwargs)
    w_plot.spectra_list = spectra
    return w_plot

def std_ping_remove(spec, n): #Removes pings from Input 2 [...] (V), if average over 3 sweeps or more
    data = pd.DataFrame()
    cnt = 0
    for channel_name in spec.data.columns:
        if 'Input 2 [0' in channel_name:
            data[channel_name] = spec.data[channel_name]
            cnt += 1
    if (cnt == 0) or (cnt == 1):
        return
    if cnt == 2:
        #print("WARNING in didv.std_ping_remove: Only two spectra per average...")
        return
    std = data.std(axis=1) # Maybe use interquartile range instead of standard deviation
    median = data.median(axis = 1)
    data[np.abs(data.sub(median,axis = 0)).gt(n*std,axis=0)] = np.nan
    spec.data['Input 2 (V)'] = data.mean(axis = 1)

class QueryException(Exception):

    def __init__(self, message):
        super(QueryException, self).__init__(message)

class _QueryTransformer(ast.NodeTransformer):

    def __init__(self):
        super(_QueryTransformer, self).__init__()

        self.whitelist = {
            ast.Expression, ast.Expr, ast.Load, ast.Name, ast.Call,
            ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.Invert,
            ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
            ast.BoolOp, ast.And, ast.Or,
            ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.IfExp,
            ast.Tuple, ast.List, ast.Dict,
            ast.Subscript, ast.Slice
        }
        try:
            self.whitelist.add(ast.Constant)
        except AttributeError:
            pass
        try:
            self.whitelist.add(ast.Num)
        except AttributeError:
            pass
        try:
            self.whitelist.add(ast.Str)
        except AttributeError:
            pass

    def visit_Name(self, node):
        return ast.Attribute(
            value = ast.Name(id = 'spec', ctx = ast.Load()),
            attr = node.id,
            ctx = node.ctx
        )

    def visit_Call(self, node):
        raise QueryException('Calling functions is not allowed in queries.')

    def generic_visit(self, node):
        if type(node) not in self.whitelist:
            raise QueryException('Illegal operation!')
        return super(_QueryTransformer, self).generic_visit(node)

def query(spec_list, query_string):

    r'''
    Filters a list of spectra using a specified condition.

    Args:
        spec_list : List[didv.spectrum]
            The input list of dI/dV spectra to filter through.
        query_string : str
            A string specifying the filtering condition.
            Examples:
                '-20 < gate < 0'
                '(gate * 10) % 4 == 0 and gate >= 12'
    
    Returns:
        A List[didv.spectrum] containing only the spectra that satisfy the condition
        specified in query_string.
    '''

    tree = ast.parse(query_string.strip(), mode = 'eval')
    tree = ast.fix_missing_locations(_QueryTransformer().visit(tree))
    code = compile(tree, '', mode = 'eval')

    fetched_spectra = []
    try:
        for spec in spec_list:
            if not isinstance(spec, spectrum):
                continue
            if eval(code):
                fetched_spectra.append(spec)
    except:
        print('INVALID QUERY STRING')
    return fetched_spectra

# TO DO: Waterfall plots would be simpler to generate if fixed_bias_plot and fixed_gate_plot
#        saved a copy of the data in the landau_fan or butterfly instance
def fixed_bias_plot(spectra_list, bias, lower_bound, upper_bound, axes = None, channel = None, cmap = None, shift_gate = 0, flip_bias = False, normalize = False, rasterized = False, parent = None):

    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    if cmap is None:
        cmap = 'RdYlBu_r'

    bias_list = spectra_list[0].data['Bias calc (V)'] # Assumes all spectra have same bias list
    try:
        left_bias = bias[0]
        right_bias = bias[1]
        left_bias_index = min(range(len(bias_list)), key = lambda bidx: abs(bias_list[bidx] - left_bias))
        right_bias_index = min(range(len(bias_list)), key = lambda bidx: abs(bias_list[bidx] - right_bias))
        if right_bias_index > left_bias_index:
            nbiases = right_bias_index - left_bias_index + 1
        else:
            nbiases = left_bias_index - right_bias_index + 1
    except TypeError:
        bias_index = min(range(len(bias_list)), key = lambda bidx: abs(bias_list[bidx] - bias))
        nbiases = 1

    if channel is None:
        channel = 'Input 2 (V)'

    gate = []
    data = []
    for spec in spectra_list:
        gate.append(spec.gate)
        if nbiases == 1:
            data.append(spec.data[channel][bias_index])
        else:
            if right_bias_index > left_bias_index:
                data.append(spec.data[channel][left_bias_index:right_bias_index+1])
            else:
                data.append(spec.data[channel][right_bias_index:left_bias_index+1])
    gate = np.array(gate) + shift_gate
    data = np.array(data)
    data = np.reshape(data, (len(data), nbiases))
    if normalize:
        gate_cp = np.copy(gate)
        rev_index_list = list(range(len(gate)))
        rev_index_list.reverse()
        unique_gates = []
        for rev_index in rev_index_list:
            if gate[rev_index] in unique_gates:
                try:
                    gate_cp[rev_index] = np.nan
                except ValueError:
                    gate_cp = np.array(gate_cp, dtype = np.float_)
                    gate_cp[rev_index] = np.nan
            else:
                unique_gates.append(gate[rev_index])
        gate_index_between = [enum_g[0] for enum_g in enumerate(gate_cp) if normalize[0] <= enum_g[1] <= normalize[1]]
        data_sum = np.nanmean(data[gate_index_between, :])
        if parent is not None:
            parent[0]._norm_scale[parent[1]] = data_sum
        data = data/data_sum

    new_gate = (gate[1:] + gate[:-1]) * 0.5
    new_gate = np.insert(new_gate, 0, gate[0] - (gate[1] - gate[0]) * 0.5)
    new_gate = np.append(new_gate, gate[-1] + (gate[-1] - gate[-2]) * 0.5)
    bounds = np.linspace(lower_bound, upper_bound, nbiases + 1)
    if flip_bias:
        bounds = np.linspace(upper_bound, lower_bound, nbiases + 1)
    x, y = np.meshgrid(new_gate, bounds)
    x = x.T
    y = y.T

    return axes.pcolormesh(x, y, data, cmap = cmap, rasterized = rasterized)

def fixed_gate_plot(spectra_list, gate, lower_bound, upper_bound, axes = None, channel = None, cmap = None, last = True, rasterized = False, scale = None):

    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    if cmap is None:
        cmap = 'RdYlBu_r'

    bias = np.array(spectra_list[0].data['Bias calc (V)'])
    try:
        left_gate = gate[0]
        right_gate = gate[1]
        if left_gate < right_gate:
            spectra = query(spectra_list, str(left_gate) + '<= gate <= ' + str(right_gate))
        else:
            spectra = query(spectra_list, str(right_gate) + '<= gate <= ' + str(left_gate))
    except TypeError:
        spectra = query(spectra_list, 'gate == ' + str(gate))
        if last:
            spectra = [ spectra[-1] ]
    ngates = len(spectra)
    if ngates == 0:
        print("ERROR: GATE NOT FOUND. FIXED_GATE_PLOT DOES NOT LOOK FOR NEAREST GATE")
        print("SKIPPING...")
        return None

    if channel is None:
        channel = 'Input 2 (V)'

    data = []
    gate = []
    for spec in spectra:
        data.append(spec.data[channel])
        gate.append(spec.gate)
    gate, _, data = zip(*sorted(zip(gate, range(len(gate)), data)))
    data = np.array(data)
    if scale is not None:
        data = data/scale
    data = np.reshape(data, (ngates, len(bias)))
    data = data.T

    new_bias = (bias[1:] + bias[:-1]) * 0.5
    new_bias = np.insert(new_bias, 0, bias[0] - (bias[1] - bias[0]) * 0.5)
    new_bias = np.append(new_bias, bias[-1] + (bias[-1] - bias[-2]) * 0.5)
    bounds = np.linspace(lower_bound, upper_bound, ngates + 1)
    x, y = np.meshgrid(new_bias, bounds)
    x = x.T
    y = y.T

    return axes.pcolormesh(x, y, data, cmap = cmap, rasterized = rasterized)

# TO DO: Implement add_data
# TO DO: Implement drag_bar
#class landau_fan(interactive_colorplot.colorplot):
class landau_fan():

    r'''
    Plot a Landau Fan described in filename.
    '''

    def __init__(self, filename, cache = None, fast = False):

        # TO DO: Implement drag_bar
        #interactive_colorplot.colorplot.__init__(self)

        self.magnet = []
        self.clow = []
        self.chigh = []
        self.spectra_list = []
        self._basenames = []
        with open(filename, 'r') as f:
            f_idx = 0
            for fline in f:
                field_line = fline.split()
                try:
                    self.magnet.append(float(field_line[0]))
                except ValueError:
                    continue
                try:
                    self.clow.append(float(field_line[1]))
                except ValueError:
                    self.clow.append(None)
                try:
                    self.chigh.append(float(field_line[2]))
                except ValueError:
                    self.chigh.append(None)
                self._basenames.append(field_line[3:])
                if (cache is not None) and f_idx < len(cache._basenames) and (field_line[3:] == cache._basenames[f_idx]):
                    if fast is False:
                        self.spectra_list.append(parse_arguments(*field_line[3:], cache = cache.spectra_list[f_idx]))
                    else:
                        self.spectra_list.append(cache.spectra_list[f_idx])
                else:
                    self.spectra_list.append(parse_arguments(*field_line[3:]))
                f_idx += 1
        if fast:
            self.spectra_list[-1] = parse_arguments(*self._basenames[-1])

        self.num_fields = len(self.spectra_list)
        self._shift_gate = np.zeros(self.num_fields)
        self._norm_scale = np.zeros(self.num_fields)

        for spectra in self.spectra_list:
            for spec in spectra:
                spec.data.rename(columns = {'Input 2 [AVG] (V)' : 'Input 2 (V)'}, inplace = True)

    def get_index_for_B(self, B):
        nearest_B_index, nearest_B = min(enumerate(self.magnet), key = lambda x: abs(x[1] - B))
        return nearest_B_index

    def ping_remove_for_B(self, ping_remove, B):
        nearest_B_index = self.get_index_for_B(B)
        print('WARNING: PING_REMOVE ALTERS THE DATA IN SPECTRA_LIST!')
        for spec in self.spectra_list[nearest_B_index]:
            std_ping_remove(spec, ping_remove)

    def ping_remove_for_all(self, ping_remove):
        print('WARNING: PING_REMOVE ALTERS THE DATA IN SPECTRA_LIST!')
        for spec_list in self.spectra_list:
            for spec in spec_list:
                std_ping_remove(spec, ping_remove)

    def plot(self, bias, cmap = None, center = False, width = None, flip_bias = False, normalize = False, rasterized = False):

        if cmap is None:
            cmap = 'RdYlBu_r'

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.bias = bias
        self.cond_lines = []
        self._draw_lines = []

        for idx in range(self.num_fields):
            field_value = self.magnet[idx]
            if width is None:
                if self.num_fields == 1:
                    lower_bound = field_value - 0.5
                    upper_bound = field_value + 0.5
                else:
                    try:
                        smaller_fvalue = max([val for val in self.magnet if val < field_value])
                        lower_bound = (smaller_fvalue + field_value) * 0.5
                    except ValueError:
                        larger_fvalue = min([val for val in self.magnet if val > field_value])
                        lower_bound = field_value - (larger_fvalue - field_value) * 0.5
                    try:
                        larger_fvalue = min([val for val in self.magnet if val > field_value])
                        upper_bound = (larger_fvalue + field_value) * 0.5
                    except ValueError:
                        smaller_fvalue = max([val for val in self.magnet if val < field_value])
                        upper_bound = field_value + (field_value - smaller_fvalue) * 0.5
                if center:
                    even_bound = min([upper_bound - field_value, field_value - lower_bound])
                    upper_bound = field_value + even_bound
                    lower_bound = field_value - even_bound
            else:
                lower_bound = field_value - width
                upper_bound = field_value + width
            parent = [self, idx]
            fbplot = fixed_bias_plot(self.spectra_list[idx], bias, lower_bound, upper_bound, axes = self.ax, cmap = cmap, shift_gate = self._shift_gate[idx], flip_bias = flip_bias, normalize = normalize, rasterized = rasterized, parent = parent)
            self.cond_lines.append(fbplot)
            if self.chigh[idx] is not None:
                if self.clow[idx] is not None:
                    fbplot.set_clim(self.clow[idx], self.chigh[idx])
                else:
                    fbplot.set_clim(0, self.chigh[idx])

    def colormap(self,cmap):
        for fbplot in self.cond_lines:
            fbplot.set_cmap(cmap)

    def clim_for_B(self, c_min, c_max, B):
        nearest_B_index = self.get_index_for_B(B)
        self.cond_lines[nearest_B_index].set_clim(c_min, c_max)

    def clim_for_all(self, c_min, c_max):
        for line in self.cond_lines:
            if line is not None:
                line.set_clim(c_min, c_max)

    def get_clim_for_B(self, B):
        nearest_B_index = self.get_index_for_B(B)
        return self.cond_lines[nearest_B_index].get_clim()

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

    def add_gate_for_B(self, gate_shift, B):
        nearest_B_index = self.get_index_for_B(B)
        self._shift_gate[nearest_B_index] += gate_shift

    def reset_gate_shift(self):
        self._shift_gate = np.zeros(self.num_fields)

    def draw(self):

        fig = self.fig
        ax = self.ax

        first_pt = [None, None]
        line = [None]
        event_handlers = [None, None, None]

        def on_motion(event):
            if event.inaxes != ax:
                return
            line[0][0].set_xdata([first_pt[0], event.xdata])
            line[0][0].set_ydata([first_pt[1], event.ydata])
            fig.canvas.draw()

        def on_click(event):
            if (event.xdata is not None) and (event.ydata is not None):
                if first_pt[0] is None :
                    first_pt[0] = event.xdata
                    first_pt[1] = event.ydata
                    line[0] = ax.plot([first_pt[0], first_pt[0]], [first_pt[1], first_pt[1]])
                    event_handlers[1] = fig.canvas.mpl_connect('motion_notify_event', on_motion)
                    event_handlers[2] = fig.canvas.mpl_connect('button_release_event', on_release)

        def on_release(event):
            for event_handler in event_handlers:
                if event_handler is not None:
                    fig.canvas.mpl_disconnect(event_handler)
            self._draw_lines.append(line[0][0])

        event_handlers[0] = fig.canvas.mpl_connect('button_press_event', on_click)

    def delete_draw(self):

        try:
            line = self._draw_lines.pop()
            line.remove()
        except IndexError:
            return

    def butterfly(self, gate, cmap = None, center = False, width = None, rasterized = False):

        if np.any(self._norm_scale == 0):
            scale = None
        else:
            scale = self._norm_scale

        return butterfly(self, gate, cmap = cmap, center = center, width = width, rasterized = rasterized, scale = scale)

    def waterfall(self, vertical_shift):

        self._waterfall_fig = plt.figure()
        self._waterfall_ax = self._waterfall_fig.add_subplot(111)

        sorted_fields = sorted(self.magnet)
        for nth in range(self.num_fields):

            nth_field = sorted_fields[nth]
            idx = self.magnet.index(nth_field)

            bias_list = self.spectra_list[idx][0].data['Bias calc (V)'] # Assumes all spectra have same bias list
            bias_index = min(range(len(bias_list)), key = lambda bidx: abs(bias_list[bidx] - self.bias))

            gate = []
            data = []
            clow, chigh = self.cond_lines[idx].get_clim()
            for spec in self.spectra_list[idx]:
                gate.append(spec.gate)
                data.append(spec.data['Input 2 (V)'][bias_index])
            gate = np.array(gate) + self._shift_gate[idx]
            data = np.array(data)/(chigh - clow) - clow + nth_field * vertical_shift

            wat_line = self._waterfall_ax.plot(gate, data)

    # TO DO: Complete help()
    def help(self):
        print(".plot(BIAS, CMAP = COLORMAP, WIDTH = WIDTH_VALUE)")
        print(".ping_remove_for_B(STANDARD_DEVIATION, MAGNETIC_FIELD)")
        print(".clim_for_B(LOWER_CLIM, UPPER_CLIM, MAGNETIC_FIELD)")
        print(".add_gate_for_B(GATE_SHIFT, MAGNETIC_FIELD)")
        print(".butterfly(GATE, CMAP = COLORMAP, WIDTH = WIDTH_VALUE)")
        print(".waterfall(VERTICAL_OFFSET)")

class butterfly():

    def __init__(self, landau_fan_object, gate, cmap = None, center = False, width = None, rasterized = False, scale = None):

        if cmap is None:
            cmap = 'RdYlBu_r'

        # energy_vs_B plot
        self._evb_fig = plt.figure()
        self._evb_ax = self._evb_fig.add_subplot(111)
        self._evb_lines = []
        self.l_fan = landau_fan_object
        self.gate = gate

        # TO DO: Remove code duplication
        for idx in range(self.l_fan.num_fields):
            field_value = self.l_fan.magnet[idx]
            if width is None:
                if self.l_fan.num_fields == 1:
                    lower_bound = field_value - 0.5
                    upper_bound = field_value + 0.5
                else:
                    try:
                        smaller_fvalue = max([val for val in self.l_fan.magnet if val < field_value])
                        lower_bound = (smaller_fvalue + field_value) * 0.5
                    except ValueError:
                        larger_fvalue = min([val for val in self.l_fan.magnet if val > field_value])
                        lower_bound = field_value - (larger_fvalue - field_value) * 0.5
                    try:
                        larger_fvalue = min([val for val in self.l_fan.magnet if val > field_value])
                        upper_bound = (larger_fvalue + field_value) * 0.5
                    except ValueError:
                        smaller_fvalue = max([val for val in self.l_fan.magnet if val < field_value])
                        upper_bound = field_value + (field_value - smaller_fvalue) * 0.5
                if center:
                    even_bound = min([upper_bound - field_value, field_value - lower_bound])
                    upper_bound = field_value + even_bound
                    lower_bound = field_value - even_bound
            else:
                lower_bound = field_value - width
                upper_bound = field_value + width
            if scale is None:
                scale_val = None
            else:
                scale_val = scale[idx]
            fgplot = fixed_gate_plot(self.l_fan.spectra_list[idx], gate, lower_bound, upper_bound, axes = self._evb_ax, cmap = cmap, rasterized = rasterized, scale = scale_val)
            self._evb_lines.append(fgplot)
            if fgplot is None:
                continue
            if self.l_fan.chigh[idx] is not None:
                if self.l_fan.clow[idx] is not None:
                    fgplot.set_clim(self.l_fan.clow[idx], self.l_fan.chigh[idx])
                else:
                    fgplot.set_clim(0, self.l_fan.chigh[idx])

    def get_index_for_B(self, B):
        nearest_B_index, nearest_B = min(enumerate(self.l_fan.magnet), key = lambda x: abs(x[1] - B))
        return nearest_B_index

    def clim_for_B(self, c_min, c_max, B):
        nearest_B_index = self.get_index_for_B(B)
        self._evb_lines[nearest_B_index].set_clim(c_min, c_max)

    def clim_for_all(self, c_min, c_max):
        for line in self._evb_lines:
            if line is not None:
                line.set_clim(c_min, c_max)

    def get_clim_for_B(self, B):
        nearest_B_index = self.get_index_for_B(B)
        return self._evb_lines[nearest_B_index].get_clim()

    def waterfall(self, vertical_shift):

        self._waterfall_fig = plt.figure()
        self._waterfall_ax = self._waterfall_fig.add_subplot(111)

        sorted_fields = sorted(self.l_fan.magnet)
        for nth in range(self.l_fan.num_fields):

            nth_field = sorted_fields[nth]
            idx = self.l_fan.magnet.index(nth_field)

            bias = self.l_fan.spectra_list[idx][0].data['Bias calc (V)']
            spectra = query(self.l_fan.spectra_list[idx], 'gate == ' + str(self.gate))
            ngates = len(spectra)
            if ngates == 0:
                print("ERROR: GATE NOT FOUND. WATERFALL DOES NOT LOOK FOR NEAREST GATE")
                print("SKIPPING...")
                continue
            clow, chigh = self._evb_lines[idx].get_clim()
            bias = np.array(spectra[0].data['Bias calc (V)'])
            data = np.array(spectra[0].data['Input 2 (V)'])/(chigh - clow) - clow + nth_field * vertical_shift

            wat_line = self._waterfall_ax.plot(bias, data)

def quick_landau_fan(filename, bias = 0, cmap = None, center = False, width = None, rasterized = False, normalize = False, cache = None, fast = False):

    fan = landau_fan(filename, cache = cache , fast = fast)
    fan.plot(bias, cmap = cmap, center = center, width = width, rasterized = rasterized, normalize = normalize)
    return fan