"""
Imports Nanonis dI/dV spectroscopy files into Python

Usage:

spec = didv.spectrum(FILENAME) loads a single Bias Spectroscopy .dat file into a variable named spec
spec.header contains ancillary information, spec.data is array of data

specs = didv.batch_load(BASENAME) loads multiple Bias Spectroscopy .dat files into a list named specs
didv.batch_load searches for all files with filename BASENAMEXXXXX.dat, where XXXXX is a five-digit number

didv.plot(spectra, channel = NameOfChannel) plots sample bias vs. the channel NameOfChannel
spectra is either a single spectrum loaded via didv.spectrum or a list of spectra loaded via didv.batch_load

didv.waterfall(spectra_list, vertical_shift = NUMBER, reverse = False) makes a waterfall plot
spectra_list is a series of either lists of didv.spectrum objects or BASENAME strings

p = didv.colorplot(spectra_list) plots dI/dV(Vs, Vg)
This defaults to channel = 'Input 2 (V)' or 'Input 2 [AVG] (V)'. Use double_lockin = True to average with 'Input 3 (V)'
p.drag_bar(direction = 'v' or 'h', locator = False)

"""

# TO DO: Better docstrings
# TO DO: Implement GUI using tkinter listbox and matplotlib.use("TkAgg")
# TO DO: Test for floating point issues?

import numpy as np
import pandas as pd

import glob

import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from . import interactive_colorplot
except ImportError:
    import interactive_colorplot

import traceback

class spectrum():

    """
    didv.spectra(filename) loads one Nanonis spectrum file (extension .dat) into Python
    """

    def __init__(self, filename, attribute = None):

        #Read the header, build the header
        self.header = {}
        self.__filename__ = filename
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
            if 'X (m)' in self.header:
                self.header['x (nm)'] = float(self.header['X (m)'])*1e9
            if 'Y (m)' in self.header:
                self.header['y (nm)'] = float(self.header['Y (m)'])*1e9
            if 'Z (m)' in self.header:
                self.header['z (nm)'] = float(self.header['Z (m)'])*1e9
            if 'Gate Voltage (V)' in self.header:
                self.header['Gate (V)'] = float(self.header['Gate Voltage (V)'])
                self.gate = self.header['Gate (V)']
            if attribute:
                self.header['attribute'] = attribute

        self.data = pd.read_csv(filename, sep = '\t', header = header_lines, skip_blank_lines = False)

    def to_clipboard(self, channel = None):
        if channel is None:
            self.data.to_clipboard()
        else:
            self.data[channel].to_clipboard(header = True)

    def plot(self, channel = 'Input 2 (V)', label = 'gate', multiply = 1, add = 0, plot_on_previous = False, **kwargs):
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
            return dat.plot(x = 'Bias calc (V)', y = channel, label = legend_label, ax = plt.gca(), **kwargs)
        return dat.plot(x = 'Bias calc (V)', y = channel, label = legend_label, **kwargs)

# Plot a spectrum
class plot():

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
                                gate_as_index = True, **kwargs):

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
                    plot_args['color'] = color
            if bias_shift != 0:
                spec_data.iloc[:,0] -= bias_shift
            spec_data.plot(**plot_args)

        #Make a legend
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
            legend_line.set_picker(5)
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

    "TO DO: WRITE DOCSTRING"

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

        self.arg_list = spectra_list
        self.state_for_update = {}
        self.terminate = False
        self.initial_kwarg_state = kwargs
        self.__bshift__ = bias_shift
        self.__linecut_event_handlers__ = []

        self.spectra_list = parse_arguments(*spectra_list)
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
            self.index_list = np.array([int(s.__filename__.split('.')[0].split('_')[-1]) for s in self.spectra_list])
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
            if gate_as_index and (start is None) and (increment is None):
                self.index_list = np.array([spec.gate for spec in self.spectra_list])
        # gate_transform depreciated
        self.gate = self.index_list

        if over_iv is not None:
            self.data = self.data/self.current*(self.bias[:,np.newaxis] - over_iv[1])
        if over_current is not None: # Do not use over_iv and over_current at the same time!!
            self.current = pd.concat(((spec.data.get('Current (A)', 0) + spec.data.get('Current [AVG] (A)', 0)) for spec in self.spectra_list),axis=1).values - over_current
            self.data = self.data/self.current
        if post_transform is not None:
            self.data = post_transform(self.data)
        #

        # TO DO: This program assumes all spectra have the same bias list.
        #        Implement the ability to handle spectra with different bias lists.
        new_bias = (bias[1:] + bias[:-1]) * 0.5
        new_bias = np.insert(new_bias, 0, bias[0] - (bias[1] - bias[0]) * 0.5)
        new_index_range = (self.index_list[1:] + self.index_list[:-1]) * 0.5
        new_index_range = np.insert(new_index_range, 0, self.index_list[0] - (self.index_list[1] - self.index_list[0]) * 0.5)
        if (transform != 'diff') and (transform != 'derivative'):
            new_bias = np.append(new_bias, bias[-1] + (bias[-1] - bias[-2]) * 0.5)
            new_index_range = np.append(new_index_range, self.index_list[-1] + (self.index_list[-1] - self.index_list[-2]) * 0.5)
        x, y = np.meshgrid(new_bias, new_index_range) # Will handle non-linear bias array
        x = x.T
        y = y.T
        try:
            bias_shift_len = len(bias_shift)
            if bias_shift_len == len(self.index_list):
                new_bias_shift = (np.array(bias_shift[1:]) + np.array(bias_shift[:-1])) * 0.5 # Is this the right thing to do?
                new_bias_shift = np.insert(new_bias_shift, 0, bias_shift[0] - (bias_shift[1] - bias_shift[0]) * 0.5)
                new_bias_shift = np.append(new_bias_shift, bias_shift[-1] + (bias_shift[-1] - bias_shift[-2]) * 0.5)
                for idx, shift_val in enumerate(new_bias_shift): # Doesn't play nice with vertical dragbar
                    x[:,idx] = x[:,idx] + shift_val
        except TypeError:
            pass
        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = pcolor_cm, rasterized = rasterized)
        self.original_cmap = self.pcolor.cmap
        if colorbar:
            self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
        self.ax.set_xlabel('Sample Bias (V)')
        self.ax.set_ylabel(index_label)
        self.__x_axes_limits__ = list(self.ax.get_xlim())
        self.__y_axes_limits__ = list(self.ax.get_ylim())

        if dark:
            plt.style.use('default')

        self.xlist = self.bias
        self.ylist = self.gate

    def std_clim(self, n):
        mean = np.mean(self.data)
        std = np.std(self.data)
        self.clim(mean - n*std, mean + n*std)

    def percentile_clim(self, lower, upper):
        self.clim(np.percentile(self.data, lower), np.percentile(self.data, upper))

    def whole_range(self):
        min_bias = np.min(self.bias)
        max_bias = np.max(self.bias)
        min_gate = np.min(self.gate)
        max_gate = np.max(self.gate)
        self.xlim(min_bias, max_bias)
        self.ylim(min_gate, max_gate)

    def update(self):

        try:
            self.spectra_list = parse_arguments(*self.arg_list)

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
            if self.state_for_update['running_index']:
                self.index_list = np.array([int(s.__filename__.split('.')[0].split('_')[-1]) for s in self.spectra_list])
            else:
                self.index_list = np.array([spec.gate for spec in self.spectra_list])
            new_bias = (bias[1:] + bias[:-1]) * 0.5
            new_bias = np.insert(new_bias, 0, bias[0] - (bias[1] - bias[0]) * 0.5)
            new_index_range = (self.index_list[1:] + self.index_list[:-1]) * 0.5
            new_index_range = np.insert(new_index_range, 0, self.index_list[0] - (self.index_list[1] - self.index_list[0]) * 0.5)
            new_bias = np.append(new_bias, bias[-1] + (bias[-1] - bias[-2]) * 0.5)
            new_index_range = np.append(new_index_range, self.index_list[-1] + (self.index_list[-1] - self.index_list[-2]) * 0.5)
            x, y = np.meshgrid(new_bias, new_index_range) # Will handle non-linear bias array
            x = x.T
            y = y.T
            cmap = self.pcolor.cmap
            clim_min, clim_max = self.pcolor.get_clim()
            self.bias = bias
            self.gate = self.index_list
            self.data = pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values  # No transform, multiply, over_iv
            colorbar = self.initial_kwarg_state['colorbar'] if ('colorbar' in self.initial_kwarg_state) else True
            if colorbar:
                self.colorbar.remove()
            self.pcolor.remove()
            self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = cmap)
            self.clim(clim_min, clim_max)
            if colorbar:
                self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
            self.xlist = self.bias
            self.ylist = self.gate
            for dragbar in self.__draggables__:
                dragbar.update_data()
            self.fig.canvas.draw()
        except:
            err_detect = traceback.format_exc()
            print(err_detect)
            raise

    def update_loop(self, wait_time):

        import time
        while not self.terminate:
            time.sleep(wait_time)
            self.update()

    def refresh(self, wait_time = 5):

        try:
            import thread
        except ModuleNotFoundError:
            import _thread as thread

        def handle_close(event):
            self.terminate = True
        self.fig.canvas.mpl_connect('close_event', handle_close)

        thread.start_new_thread(self.update_loop, (wait_time, ))

    def save_data_to_file(self, filename):

        x, y = np.meshgrid(self.bias, self.index_list)
        x = x.T
        y = y.T
        try:
            bias_shift_len = len(self.__bshift__)
            if bias_shift_len == len(self.index_list):
                for idx, shift_val in enumerate(self.__bshift__):
                    x[:,idx] = x[:,idx] + shift_val
        except TypeError:
            pass
        x = np.reshape(x, (x.size, 1))
        y = np.reshape(y, (y.size, 1))
        z = np.reshape(self.data, (self.data.size, 1))
        cols = np.append(x, y, axis = 1)
        cols = np.append(cols, z, axis = 1)
        np.savetxt(filename, cols)

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
        self.__peak_pairs__ = pairs
        self.__peak_cluster_labels__ = label_list

    def get_peak_cluster(self, bias, gate):
        gate_index = min(range(len(self.gate)), key = lambda idx: abs(self.gate[idx] - gate))
        bias_index = min(range(len(self.bias)), key = lambda idx: abs(self.bias[idx] - bias))
        nearest_pair_index = min(range(len(self.__peak_pairs__)), key = lambda idx: (bias_index - self.__peak_pairs__[idx][0])**2 + (gate_index - self.__peak_pairs__[idx][1])**2)
        cluster_index = self.__peak_cluster_labels__[nearest_pair_index]
        cluster_list = [idx for idx, val in enumerate(self.__peak_cluster_labels__) if val == cluster_index]
        relevant_pairs = [self.__peak_pairs__[idx] for idx in cluster_list]
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
        all_pairs = list(zip(self.__peak_pairs__,self.__peak_cluster_labels__))
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
            bias_shift_len = len(self.__bshift__)
            if bias_shift_len == len(self.index_list):
                for idx, shift_val in enumerate(self.__bshift__):
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

        #for event_handler in self.__linecut_event_handlers__:
        #    for itm in event_handler:
        #        if itm is not None:
        #        fig.canvas.mpl_disconnect(itm)
        event_handlers = [None, None, None]
        self.__linecut_event_handlers__.append(event_handlers)

        if axes is None:
            linecut_fig = plt.figure()
            linecut_ax = linecut_fig.add_subplot(111)
        else:
            linecut_ax = axes
            linecut_fig = linecut_ax.figure

        def nearest_pt(bias_val, gate_val):
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

def batch_load(basename, file_range = None, attribute_list = None):

    if file_range is None:
        file_range = range(9999)

    file_string = basename + '*.dat'
    file_exist = glob.glob(file_string)

    file_list = []
    spectrum_array = []
    for idx, file_number in enumerate(file_range):
        filename = basename + '%0*d' % (5, file_number) + '.dat'
        if filename in file_exist:
            file_list.append(filename)
            spectrum_inst = spectrum(filename)
            if attribute_list:
                spectrum_inst.header['attribute'] = attribute_list[idx]
            spectrum_array.append(spectrum_inst)

    return (spectrum_array, file_list)

def parse_arguments(*spectra_arguments):

    spectra = []
    for arg in spectra_arguments:
        if type(arg) == str:
            s, f = batch_load(arg)
            if f == []:
                print('WARNING: NO FILES WITH BASENAME ' + arg)
            # monotonic keyword depreciated
            spectra.extend(s)
        elif type(arg) == list:
            spectra.extend(arg) # Does not check if list contains only didv.spectrum
        elif type(arg) == didv.spectrum:
            spectra.append(arg)
        else:
            print('INCORRECT TYPE ERROR IN didv.parse_arguments')
    if not spectra:
        print('ERROR: NO FILES!')
    return spectra

def quick_colorplot(*args, **kwargs):

    return colorplot(*args, **kwargs)

class multi_colorplot():

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

        self.__color_cycle__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def add_data(self, *args, **kwargs):

        if self.count < self.max:
            new_colorplot = quick_colorplot(*args, axes = self.axes[len(self.colorplots)], **kwargs)
            self.colorplots.append(new_colorplot)
            self.drag_bars.append(new_colorplot.drag_bar(direction = self.direction, axes = self.drag_ax, color = self.__color_cycle__[self.count]))
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

    "TO DO: WRITE DOCSTRING"

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

def std_ping_remove(spectrum, n): #Removes pings from Input 2 [...] (V), if average over 3 sweeps or more
    data = pd.DataFrame()
    cnt = 0
    for channel_name in spectrum.data.columns:
        if 'Input 2 [0' in channel_name:
            data[channel_name] = spectrum.data[channel_name]
            cnt += 1
    if (cnt == 0) or (cnt == 1):
        return
    if cnt == 2:
        #print("WARNING in didv.std_ping_remove: Only two spectra per average...")
        return
    std = data.std(axis=1) # Maybe use interquartile range instead of standard deviation
    median = data.median(axis = 1)
    data[np.abs(data.sub(median,axis = 0)).gt(n*std,axis=0)] = np.nan
    spectrum.data['Input 2 (V)'] = data.mean(axis = 1)

def query(spec_list, query_string):

    new_query_string = query_string.replace('gate','spec.header["Gate (V)"]')

    fetched_spectra = []
    try:
        for spec in spec_list:
            if eval(new_query_string, {'__builtins__': None}, {'spec': spec} ):
                fetched_spectra.append(spec)
    except TypeError: # What about SyntaxError?
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
        data_sum = np.sum(data[gate_index_between, :]) / (len(gate_index_between) * nbiases)
        if parent is not None:
            parent[0].__norm_scale__[parent[1]] = data_sum
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
#class landau_fan(interactive_colorplot.colorplot):
class landau_fan():

    "TO DO: WRITE DOCSTRING"

    def __init__(self, filename):

        # TO DO: Implement drag_bar
        #interactive_colorplot.colorplot.__init__(self)

        self.magnet = []
        self.clow = []
        self.chigh = []
        self.spectra_list = []
        with open(filename, 'r') as f:
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
                self.spectra_list.append(parse_arguments(*field_line[3:]))

        self.num_fields = len(self.spectra_list)
        self.__shift_gate__ = np.zeros(self.num_fields)
        self.__norm_scale__ = np.zeros(self.num_fields)

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
        self.__draw_lines__ = []

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
            fbplot = fixed_bias_plot(self.spectra_list[idx], bias, lower_bound, upper_bound, axes = self.ax, cmap = cmap, shift_gate = self.__shift_gate__[idx], flip_bias = flip_bias, normalize = normalize, rasterized = rasterized, parent = parent)
            self.cond_lines.append(fbplot)
            if self.chigh[idx] is not None:
                if self.clow[idx] is not None:
                    fbplot.set_clim(self.clow[idx], self.chigh[idx])
                else:
                    fbplot.set_clim(0, self.chigh[idx])

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

    def add_gate_for_B(self, gate_shift, B):
        nearest_B_index = self.get_index_for_B(B)
        self.__shift_gate__[nearest_B_index] += gate_shift

    def reset_gate_shift(self):
        self.__shift_gate__ = np.zeros(self.num_fields)

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
            self.__draw_lines__.append(line[0][0])

        event_handlers[0] = fig.canvas.mpl_connect('button_press_event', on_click)

    def delete_draw(self):

        try:
            line = self.__draw_lines__.pop()
            line.remove()
        except IndexError:
            return

    def butterfly(self, gate, cmap = None, center = False, width = None, rasterized = False):

        if np.any(self.__norm_scale__ == 0):
            scale = None
        else:
            scale = self.__norm_scale__

        return butterfly(self, gate, cmap = cmap, center = center, width = width, rasterized = rasterized, scale = scale)

    def waterfall(self, vertical_shift):

        self.__waterfall_fig__ = plt.figure()
        self.__waterfall_ax__ = self.__waterfall_fig__.add_subplot(111)

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
            gate = np.array(gate) + self.__shift_gate__[idx]
            data = np.array(data)/(chigh - clow) - clow + nth_field * vertical_shift

            wat_line = self.__waterfall_ax__.plot(gate, data)

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
        self.__evb_fig__ = plt.figure()
        self.__evb_ax__ = self.__evb_fig__.add_subplot(111)
        self.__evb_lines__ = []
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
            fgplot = fixed_gate_plot(self.l_fan.spectra_list[idx], gate, lower_bound, upper_bound, axes = self.__evb_ax__, cmap = cmap, rasterized = rasterized, scale = scale_val)
            self.__evb_lines__.append(fgplot)
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
        self.__evb_lines__[nearest_B_index].set_clim(c_min, c_max)

    def clim_for_all(self, c_min, c_max):
        for line in self.__evb_lines__:
            if line is not None:
                line.set_clim(c_min, c_max)

    def get_clim_for_B(self, B):
        nearest_B_index = self.get_index_for_B(B)
        return self.__evb_lines__[nearest_B_index].get_clim()

    def waterfall(self, vertical_shift):

        self.__waterfall_fig__ = plt.figure()
        self.__waterfall_ax__ = self.__waterfall_fig__.add_subplot(111)

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
            clow, chigh = self.__evb_lines__[idx].get_clim()
            bias = np.array(spectra[0].data['Bias calc (V)'])
            data = np.array(spectra[0].data['Input 2 (V)'])/(chigh - clow) - clow + nth_field * vertical_shift

            wat_line = self.__waterfall_ax__.plot(bias, data)

def quick_landau_fan(filename, bias = 0, cmap = None, center = False, width = None, rasterized = False, normalize = False):

    fan = landau_fan(filename)
    fan.plot(bias, cmap = cmap, center = center, width = width, rasterized = rasterized, normalize = normalize)
    return fan
