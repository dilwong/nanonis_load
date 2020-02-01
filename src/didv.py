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

import numpy as np
import pandas as pd

import glob

import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from . import interactive_colorplot
except ImportError:
    import interactive_colorplot

class spectrum():

    """
    didv.spectra(filename) loads one Nanonis spectrum file (extension .dat) into Python
    """

    def __init__(self, filename, attribute = None):

        #Read the header, build the header
        self.header = {}
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

    def to_clipboard(self):
        self.data.to_clipboard()

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

        self.arg_list = spectra_list
        self.state_for_update = {}
        self.terminate = False

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

        bias = self.spectra_list[0].data.iloc[:,0].values - bias_shift
        if transform is None:
            if multiply is None:
                self.data = pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values
            else:
                self.data = pd.concat((spec.data[self.channel] for spec in self.spectra_list),axis=1).values * multiply
            if over_iv is not None:
                try:
                    self.current = pd.concat((spec.data['Current (A)'] for spec in self.spectra_list),axis=1).values - over_iv[0]
                except KeyError:
                    self.current = pd.concat((spec.data['Current [AVG] (A)'] for spec in self.spectra_list),axis=1).values - over_iv[0]
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
        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = pcolor_cm, rasterized = rasterized)
        self.original_cmap = self.pcolor.cmap
        self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
        self.ax.set_xlabel('Sample Bias (V)')
        self.ax.set_ylabel(index_label)

        if dark:
            plt.style.use('default')

        self.xlist = self.bias
        self.ylist = self.gate

    def update(self):

        self.spectra_list = parse_arguments(*self.arg_list)

        # Only works for Input 2 (V) or similar types of data
        for spec in self.spectra_list:
            spec.data.rename(columns = {'Input 2 [AVG] (V)' : 'Input 2 (V)'}, inplace = True)
            if 'double_lockin' in self.state_for_update:
                spec.data.rename(columns = {'Input 3 [AVG] (V)' : 'Input 3 (V)'}, inplace = True)
                spec.data['Input 2 (V)'] = (spec.data['Input 2 (V)'] + spec.data['Input 3 (V)']) * 0.5
        if 'ping_remove' in self.state_for_update:
            for spec in self.spectra_list:
                std_ping_remove(spec, ping_remove)
        bias = self.spectra_list[0].data.iloc[:,0].values # No bias_shift
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
        self.colorbar.remove()
        self.pcolor.remove()
        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = cmap)
        self.clim(clim_min, clim_max)
        self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
        self.xlist = self.bias
        self.ylist = self.gate
        for dragbar in self.__draggables__:
            dragbar.update_data()
        self.fig.canvas.draw()

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

def batch_load(basename, file_range = None, attribute_list = None):

    if file_range is None:
        file_range = range(1000)

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
    for channel_name in spectrum.data.columns:
        if 'Input 2 [0' in channel_name:
            data[channel_name] = spectrum.data[channel_name]
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

# gui_colorplot
