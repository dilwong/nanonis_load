#Imports Nanonis dI/dV spectroscopy files into Python

import numpy as np
import pandas as pd

import glob

import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from . import interactive_colorplot
except ImportError:
    import interactive_colorplot

#didv.spectra(filename) loads one Nanonis spectrum file (extension .dat) into Python
class spectrum():

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

    def __init__(self, spectra, channel = 'Input 2 (V)', names = None, use_attributes = False, start = None, increment = None, waterfall = 0.0, dark = False, multiply = None, plot_on_previous = False, gate_as_index = True):

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
            if waterfall == 0:
                spec_data.plot(x = spec_data.columns[0], y = channel, ax = self.ax, legend = False, label = spectrum_label)
            else:
                spec_data[channel] = spec_data[channel] + waterfall * idx * np.sign(increment) + 0.5 * (-np.sign(increment) + 1) * waterfall * (len(spectra) - 1)
                spec_data.plot(x = spec_data.columns[0], y = channel, ax = self.ax, legend = False, label = spectrum_label, color=tuple(cmap[idx]))

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

    def __init__(self, spectra_list, channel, index_range = None, index_label = 'Gate Voltage (V)', start = None, increment = None, transform = None, diff_axis = 0, dark = False, axes = None, over_iv = None, multiply = None, gate_as_index = False, rasterized = False, **kwargs):

        interactive_colorplot.colorplot.__init__(self)
        
        self.channel = channel
        self.spectra_list = spectra_list

        pcolor_cm = 'RdYlBu_r'

        if dark:
            plt.style.use('dark_background')

        bias = spectra_list[0].data.iloc[:,0].values
        if transform is None:
            if multiply is  None:
                self.data = pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values
            else:
                self.data = pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values * multiply
            if over_iv is not None:
                try:
                    self.current = pd.concat((spec.data['Current (A)'] for spec in spectra_list),axis=1).values - over_iv[0]
                except KeyError:
                    self.current = pd.concat((spec.data['Current [AVG] (A)'] for spec in spectra_list),axis=1).values - over_iv[0]
            self.bias = bias
        else:
            if (transform == 'diff') or (transform == 'derivative'):
                self.data = np.diff(pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values, axis = diff_axis)
                self.bias = bias[:-1]
                pcolor_cm = 'seismic'
            else:
                self.data = transform(pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values)
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
                    index_range = np.arange(len(spectra_list)) * increment + start # increment must be signed
        if len(index_range) == 2:
            self.index_list = np.linspace(index_range[0],index_range[1],len(spectra_list))
        if len(index_range) == len(spectra_list):
            self.index_list = np.array(index_range)
        if gate_as_index:
            self.index_list = np.array([spec.header['Gate (V)'] for spec in spectra_list])
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

def quick_colorplot(*args, **kwargs):

    spectra = []
    for arg in args:
        s, f = batch_load(arg)
        if f == []:
            print('WARNING: NO FILES WITH BASENAME ' + arg)
        if ('monotonic' in kwargs) and kwargs['monotonic']:
            if len(spectra) != 0:
                extreme_gate = s[0].header['Gate (V)']
                if spectra[0].header['Gate (V)'] > spectra[1].header['Gate (V)']: # Decreasing
                    spectra = [spec for spec in spectra if spec.header['Gate (V)'] > extreme_gate]
                elif spectra[0].header['Gate (V)'] < spectra[1].header['Gate (V)']: # Increasing
                    spectra = [spec for spec in spectra if spec.header['Gate (V)'] < extreme_gate]
        spectra.extend(s)

    if spectra == []:
        print('ERROR: NO FILES!')
        return
    if 'channel' in kwargs:
        return colorplot(spectra, **kwargs)
    else:
        for spec in spectra:
            spec.data.rename(columns = {'Input 2 [AVG] (V)' : 'Input 2 (V)'}, inplace = True)
            if 'double_lockin' in kwargs:
                if kwargs['double_lockin'] == True:
                    spec.data.rename(columns = {'Input 3 [AVG] (V)' : 'Input 3 (V)'}, inplace = True)
                    spec.data['Input 2 (V)'] = (spec.data['Input 2 (V)'] + spec.data['Input 3 (V)']) * 0.5
            if 'ping_remove' in kwargs:
                ping_remove(spec, kwargs['ping_remove'])
        if ('start' not in kwargs) and ('increment' not in kwargs) and ('Gate (V)' in spectra[0].header):
            gate_range = [spec.header['Gate (V)'] for spec in spectra]
            if 'gate_transform' in kwargs:
                gate_range = kwargs['gate_transform'](np.array(gate_range))
            return colorplot(spectra, channel = 'Input 2 (V)', index_range = gate_range, **kwargs)
        else:
            return colorplot(spectra, channel = 'Input 2 (V)', **kwargs)

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

def ping_remove(spectrum, n): #Removes pings from Input 2 [...] (V), if average over 3 sweeps or more
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