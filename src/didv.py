#Imports Nanonis dI/dV spectroscopy files into Python

import numpy as np
import pandas as pd

import glob

import matplotlib.pyplot as plt
from matplotlib import cm

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
            if attribute:
                self.header['attribute'] = attribute

        self.data = pd.read_csv(filename, sep = '\t', header = header_lines, skip_blank_lines = False)

# Plot a spectrum
class plot():

    def __init__(self, spectra, channel, names = None, use_attributes = False, start = None, increment = None, waterfall = 0.0, dark = False):

        if waterfall != 0: # Does not work if spectra is a non-list iterator
            if dark:
                plt.style.use('dark_background')
                cmap = cm.get_cmap('RdYlBu')(np.linspace(0.1,0.8,len(spectra)))
            else:
                plt.style.use('default')
                cmap = cm.get_cmap('brg')(np.linspace(0,0.6,len(spectra)))
            cmap=cmap[::-1]
        
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
            if waterfall == 0:
                spectrum_inst.data.plot(x = spectrum_inst.data.columns[0], y = channel, ax = self.ax, legend = False, label = spectrum_label)
            else:
                spec_data = spectrum_inst.data.copy()
                spec_data[channel] = spec_data[channel] + waterfall * idx * np.sign(increment) + 0.5 * (-np.sign(increment) + 1) * waterfall * len(spectra)
                spec_data.plot(x = spectrum_inst.data.columns[0], y = channel, ax = self.ax, legend = False, label = spectrum_label, color=tuple(cmap[idx]))
        
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

class colorplot():

    def __init__(self, spectra_list, channel, index_range = None, start = None, increment = None, transform = None, diff_axis = 0, dark = False):

        self.channel = channel
        self.spectra_list = spectra_list
        
        self.__draggables__ = []
        self.__drag_h_count__ = 0
        self.__drag_v_count__ = 0
        self.__color_cycle__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.__drag_color_index__ = 0
        self.__drag_rev_legend_map__ = dict()

        if dark:
            plt.style.use('dark_background')
        
        bias = spectra_list[0].data.iloc[:,0].values
        if transform is None:
            self.data = pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values
            self.bias = bias
        else:
            if (transform == 'diff') or (transform == 'derivative'):
                self.data = np.diff(pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values, axis = diff_axis)
                self.bias = bias[:-1]
            else:
                self.data = transform(pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values)
                self.bias = bias
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        if index_range is None:
            if start is None:
                index_range = [-1000, 1000]
            else:
                if increment is None:
                    index_range = [start, 1000]
                else:
                    index_range = np.arange(len(spectra_list)) * increment + start # increment must be signed
        if len(index_range) == 2:
            x, y = np.mgrid[bias[0]:bias[-1]:bias.size*1j,index_range[0]:index_range[1]:len(spectra_list)*1j] # Will not handle non-linear bias array
            self.index_list = np.linspace(index_range[0],index_range[1],len(spectra_list))
        elif len(index_range) == len(spectra_list):
            x, y = np.meshgrid(bias, index_range) # Will handle non-linear bias array
            x = x.T
            y = y.T
            self.index_list = np.array(index_range)
        else:
            x, y = np.mgrid[bias[0]:bias[-1]:bias.size*1j,-1000:1000:len(spectra_list)*1j] # Will not handle non-linear bias array
            self.index_list = np.linspace(-1000,1000,len(spectra_list))
        pcolor_cm = 'RdYlBu_r' # pcolor_cm = 'seismic'
        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = pcolor_cm)
        self.fig.colorbar(self.pcolor, ax = self.ax)

        self.xdata_array = []
        self.ydata_array = []
        def on_click(event):
            if event.xdata is not None:
                self.xdata_array.append(event.xdata)
            if event.ydata is not None:
                self.ydata_array.append(event.ydata)
            if event.button == 3: # Right click to clear
                self.xdata_array = []
                self.ydata_array = []

        self.on_click = on_click
        self.fig.canvas.mpl_connect('button_press_event', on_click)

        if dark:
            plt.style.use('default')

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

    def clim(self, c_min, c_max):
        self.pcolor.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.pcolor.set_cmap(cmap)

    def clear_line_plots(self):
        self.xdata_array = []
        self.ydata_array = []

    def show_spectra(self, channel = None):

        sweeps = []
        attrib_list = []
        for index_value in self.ydata_array:
            idx, index_num = min(enumerate(self.index_list), key = lambda x: abs(x[1] - index_value))
            sweeps.append(self.spectra_list[idx])
            attrib_list.append(index_num)

        if channel is None:
            plot_channel = self.channel
        else:
            plot_channel = channel
        plot(sweeps, plot_channel, names = attrib_list)
    
    def show_index(self, sweep = False):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not sweep:
            points_array = self.xdata_array
            x_axis = self.index_list
            selection_array = self.bias
            slice_dict = {'right' : slice(None)}
            slice_const = 'left'
        else:
            points_array = self.ydata_array
            x_axis = self.bias
            selection_array = self.index_list
            slice_dict = {'left' : slice(None)}
            slice_const = 'right'
        for value in points_array:
            slice_dict[slice_const], num = min(enumerate(selection_array), key = lambda x: abs(x[1] - value))
            ax.plot(x_axis, self.data[slice_dict['left'],slice_dict['right']], label = str(num))
        legend = ax.legend()

        plot_lines = ax.get_lines()
        legend_lines = legend.get_lines()
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
            fig.canvas.draw()
        
        fig.canvas.mpl_connect('pick_event', pick_line)

    def show_sweep(self):
        self.show_index(sweep = True)

    # Does not work with non-None transform
    def replace_data(self, index, channel):
        idx, _ = min(enumerate(self.index_list), key = lambda x: abs(x[1] - index))
        self.data[:,idx] = self.spectra_list[idx].data[channel].values
        self.pcolor.set_array(self.data[:-1,:-1].ravel())
        self.fig.canvas.draw()

    def plot_all(self, rng = slice(None)):
        p = plot(self.spectra_list[rng], self.channel, names = self.index_list)
        plot_lines = p.ax.get_lines()
        for plot_line in plot_lines:
            plot_line.set_visible(False)

    def drag_bar(self, direction = 'horizontal', locator = False):

        drag_index = len(self.__draggables__)
        if direction[0] == 'h':
            if self.__drag_h_count__ == 0:
                self.__drag_h_fig__ = plt.figure()
                self.__drag_h_ax__ = self.__drag_h_fig__.add_subplot(111)
            y_init = self.ax.get_ylim()[1] - (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.1
            line = self.ax.axhline(y_init, color = self.__color_cycle__[self.__drag_color_index__])
            idx, index_value = min(enumerate(self.index_list), key = lambda x: abs(x[1] - y_init))
            p_line, = self.__drag_h_ax__.plot(self.bias, self.data[:,idx], label = str(index_value), color = self.__color_cycle__[self.__drag_color_index__])
            legend = self.__drag_h_ax__.legend()
            count = self.__drag_h_count__
            self.__drag_h_count__ += 1
        elif direction[0] == 'v':
            if self.__drag_v_count__ == 0:
                self.__drag_v_fig__ = plt.figure()
                self.__drag_v_ax__ = self.__drag_v_fig__.add_subplot(111)
            x_init = self.ax.get_xlim()[0] + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.1
            line = self.ax.axvline(x_init, color = self.__color_cycle__[self.__drag_color_index__])
            idx, bias_value = min(enumerate(self.bias), key = lambda x: abs(x[1] - x_init))
            p_line, = self.__drag_v_ax__.plot(self.index_list, self.data[idx,:], label = str(bias_value), color = self.__color_cycle__[self.__drag_color_index__])
            legend = self.__drag_v_ax__.legend()
            count = self.__drag_v_count__
            self.__drag_v_count__ += 1
            if locator:
                v_line = self.__drag_h_ax__.axvline(x_init, color = self.__color_cycle__[self.__drag_color_index__])
        else:
            print('Direction must be "h" for horizontal or "v" for vertical.')
            return
        self.__draggables__.append({'count':count, 'direction': direction[0], 'line':line, 'press':False, 'plot':p_line, 'color':self.__color_cycle__[self.__drag_color_index__]})
        line.set_picker(5)
        self.__drag_color_index__ += 1
        self.__drag_color_index__ = self.__drag_color_index__ % len(self.__color_cycle__)
        for v in self.__draggables__:
            v['plot'].axes.get_legend().get_lines()[v['count']].set_visible(True)
            v['plot'].axes.get_legend().get_lines()[v['count']].set_picker(5)
            self.__drag_rev_legend_map__[v['plot']] = v['plot'].axes.get_legend().get_lines()[v['count']]

        def on_press(event):
            if event.inaxes != self.ax:
                return
            contains, _ = line.contains(event)
            if not contains:
                return
            self.__draggables__[drag_index]['press'] = True

        def on_motion(event):
            if event.inaxes != self.ax:
                return
            if self.__draggables__[drag_index]['press'] is False:
                return
            if direction[0] == 'h':
                self.__draggables__[drag_index]['line'].set_ydata([event.ydata, event.ydata])
                idx, index_value = min(enumerate(self.index_list), key = lambda x: abs(x[1] - event.ydata))
                self.__draggables__[drag_index]['plot'].set_ydata(self.data[:,idx])
                self.__draggables__[drag_index]['plot'].set_label(str(index_value))
                self.__drag_h_ax__.legend()
                for v in self.__draggables__:
                    v['plot'].axes.get_legend().get_lines()[v['count']].set_visible(True)
                self.__drag_h_fig__.canvas.draw()
            elif direction[0] == 'v':
                self.__draggables__[drag_index]['line'].set_xdata([event.xdata, event.xdata])
                idx, bias_value = min(enumerate(self.bias), key = lambda x: abs(x[1] - event.xdata))
                self.__draggables__[drag_index]['plot'].set_ydata(self.data[idx,:])
                self.__draggables__[drag_index]['plot'].set_label(str(bias_value))
                self.__drag_v_ax__.legend()
                for v in self.__draggables__:
                    v['plot'].axes.get_legend().get_lines()[v['count']].set_visible(True)
                self.__drag_v_fig__.canvas.draw()
            else:
                pass
            self.fig.canvas.draw()
            if locator:
                v_line.set_xdata([event.xdata, event.xdata])
                self.__drag_h_fig__.canvas.draw()

        def on_release(event):
            for v in self.__draggables__:
                v['plot'].axes.get_legend().get_lines()[v['count']].set_visible(True)
                v['plot'].axes.get_legend().get_lines()[v['count']].set_picker(5)
                self.__drag_rev_legend_map__[v['plot']] = v['plot'].axes.get_legend().get_lines()[v['count']]
            self.__draggables__[drag_index]['press'] = False
            self.fig.canvas.draw()

        self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.fig.canvas.mpl_connect('button_release_event', on_release)

        if ((direction[0] == 'h') and (self.__drag_h_count__ == 1)) or ((direction[0] == 'v') and (self.__drag_v_count__ == 1)):
            
            def pick_line(event):
                legend_line = event.artist
                for key, value in self.__drag_rev_legend_map__.items():
                    if id(value) == id(legend_line):
                        plot_line = key
                visibility = not plot_line.get_visible()
                plot_line.set_visible(visibility)
                plot_line.figure.canvas.draw()
            
            p_line.figure.canvas.mpl_connect('pick_event', pick_line)

def batch_load(basename, file_range = range(1000), attribute_list = None):
    
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