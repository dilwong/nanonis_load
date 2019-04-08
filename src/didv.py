#Imports Nanonis dI/dV spectroscopy files into Python

import numpy as np
import pandas as pd

import glob

import matplotlib.pyplot as plt

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

    def __init__(self, spectra, channel, names = None, use_attributes = False):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        try:
            spectra_iterator = iter(spectra)
        except TypeError:
            spectra_iterator = iter([spectra])
        for idx, spectrum_inst in enumerate(spectra_iterator):
            try:
                if use_attributes:
                    spectrum_label = str(spectrum_inst.header['attribute'])
                else:
                    spectrum_label = str(names[idx])
            except (TypeError, IndexError):
                spectrum_label = str(idx)
            spectrum_inst.data.plot(x = spectrum_inst.data.columns[0], y = channel, ax = self.ax, legend = False, label = spectrum_label)
        
        #Make a legend
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        self.legend = self.ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
        plot_lines = self.ax.get_lines()
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

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

class colorplot():

    def __init__(self, spectra_list, channel, index_range = [-1000, 1000]):

        self.channel = channel
        self.spectra_list = spectra_list
        
        self.data = pd.concat((spec.data[channel] for spec in spectra_list),axis=1).values
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        bias = spectra_list[0].data.iloc[:,0].values
        self.bias = bias
        if len(index_range) == 2:
            x, y = np.mgrid[bias[0]:bias[-1]:bias.size*1j,index_range[0]:index_range[1]:len(spectra_list)*1j]
            self.index_list = np.linspace(index_range[0],index_range[1],len(spectra_list))
        elif len(index_range) == len(spectra_list):
            x, y = np.meshgrid(bias, index_range)
            x = x.T
            y = y.T
            self.index_list = np.array(index_range)
        else:
            x, y = np.mgrid[bias[0]:bias[-1]:bias.size*1j,-1000:1000:len(spectra_list)*1j]
            self.index_list = np.linspace(-1000,1000,len(spectra_list))
        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = 'seismic')
        self.fig.colorbar(self.pcolor, ax = self.ax)

        self.xdata_array = []
        self.ydata_array = []
        def on_click(event):
            if event.xdata is not None:
                self.xdata_array.append(event.xdata)
            if event.ydata is not None:
                self.ydata_array.append(event.ydata)
            if event.button == 3:
                self.xdata_array = []
                self.ydata_array = []

        self.on_click = on_click
        cid = self.fig.canvas.mpl_connect('button_press_event', on_click)

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

    def show_spectra(self):

        sweeps = []
        attrib_list = []
        for index_value in self.ydata_array:
            idx, index_num = min(enumerate(self.index_list), key = lambda x: abs(x[1] - index_value))
            sweeps.append(self.spectra_list[idx])
            attrib_list.append(index_num)

        plot(sweeps, self.channel, names = attrib_list)
    
    # TO DO: Make interactive?
    def show_index(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for bias_value in self.xdata_array:
            idx, bias_num = min(enumerate(self.bias), key = lambda x: abs(x[1] - bias_value))
            ax.plot(self.index_list, self.data[idx,:] ,label = str(bias_num))
        ax.legend()

def batch_load(basename, file_range, attribute_list = None):
    
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

