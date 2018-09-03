#Imports Nanonis dI/dV spectroscopy files into Python

import numpy
import os.path
import glob
import re

import matplotlib
import matplotlib.pyplot

import scipy
import scipy.signal

import sxm

#get_spectra(filename) loads one Nanonis spectrum file (extension .dat) into Python
class get_spectra():

    def __init__(self,filename):

        self.header = {}
        self.data = {}
        
        #Read the header, build the header
        with open(os.path.normpath(filename),'r') as file_id:
            file_line=file_id.readline()
            file_line=file_id.readline().strip()
            self.header['date']=file_line.split('\t')[1]
            file_line=file_id.readline()
            file_line=file_id.readline().strip()
            self.header['x (nm)']=float(file_line.split('\t')[1])*1e9
            file_line=file_id.readline().strip()
            self.header['y (nm)']=float(file_line.split('\t')[1])*1e9
            file_line=file_id.readline().strip()
            self.header['z (nm)']=float(file_line.split('\t')[1])*1e9
            file_line=file_id.readline().strip()
            self.header['z offset (nm)']=float(file_line.split('\t')[1])*1e9
            file_line=file_id.readline().strip()
            self.header['settling time (s)']=float(file_line.split('\t')[1])*1e9
            file_line=file_id.readline().strip()
            self.header['integration time (s)']=float(file_line.split('\t')[1])*1e9
            file_line=file_id.readline()
            file_line=file_id.readline()
            file_line=file_id.readline()
            file_line=file_id.readline()
            file_line=file_id.readline()
            file_line=file_id.readline()
            file_line=file_id.readline()
            file_line=file_id.readline().strip()
            self.header['data info']=file_line.split('\t')

        #Read data from .dat file into self.data dictionary
        data_array=numpy.genfromtxt(filename,skip_header=17)
        for data_title in self.header['data info']:
            self.data[data_title]=data_array[:,self.header['data info'].index(data_title)]
            
        return

#Function plot_spectra(files) takes a list of filenames (.dat) and plots the spectra
#Alternatively, if files is an integer, use basename *mxxx_*.dat
class plot_spectra():

    def __init__(self,files,*data_channels):

        self.sweeps=[]

        title='Generic Figure'
        lockin_trig=True
        current_trig=True

        if type(files) == list:
            file_list = files
        elif type(files) == int:
            file_string = '*m' + '%0*d' % (3, files) + '_*.dat'
            file_list = glob.glob(file_string)
            title='m' + '%0*d' % (3, files) + '.sxm'
            disp=sxm.plot_sxm(files,'Z')

        #Plots dI/dV spectroscopy
        for filename in file_list:
            spectrum=get_spectra(filename)
            self.sweeps.append(spectrum)
            file_index=file_list.index(filename)
            if type(files) == int:
                x_point=spectrum.header['x (nm)'] - disp.header['x_center (nm)'] + disp.header['x_range (nm)']/2.
                y_point=spectrum.header['y (nm)'] - disp.header['y_center (nm)'] + disp.header['y_range (nm)']/2.
                disp.plot.text(x_point,y_point,str(file_index+1))
                disp.figure.canvas.draw()
            if 'Lock In X (V)' or 'Lock In X [AVG] (V)' in spectrum.header['data info']:
                if 'Lock In X (V)' in spectrum.header['data info']:
                    lockin_sweeps=self.sweeps[file_index].data['Lock In X (V)']
                else:
                    lockin_sweeps=self.sweeps[file_index].data['Lock In X [AVG] (V)']
                if lockin_trig:
                    lockin_figure,lockin_ax=matplotlib.pyplot.subplots()
                    lockin_trig=False
                    lockin_lines=[]
                lockin_tmp=lockin_ax.plot(self.sweeps[file_index].data['Bias calc (V)'],lockin_sweeps,label=str(file_index+1))
                lockin_lines.append(lockin_tmp[0])
                lockin_legend=lockin_ax.legend(loc='upper left', fancybox=True, shadow=True)
                lockin_figure.canvas.draw()
            if 'Current (A)' or 'Current [AVG] (A)' in spectrum.header['data info']:
                if 'Current (A)' in spectrum.header['data info']:
                    current_sweeps=self.sweeps[file_index].data['Current (A)']*1e9
                else:
                    current_sweeps=self.sweeps[file_index].data['Current [AVG] (A)']*1e9
                if current_trig:
                    current_figure=matplotlib.pyplot.figure()
                    current_trig=False
                current_plot=current_figure.add_subplot(111)
                current_plot.plot(self.sweeps[file_index].data['Bias calc (V)'],current_sweeps,label=str(file_index+1))
                current_legend=current_plot.legend(loc='upper left', fancybox=True, shadow=True)
                current_figure.canvas.draw()
        for channel in data_channels:
            channel_figure=matplotlib.pyplot.figure()
            for filename in file_list:
                spectrum=get_spectra(filename)
                matplotlib.pyplot.plot(spectrum.data['Bias calc (V)'],spectrum.data[channel])
            matplotlib.pyplot.xlabel('Sample Bias (V)')
            matplotlib.pyplot.title(title)
            matplotlib.pyplot.draw()

        #Adds axis labels
        if not lockin_trig:
            lockin_ax.set_xlabel('Sample Bias (V)')
            lockin_ax.set_ylabel('Unnormalized dI/dV')
            lockin_ax.set_title(title)
            lined=dict()
            #This part comes from
            #http://matplotlib.org/examples/event_handling/legend_picking.html
            for legline, origline in zip(lockin_legend.get_lines(),lockin_lines):
                legline.set_picker(5)
                lined[legline]=origline
            def onpick(event):
                legline=event.artist
                origline=lined[legline]
                vis=not origline.get_visible()
                origline.set_visible(vis)
                if vis:
                    legline.set_alpha(1.0)
                else:
                    legline.set_alpha(0.2)
                lockin_figure.canvas.draw()
            lockin_figure.canvas.mpl_connect('pick_event',onpick)
            lockin_figure.show()
        if not current_trig:
            current_plot.set_xlabel('Sample Bias (V)')
            current_plot.set_ylabel('Current (nA)')
            current_plot.set_title(title)
            current_figure.show()
          
        return

