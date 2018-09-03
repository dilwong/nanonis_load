#Imports Grid Spectroscopy into Python

import numpy
import os.path
import glob
import re

import matplotlib
import matplotlib.pyplot

import scipy
import scipy.signal

#load_nanonis_3ds(filename) loads Nanonis 3ds files int Python
class load_nanonis_3ds():

    def __init__(self,filename):

        self.header={}
        self.data={}

        with open(filename,'rb') as f:
            s1=f.readline().strip()
            s1_temp=re.split(' |"',s1)
            self.header['x_pixels']=int(s1_temp[2])
            self.header['y_pixels']=int(s1_temp[4])
            s1=f.readline().strip()
            s1_temp=re.split(';|=',s1)
            self.header['x_center (nm)']=float(s1_temp[1])*1e9
            self.header['y_center (nm)']=float(s1_temp[2])*1e9
            self.header['x_size (nm)']=float(s1_temp[3])*1e9
            self.header['y_size (nm)']=float(s1_temp[4])*1e9
            s1=f.readline().strip()
            s1_temp=re.split('=',s1)
            if s1_temp[0] == 'Filetype':
                s1=f.readline().strip()
            s1=f.readline().strip()
            s1=f.readline().strip()
            s1=f.readline().strip()
            s1_temp=re.split('=',s1)
            self.header['n_parameters']=int(s1_temp[1])
            s1=f.readline().strip() #This line reads "Experiment Size"
            s1=f.readline().strip() #This line read Points/Spectra
            s1_temp=re.split('=',s1)
            self.header['points']=int(s1_temp[1])
            s1=f.readline().strip()
            s1_temp=re.split('"|;',s1)
            channels=s1_temp[1:len(s1_temp)-1]
            while s1 != ':HEADER_END:':
                s1=f.readline().strip()
            raw_data=f.read()
            bpp=self.header['points']*len(channels)+self.header['n_parameters']
            data_pts=self.header['x_pixels']*self.header['y_pixels']*bpp
            numerical_data=numpy.frombuffer(raw_data, dtype='>f')
            self.header['Start Bias (V)']=numerical_data[0]
            self.header['End Bias (V)']=numerical_data[1]

        predata=[[{} for y in range(self.header['y_pixels'])] \
                   for x in range(self.header['x_pixels'])]
        for i in range(self.header['x_pixels']):
            for j in range(self.header['y_pixels']):
                for k in range(len(channels)):
                    start_index=(i*self.header['y_pixels']+j) * \
                                 bpp+self.header['n_parameters']+k*self.header['points']
                    end_index=start_index+self.header['points']
                    if numerical_data[start_index:end_index] != []:
                        predata[i][j][channels[k]]=numerical_data[start_index:end_index]
                    else:
                        predata[i][j][channels[k]]=numpy.zeros(end_index-start_index)
        self.energy=numpy.linspace(self.header['Start Bias (V)'], \
                                   self.header['End Bias (V)'],self.header['points'])
        for ty in channels:
            self.data[ty]=numpy.array([[predata[x][y][ty] \
                                        for y in range(self.header['y_pixels'])] \
                                        for x in range(self.header['x_pixels'])])

        return

#Plot Nanonis 3ds File
#Use arrow keys to change energy and color scheme
#Left drag moves the maximum and minimum of scale
#Right drag changes the range of scale
class plot_nanonis_3ds():

    def __init__(self,filename):

        tmp=load_nanonis_3ds(filename)
        self.header=tmp.header
        self.data=tmp.data['Lock In X (V)']
        self.energy=tmp.energy

        #Load and Plot Data
        #Interactive colorbar code borrowed from
        #ster.kuleuven.be/~pieterd/python/html/plotting/interactive_colorbar.html
        x_size=tmp.header['x_size (nm)']
        y_size=tmp.header['x_size (nm)']
        self.fig=matplotlib.pyplot.figure()
        pl=self.fig.add_subplot(111)
        ax=pl.imshow(numpy.flipud(self.data[:,:,0]),extent=[0,x_size,0,y_size])
        pl.set_xlabel('X (nm)')
        pl.set_ylabel('Y (nm)')
        cbar=matplotlib.pyplot.colorbar(ax)
        self.cycle = sorted([i for i in dir(matplotlib.pyplot.cm) \
                             if hasattr(getattr(matplotlib.pyplot.cm,i),'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)
        self.free=0
        title='Color Scheme = ' + self.cycle[self.index] + ', Energy = ' + \
            str(self.energy[self.free]) + ' eV'
        ax.get_axes().set_title(title)
        self.press=None
        def key_press(event):
            if event.key[0:4] == 'alt+':
                key=event.key[4:]
            else:
                key=event.key
            #color_schemes=['jet','pink','gray','Accent','BuGn', \
            #               'BuPu','PiYG','Purples','Greens', \
            #               'Spectral','cool','copper','hot', \
            #               'spectral','spring','summer','winter']
            if key == 'right':
                self.index += 1
            elif key == 'left':
                self.index -= 1
            if self.index < 0:
                self.index = len(self.cycle)-1
            elif self.index >= len(self.cycle):
                self.index = 0
            cmap = self.cycle[self.index]
            cbar.set_cmap(cmap)
            cbar.draw_all()
            ax.set_cmap(cmap)
            if key == 'up':
                self.free -= 1
            elif key == 'down':
                self.free += 1
            if self.free < 0:
                self.free = len(self.energy)-1
            elif self.free >= len(self.energy):
                self.free = 0
            ax.set_data(numpy.flipud(self.data[:,:,self.free]))
            title='Color Scheme = ' + cmap + ', Energy = ' + \
            str(self.energy[self.free]) + ' eV'
            ax.get_axes().set_title(title)
            cbar.patch.figure.canvas.draw()
        def on_press(event):
            if event.inaxes != cbar.ax:
                return
            self.press = event.x, event.y
        def on_motion(event):
            if self.press is None: return
            if event.inaxes != cbar.ax: return
            xprev, yprev = self.press
            dx = event.x - xprev
            dy = event.y - yprev
            self.press = event.x,event.y
            scale = cbar.norm.vmax - cbar.norm.vmin
            perc = 0.03
            if event.button==1:
                cbar.norm.vmin -= (perc*scale)*numpy.sign(dy)
                cbar.norm.vmax -= (perc*scale)*numpy.sign(dy)
            elif event.button==3:
                cbar.norm.vmin -= (perc*scale)*numpy.sign(dy)
                cbar.norm.vmax += (perc*scale)*numpy.sign(dy)
            cbar.draw_all()
            ax.set_norm(cbar.norm)
            cbar.patch.figure.canvas.draw()
        def on_release(event):
            self.press = None
            ax.set_norm(cbar.norm)
            cbar.patch.figure.canvas.draw()
        self.fig.canvas.mpl_connect('button_press_event',on_press)
        self.fig.canvas.mpl_connect('key_press_event',key_press)
        self.fig.canvas.mpl_connect('button_release_event',on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',on_motion)
        self.fig.show()

        return
