#Imports Grid Spectroscopy into Python

import numpy as np
import matplotlib.pyplot as plt

import re

#grid.nanonis_3ds(filename) loads Nanonis 3ds files into Python
class nanonis_3ds():

    def __init__(self, filename):

        with open(filename,'rb') as f:
            file = f.read()

        header_text = ''
        idx = 0
        while True:
            try:
                header_text += chr(file[idx]) # Python 3
            except TypeError:
                header_text += file[idx] # Python 2
            idx += 1
            if ':HEADER_END:' in header_text:
                break
        header_text = header_text.split('\r\n')[:-1]
        self.header = dict()
        for entry in header_text:
            self.header[entry.split('=')[0]] = entry.split('=')[1]

        temp = re.split(' |"', self.header['Grid dim'])
        self.header['x_pixels'] = int(temp[1])
        self.header['y_pixels'] = int(temp[3])
        temp = re.split(';|=', self.header['Grid settings'])
        self.header['x_center (nm)'] = float(temp[1])*1e9
        self.header['y_center (nm)'] = float(temp[2])*1e9
        self.header['x_size (nm)'] = float(temp[3])*1e9
        self.header['y_size (nm)'] = float(temp[4])*1e9
        self.header["n_parameters"] = int(self.header["# Parameters (4 byte)"])
        self.header['points'] = int(self.header['Points'])
        channels = re.split('"|;',self.header['Channels'])[1:-1]

        self.data = {}
        raw_data = file[idx+2:]
        bpp=self.header['points']*len(channels)+self.header['n_parameters']
        data_pts=self.header['x_pixels']*self.header['y_pixels']*bpp
        numerical_data=np.frombuffer(raw_data, dtype='>f')
        self.header['Start Bias (V)']=numerical_data[0]
        self.header['End Bias (V)']=numerical_data[1]

        predata=[[{} for y in range(self.header['y_pixels'])] for x in range(self.header['x_pixels'])]
        for i in range(self.header['x_pixels']):
            for j in range(self.header['y_pixels']):
                for k in range(len(channels)):
                    start_index=(i*self.header['y_pixels']+j) * bpp+self.header['n_parameters']+k*self.header['points']
                    end_index=start_index+self.header['points']
                    if numerical_data[start_index:end_index] != []:
                        predata[i][j][channels[k]]=numerical_data[start_index:end_index]
                    else:
                        predata[i][j][channels[k]]=np.zeros(end_index-start_index)
        self.energy=np.linspace(self.header['Start Bias (V)'], self.header['End Bias (V)'],self.header['points'])
        for ty in channels:
            self.data[ty]=np.array([[predata[x][y][ty] for y in range(self.header['y_pixels'])] for x in range(self.header['x_pixels'])])


#Plot Nanonis 3ds data
#Key press UP and DOWN to change energy
#TO DO: Implement Fourier Transform
class plot():

    def __init__(self, nanonis_3ds, channel):

        self.header = nanonis_3ds.header
        self.data = nanonis_3ds.data[channel]
        self.energy = nanonis_3ds.energy

        x_size = self.header['x_size (nm)']
        y_size = self.header['x_size (nm)']
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.plot = self.ax.imshow(np.flipud(self.data[:,:,0]), extent=[0,x_size,0,y_size], cmap = 'Blues_r')
        self.ax.set_xlabel('X (nm)')
        self.ax.set_ylabel('Y (nm)')
        self.fig.colorbar(self.plot, ax = self.ax)
        self.free=0
        title = 'Energy = ' + str(self.energy[self.free]) + ' eV'
        self.ax.set_title(title)

        def key_press(event):
            if event.key[0:4] == 'alt+':
                key=event.key[4:]
            else:
                key=event.key
            if key == 'up':
                self.free -= 1
            elif key == 'down':
                self.free += 1
            if self.free < 0:
                self.free = len(self.energy)-1
            elif self.free >= len(self.energy):
                self.free = 0
            self.plot.set_data(np.flipud(self.data[:,:,self.free]))
            title='Energy = ' + str(self.energy[self.free]) + ' eV'
            self.ax.set_title(title)
            self.fig.canvas.draw()

        self.fig.canvas.mpl_connect('key_press_event',key_press)

    def clim(self, c_min, c_max):
        self.plot.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.plot.set_cmap(cmap)
