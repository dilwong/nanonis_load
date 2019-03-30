#Imports Nanonis .sxm files into Python

import numpy as np

import matplotlib.pyplot as plt

import scipy.signal

#Loads .sxm files into Nanonis
class sxm():

    def __init__(self, filename):

        self.data = {}
        self.header = {}

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
            if ':SCANIT_END:' in header_text:
                break
        header_text = header_text.split('\n')
        for header_line in header_text:
            if ':' in header_line:
                prev_header = header_line
                self.header[header_line] = []
            else:
                self.header[prev_header].append(header_line)
        temp = self.header[':SCAN_PIXELS:'][0].strip().split()
        self.header['x_pixels'] = int(temp[0])
        self.header['y_pixels'] = int(temp[1])
        temp = self.header[':SCAN_RANGE:'][0].strip().split()
        self.header['x_range (nm)'] = float(temp[0])*1e9
        self.header['y_range (nm)'] = float(temp[1])*1e9
        temp = self.header[':SCAN_OFFSET:'][0].strip().split()
        self.header['x_center (nm)'] = float(temp[0])*1e9
        self.header['y_center (nm)'] = float(temp[1])*1e9
        temp = self.header[':SCAN_ANGLE:'][0].strip().split()
        self.header['angle'] = float(temp[0]) #Clockwise
        self.header['direction'] = self.header[':SCAN_DIR:'][0]
        temp = [chnls.split('\t') for chnls  in self.header[':DATA_INFO:'][1:-1]] # Will this handle multipass?
        self.header['channels'] = [chnls[2].replace('_', ' ') + ' (' + chnls[3] + ')' for chnls in temp]

        raw_data = file[idx+5:]
        #size_in_bytes = 4 * self.header['x_pixels'] * self.header['y_pixels']
        #raw_data = list(zip(*[iter(raw_data)]*size))
        size = self.header['x_pixels'] * self.header['y_pixels']
        raw_data = np.frombuffer(raw_data, dtype='>f')
        for idx, channel_name in enumerate(self.header['channels']):
            channel_data = raw_data[idx*size*2:(idx+1)*size*2]
            self.data[channel_name] = [channel_data[0:size].reshape(self.header['x_pixels'], self.header['y_pixels'])]
            self.data[channel_name].append(channel_data[size:2*size].reshape(self.header['x_pixels'], self.header['y_pixels']))

#direction = 0 for forward, direction = 1 for backwards
class plot():

    #This class is named after plot_sxm.m
    def __init__(self, sxm_data, channel, direction = 0, flatten = True):

        self.data = sxm_data

        image_data = np.copy(sxm_data.data[channel][direction])
        avg_dat = image_data[~np.isnan(image_data)].mean()
        image_data[np.isnan(image_data)] = avg_dat
        if flatten:
            image_data=scipy.signal.detrend(image_data)

        #Flip upside down if image was taken scanning down
        if sxm_data.header['direction'] == 'down':
            image_data=np.flipud(image_data)

        #Flip left to right if backwards scan
        if direction:
            image_data=np.fliplr(image_data)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        x_range = sxm_data.header['x_range (nm)']
        y_range = sxm_data.header['y_range (nm)']
        x_pixels = sxm_data.header['x_pixels']
        y_pixels = sxm_data.header['y_pixels']
        y, x = np.mgrid[0:x_range:x_pixels*1j,0:y_range:y_pixels*1j]
        self.pcolor = self.ax.pcolormesh(x, y, image_data, cmap = 'copper')
        self.fig.colorbar(self.pcolor, ax = self.ax)
        self.image_data = image_data

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

    def clim(self, c_min, c_max):
        self.pcolor.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.pcolor.set_cmap(cmap)

    
    # TO DO: Add feature such that clicking a red X shows spectrum.  Or hover over X to show spectrum.
    def add_spectra(self, spectra, labels):
        theta = np.radians(self.data.header['angle'])
        R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        try:
            spectra_iterator = iter(spectra)
            label_iterator = iter(labels)
        except TypeError:
            spectra_iterator = iter([spectra])
            label_iterator = iter([labels])
        for spectrum_inst, label_inst in zip(spectra_iterator, label_iterator):
            spectrum_to_center = (spectrum_inst.header['x (nm)'] - self.data.header['x_center (nm)'], spectrum_inst.header['y (nm)'] - self.data.header['y_center (nm)'])
            spectrum_to_center = R.dot(spectrum_to_center)
            x = spectrum_to_center[0] + self.data.header['x_range (nm)']/2
            y = spectrum_to_center[1] + self.data.header['y_range (nm)']/2
            self.ax.scatter(x, y, marker='x', color='red')
            self.ax.text(x, y, label_inst, fontsize = 10)
    
    def fft(self):
        self.fft_fig = plt.figure()
        self.fft_ax = self.fft_fig.add_subplot(111)
        fft_array = np.absolute(np.fft.fft2(self.image_data))
        max_fft = np.max(fft_array[1:-1,1:-1])
        fft_array = np.fft.fftshift(fft_array)
        fft_x = -np.pi/self.data.header['x_range (nm)']
        fft_y = np.pi/self.data.header['y_range (nm)']
        self.fft_plot = self.fft_ax.imshow(fft_array, extent=[fft_x, -fft_x, -fft_y, fft_y],origin='lower')
        self.fft_fig.colorbar(self.fft_plot, ax = self.fft_ax)
        self.fft_clim(0,max_fft)

    def fft_clim(self, c_min, c_max):
        self.fft_plot.set_clim(c_min, c_max)

    def fft_colormap(self, cmap):
        self.fft_plot.set_cmap(cmap)