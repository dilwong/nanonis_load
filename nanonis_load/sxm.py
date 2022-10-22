r'''
Loads and plots Nanonis .sxm data.
'''

import numpy as np

import matplotlib.pyplot as plt

import scipy.signal

#import traceback

#TO DO: Maybe flip backwards direction?

#Loads .sxm files into Nanonis
class sxm():

    r'''
    Loads data from a Nanonis .sxm file.

    Args:
        filename : str
    
    Attributes:
        header : Dict[str, object]
            A dictionary containing all of the header information from the .sxm file.
        data : Dict[List[numpy.ndarray]]
            A dictionary indexed by the data channels.
            The items in the dictionary are lists of length 2 (for forwards and backwards).
            Each entry in each list is a numpy array that contains the numeric data.
    '''

    def __init__(self, filename):

        self.data = {}
        extra_info = [None, None]
        self.header = sxm_header(filename, extra_info = extra_info)
        file, idx = extra_info

        raw_data = file[idx+5:]
        #size_in_bytes = 4 * self.header['x_pixels'] * self.header['y_pixels']
        #raw_data = list(zip(*[iter(raw_data)]*size))
        size = self.header['x_pixels'] * self.header['y_pixels']
        raw_data = np.frombuffer(raw_data, dtype='>f')
        for idx, channel_name in enumerate(self.header['channels']):
            channel_data = raw_data[idx*size*2:(idx+1)*size*2]
            self.data[channel_name] = [channel_data[0:size].reshape(self.header['y_pixels'], self.header['x_pixels'])]
            self.data[channel_name].append(channel_data[size:2*size].reshape(self.header['y_pixels'], self.header['x_pixels']))

def sxm_header(filename, extra_info = None):
    '''
    Returns the header of an sxm file as a dict
    '''
    if extra_info is None:
        extra_info = [None, None]

    if not filename.endswith('.sxm'):
        return {}
    header = {}

    with open(filename,'rb') as f:
        file = f.read()
    extra_info[0] = file

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
            header[header_line] = []
        else:
            header[prev_header].append(header_line)
    temp = header[':SCAN_PIXELS:'][0].strip().split()
    header['x_pixels'] = int(temp[0])
    header['y_pixels'] = int(temp[1])
    temp = header[':SCAN_RANGE:'][0].strip().split()
    header['x_range (nm)'] = float(temp[0])*1e9
    header['y_range (nm)'] = float(temp[1])*1e9
    temp = header[':SCAN_OFFSET:'][0].strip().split()
    header['x_center (nm)'] = float(temp[0])*1e9
    header['y_center (nm)'] = float(temp[1])*1e9
    temp =header[':SCAN_ANGLE:'][0].strip().split()
    header['angle'] = float(temp[0]) #Clockwise
    header['direction'] = header[':SCAN_DIR:'][0]
    temp = [chnls.split('\t') for chnls  in header[':DATA_INFO:'][1:-1]] # Will this handle multipass?
    header['channels'] = [chnls[2].replace('_', ' ') + ' (' + chnls[3] + ')' for chnls in temp]

    try:
        multipass_header = header[':Multipass-Config:'][0]
        multipass_rows = header[':Multipass-Config:'][1:]
        header['multipass biases'] = []
        for row in multipass_rows:
            header['multipass biases'].append(float(row.split('\t')[6]))
    except (KeyError, IndexError):
        pass
        
    extra_info[1] = idx
    return header

#direction = 0 for forward, direction = 1 for backwards
class plot():

    r'''
    Plots the 2D .sxm data.

    Args:
        sxm_data : sxm.sxm
            The sxm.sxm object that contains the data to be plotted.
        channel : str
            A string specifying which data channel is to be plotted.
        direction : int (defauts to 0)
            If direction is 0, plot the forward direction.
            If direction is 1, plot the backwards direction.
        flatten : bool (defaults to True)
            If True, subtract a linear fit from every fast-scan line.
        subtract_plane : bool (defaults to False)
            If True, fit a 2D plane to the data, and subtract this plane from the data.

    Attributes:
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot

    Methods:
        xlim(x_min : float, x_max : float) : None
            Set the x-axis limits for the real-space image. x_min < x_max
        ylim(y_min : float, y_max : float) : None
            Set the y-axis limits for the real-space image. y_min < y_max
        clim(c_min : float, c_max : float) : None
            Set the color axis limits for the real-space image. c_min < c_max
        colormap(cmap) : None
            Change the colormap to cmap for the real-space image, where cmap is an acceptable matplotlib colormap.
        fft() : None
            Plot the Fourier transform.
        fft_clim(c_min : float, c_max : float) : None
            Set the color axis limits on the Fourier transform. c_min < c_max
        fft_colormap(cmap) : None
            Change the colormap to cmap for the Fourier transform, where cmap is an acceptable matplotlib colormap.
        add_spectra(spectra : List[didv.spectrum]) : None
            Input a list of didv.spectrum objects, and a red 'X' will appear on the real-space image indicating
            the position where each spectrum in the list was acquired. Clicking on a red X will plot the
            'Input 2 (V)' or 'Input 2 [AVG] (V)' channel from the spectrum acquired at that location.
    '''

    def __init__(self, sxm_data, channel, direction = 0, flatten = True, subtract_plane = False):

        self.data = sxm_data

        image_data = np.copy(sxm_data.data[channel][direction])
        avg_dat = image_data[~np.isnan(image_data)].mean()
        image_data[np.isnan(image_data)] = avg_dat
        if (flatten == True) and (subtract_plane == False):
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
        #x = x.T
        #y = y.T
        if subtract_plane == True:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(np.vstack((x.flatten(),y.flatten())).T, image_data.flatten())
            # TO DO: Check for non-square images.  x_pixels and y_pixels may need to be reversed...
            plane = np.reshape(reg.predict(np.vstack((x.flatten(),y.flatten())).T), (x_pixels, y_pixels))
            image_data = image_data - plane
        # shading = 'auto' in the pcolormesh command forces pcolormesh to accept x, y with the same dimensions as image_data.T
        self.pcolor = self.ax.pcolormesh(y, x, image_data.T, cmap = 'copper', shading = 'auto') # pcolormesh chops off last column and row here
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

    def add_spectra(self, spectra, labels = None):

        try:
            from . import didv
        except ImportError:
            try:
                import didv
            except ImportError:
                from nanonis_load import didv

        theta = np.radians(self.data.header['angle'])
        R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        if labels is None:
            labels = [''] * len(spectra) # labels = range(1, len(spectra) + 1)
        try:
            spectra_iterator = iter(spectra)
            label_iterator = iter(labels)
        except TypeError:
            spectra_iterator = iter([spectra])
            label_iterator = iter([labels])
        for spectrum_inst, label_inst in zip(spectra_iterator, label_iterator):
            spectrum_to_center = (spectrum_inst.header['x (nm)'] - self.data.header['x_center (nm)'], spectrum_inst.header['y (nm)'] - self.data.header['y_center (nm)'])
            spectrum_to_center = R.dot(spectrum_to_center)
            x = spectrum_to_center[0] + self.data.header['x_range (nm)'] * 0.5
            y = spectrum_to_center[1] + self.data.header['y_range (nm)'] * 0.5
            s_plt = self.ax.scatter(x, y, marker='x', color='red', picker = True)
            lbl_plt = self.ax.text(x, y, label_inst, fontsize = 10)
            def picker_factory(spec_obj, scatter_plot):
                def on_pick(event):
                    if scatter_plot == event.artist:
                        try:
                            spec_obj.data['Input 2 (V)']
                            didv.plot(spec_obj, channel = 'Input 2 (V)')
                        except KeyError:
                            #err_detect = traceback.format_exc()
                            #print(err_detect)
                            didv.plot(spec_obj, channel = 'Input 2 [AVG] (V)')
                return on_pick
            pick_caller = picker_factory(spectrum_inst, s_plt)
            self.fig.canvas.mpl_connect('pick_event', pick_caller)

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
