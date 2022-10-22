r'''
Loads and plots Nanonis Grid Spectroscopy (.3ds) data.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import re

try:
    from . import interactive_colorplot
except ImportError:
    import interactive_colorplot

class nanonis_3ds():

    r'''
    grid.nanonis_3ds loads Nanonis .3ds files.

    Args:
        filename : str
    
    Attributes:
        header : dict
            A dictionary containing all of the header information from the .3ds file.
        data : dict
            A dictionary indexed by the data channels.
            The items in the dictionary are numpy arrays containing the numeric data.
        energy : numpy.ndarray
            A numpy array containing the spectroscopy biases.
    '''

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
            entry_array = entry.split('=')
            self.header[entry_array[0]] = entry_array[1]
            if entry_array[0] == 'Comment':
                self.header[entry_array[0]] = '='.join(entry_array[1:])

        temp = re.split(' |"', self.header['Grid dim'])
        self.header['x_pixels'] = int(temp[1])
        self.header['y_pixels'] = int(temp[3])
        temp = re.split(';|=', self.header['Grid settings'])
        self.header['x_center (nm)'] = float(temp[0])*1e9
        self.header['y_center (nm)'] = float(temp[1])*1e9
        self.header['x_size (nm)'] = float(temp[2])*1e9
        self.header['y_size (nm)'] = float(temp[3])*1e9
        self.header['angle'] = float(temp[4])
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

        self.parameters = dict()
        self.parameter_list = []
        for param_name in self.header['Fixed parameters'].strip('"').split(';'):
            self.parameters[param_name] = []
            self.parameter_list.append(param_name)
        for param_name in self.header['Experiment parameters'].strip('"').split(';'):
            self.parameters[param_name] = []
            self.parameter_list.append(param_name)

        predata=[[{} for y in range(self.header['y_pixels'])] for x in range(self.header['x_pixels'])]
        for i in range(self.header['x_pixels']):
            for j in range(self.header['y_pixels']):
                for k in range(len(channels)):
                    start_index=(i*self.header['y_pixels']+j) * bpp+self.header['n_parameters']+k*self.header['points']
                    end_index=start_index+self.header['points']
                    if numerical_data[start_index:end_index] != []:
                        predata[i][j][channels[k]]=numerical_data[start_index:end_index]
                        if k == 0:
                            for idx, param_name in enumerate(self.parameter_list):
                                param_idx = (i*self.header['y_pixels']+j) * bpp + k*self.header['points'] + idx
                                self.parameters[param_name].append(numerical_data[param_idx])
                    else:
                        predata[i][j][channels[k]]=np.zeros(end_index-start_index)
                        if k == 0:
                            for param in self.parameters:
                                self.parameters[param].append(0)
        self.energy=np.linspace(self.header['Start Bias (V)'], self.header['End Bias (V)'],self.header['points'])
        for ty in channels:
            self.data[ty]=np.array([[predata[x][y][ty] for y in range(self.header['y_pixels'])] for x in range(self.header['x_pixels'])])

#TO DO: Copy data to clipboard
class plot():

    r'''
    Plots the 2D grid spectroscopy data.
    Press the keyboard arrow keys (UP and DOWN) to change the bias/energy of the image.

    Args:
        nanonis_3ds : grid.nanonis_3ds
            The grid.nanonis_3ds object that contains the data to be plotted.
        channel : str
            A string specifying which data channel is to be plotted.
        fft : bool (defaults to False)
            If True, plot the Fourier transform of the data.

    Attributes:
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot

    Methods:
        clim(c_min : float, c_max : float) : None
            Set the color axis limits for the real-space image. c_min < c_max
        colormap(cmap) : None
            Change the colormap to cmap for the real-space image, where cmap is an acceptable matplotlib colormap.
        fft_clim(c_min : float, c_max : float) : None
            Set the color axis limits on the Fourier transform. c_min < c_max
        fft_colormap(cmap) : None
            Change the colormap to cmap for the Fourier transform, where cmap is an acceptable matplotlib colormap.
        show_spectra() : None
            If the user has clicked in the real-space image, show_spectra() will plot the spectra corresponding
            to the clicked pixel in a new matplotlib Figure.
    '''

    def __init__(self, nanonis_3ds, channel, fft = False):

        self.header = nanonis_3ds.header
        self.data = nanonis_3ds.data[channel]
        self.energy = nanonis_3ds.energy
        self.channel = channel
        self.press = None
        self.click = None
        self.fft = fft
        self._nanonis_3ds = nanonis_3ds

        x_size = self.header['x_size (nm)']
        y_size = self.header['y_size (nm)']
        if fft:
            self.fig = plt.figure(figsize=[2*6.4, 4.8])
            self.ax = self.fig.add_subplot(121)
            self.fft_ax = self.fig.add_subplot(122)
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        self.plot = self.ax.imshow(np.flipud(self.data[:,:,0]), extent=[0,x_size,0,y_size], cmap = 'Blues_r') # Check to make sure x_size and y_size aren't mixed up
        if fft:
            fft_array = np.absolute(np.fft.fft2(np.flipud(self.data[:,:,0])))
            max_fft = np.max(fft_array[1:-1,1:-1])
            fft_array = np.fliplr(np.fft.fftshift(fft_array))  # Is this the correct orientation?
            fft_x = -np.pi/x_size
            fft_y = np.pi/y_size
            self.fft_plot = self.fft_ax.imshow(fft_array, extent=[fft_x, -fft_x, -fft_y, fft_y], origin='lower')
            self.fft_colorbar = self.fig.colorbar(self.fft_plot, ax = self.fft_ax)
            self.fft_clim(0,max_fft)
        else:
            self.fft_plot = None
        self.ax.set_xlabel('X (nm)')
        self.ax.set_ylabel('Y (nm)')
        self.colorbar = self.fig.colorbar(self.plot, ax = self.ax)
        self.free = 0
        title = 'Energy = ' + str(self.energy[self.free]) + ' eV'
        self.ax.set_title(title)

        def key_press(event):
            if event.key[0:4] == 'alt+':
                key=event.key[4:]
            else:
                key=event.key
            self.increment_energy(key)
        def on_press(event):
            if event.inaxes == self.colorbar.ax:
                self.press = (event.x, event.y, self.colorbar, self.plot)
            elif (self.fft_plot is not None) and (event.inaxes == self.fft_colorbar.ax):
                self.press = (event.x, event.y, self.fft_colorbar, self.fft_plot)
            elif event.inaxes == self.ax:
                self.click = (event.xdata, event.ydata)
            else:
                return
        def on_motion(event):
            if self.press is None:
                return
            colorbar = self.press[2]
            if event.inaxes != colorbar.ax:
                return
            dx = event.x - self.press[0]
            dy = event.y - self.press[1]
            self.press = (event.x, event.y, colorbar, self.press[3])
            scale = colorbar.norm.vmax - colorbar.norm.vmin
            perc = 0.05
            if event.button == 1:
                colorbar.norm.vmin -= (perc * scale) * np.sign(dy)
                colorbar.norm.vmax -= (perc * scale) * np.sign(dy)
            elif event.button == 3:
                colorbar.norm.vmin -= (perc* scale) * np.sign(dy)
                colorbar.norm.vmax += (perc * scale) *np.sign(dy)
            colorbar.draw_all()
            self.press[3].set_norm(colorbar.norm)
            colorbar.patch.figure.canvas.draw()
        def on_release(event):
            self.press = None

        self.key_press = key_press
        self.fig.canvas.mpl_connect('key_press_event',key_press)
        self.fig.canvas.mpl_connect('button_press_event',on_press)
        self.fig.canvas.mpl_connect('motion_notify_event',on_motion)
        self.fig.canvas.mpl_connect('button_release_event',on_release)

    def clim(self, c_min, c_max):
        self.plot.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.plot.set_cmap(cmap)

    def fft_clim(self, c_min, c_max):
        self.fft_plot.set_clim(c_min, c_max)

    def fft_colormap(self, cmap):
        self.fft_plot.set_cmap(cmap)

    # TO DO: Implement channel keyword
    def show_spectra(self, channel = None, ax = None):

        if self.click is None:
            return
        x_pixel = int(np.floor(self.click[0] * self.header['x_pixels'] / self.header['x_size (nm)']))
        y_pixel = int(np.floor(self.click[1] * self.header['y_pixels'] / self.header['y_size (nm)']))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.energy, self.data[y_pixel, x_pixel, :])

        # TO DO: test this
        x = (x_pixel + 0.5) * self.header['x_size (nm)'] / self.header['x_pixels']
        y = (y_pixel + 0.5) * self.header['y_size (nm)'] / self.header['x_pixels']
        theta = -np.radians(self.header['angle'])
        R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        xy_vec = (x - self.header['x_size (nm)'] *  0.5, y - self.header['y_size (nm)'] * 0.5)
        transformed_vec = R.dot(xy_vec)
        transformed_x = transformed_vec[0] + self.header['x_center (nm)']
        transformed_y = transformed_vec[1] + self.header['y_center (nm)']
        print('x = ' + str(transformed_x) + ' nm')
        print('y = ' + str(transformed_y) + ' nm')

    def increment_energy(self, key):
        if key == 'down':
            self.free -= 1
        elif key == 'up':
            self.free += 1
        if self.free < 0:
            self.free = len(self.energy)-1
        elif self.free >= len(self.energy):
            self.free = 0
        self.plot.set_data(np.flipud(self.data[:,:,self.free]))
        if self.fft:
            fft_array = np.absolute(np.fft.fft2(np.flipud(self.data[:,:,self.free])))
            fft_array = np.fliplr(np.fft.fftshift(fft_array)) # Is this the correct orientation?
            self.fft_plot.set_data(fft_array)
        title='Energy = ' + str(self.energy[self.free]) + ' eV'
        self.ax.set_title(title)
        self.fig.canvas.draw()
        return (self.fig, )

    # dpi does not work... why?
    # Needs a MovieWriter
    def animate(self, key, wait_time = 200, filename = None, dpi = 80, writer = 'pillow'):
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(self.fig, self.increment_energy, frames=[key] * (len(self.energy) - 0), interval = wait_time) # Fix ordering
        if filename is not None:
            anim.save(filename, dpi = dpi, writer = writer)

#Loads and plots 3DS line cuts
class linecut(interactive_colorplot.colorplot):

    r'''
    Loads and plots a 1D line cut from a .3ds file as a colorplot with the bias on the
    x-axis and the distance on the y-axis.

    Args:
        filename : str
            The name of the .3ds file to load.
        channel : str
            The name of the channel to plot on the color axis.

    Attributes:
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot

    Methods:
        drag_bar(direction = 'horizontal', locator = False, axes = None, color = None) : interactive_colorplot.drag_bar
            Creates a "drag_bar" object.
        show_image(filename, flatten = True, subtract_plane = False) : None
            Plots an image from a .sxm file, and draws a line on the image indicating the location of the .3ds line cut.
    '''

    def __init__(self, filename, channel):

        interactive_colorplot.colorplot.__init__(self)
        self.sxm_fig = None
        self.sxm_data = None

        #Load data with filename
        self.nanonis_3ds = nanonis_3ds(filename)
        self.n_positions = self.nanonis_3ds.header['x_pixels']
        if self.nanonis_3ds.header['y_pixels'] != 1:
            print('WARNING: ' + filename + ' IS NOT A LINE CUT')
            print('         grid.linecut MAY NOT WORK AS EXPECTED')
        self.n_energies = self.nanonis_3ds.header['points']
        self.bias = self.nanonis_3ds.energy
        self.x_values = np.array(self.nanonis_3ds.parameters['X (m)']) * 1e9
        self.y_values = np.array(self.nanonis_3ds.parameters['Y (m)']) * 1e9
        self.dist = np.sqrt((self.x_values - self.x_values[0]) **2 + (self.y_values - self.y_values[0]) **2)

        self.data = np.array([self.nanonis_3ds.data[channel][site].flatten() for site in range(self.n_positions)])
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Sample Bias (V)')
        self.ax.set_ylabel('Distance (nm)')

        new_bias = (self.bias[1:] + self.bias[:-1]) * 0.5
        new_bias = np.insert(new_bias, 0, self.bias[0] - (self.bias[1] - self.bias[0]) * 0.5)
        new_bias = np.append(new_bias, self.bias[-1] + (self.bias[-1] - self.bias[-2]) * 0.5)
        new_dist = (self.dist[1:] + self.dist[:-1]) * 0.5
        new_dist = np.insert(new_dist, 0, self.dist[0] - (self.dist[1] - self.dist[0]) * 0.5)
        new_dist = np.append(new_dist, self.dist[-1] + (self.dist[-1] - self.dist[-2]) * 0.5)
        x, y = np.meshgrid(new_bias, new_dist)
        x = x.T
        y = y.T
        self.data = self.data.T
        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = 'RdYlBu_r')
        self.fig.colorbar(self.pcolor, ax = self.ax)

        self.show_image_set = False
        self.xlist = self.bias
        self.ylist = self.dist

    def point_transform(self, x, y):
        if self.sxm_data is None:
            return None
        theta = np.radians(self.sxm_data.header['angle'])
        R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        transformed = (x - self.sxm_data.header['x_center (nm)'], y - self.sxm_data.header['y_center (nm)'])
        transformed = R.dot(transformed)
        transformed_x = transformed[0] + self.sxm_data.header['x_range (nm)'] * 0.5
        transformed_y = transformed[1] + self.sxm_data.header['y_range (nm)'] * 0.5
        return (transformed_x, transformed_y)

    def show_image(self, filename, flatten = True, subtract_plane = False):

        try:
            from . import sxm
        except ImportError:
            import sxm

        self.sxm_data = sxm.sxm(filename)
        self.sxm_fig = sxm.plot(self.sxm_data, 'Z (m)', flatten = flatten, subtract_plane = subtract_plane)

        transformed_pts = [self.point_transform(x, y) for x, y in zip(self.x_values, self.y_values)]
        x_values, y_values = zip(*transformed_pts)
        self.transformed_x_values = x_values
        self.transformed_y_values = y_values
        self.sxm_fig.ax.plot(x_values, y_values, color='k')
        self.sxm_fig.ax.set_aspect('equal')
        for bar in self._draggables:
            if bar.direction[0] == 'h':
                bar.sxm_circle = matplotlib.patches.Circle((x_values[bar.index], y_values[bar.index]), radius = 0.5, color = bar.color, zorder = 10)
                self.sxm_fig.ax.add_patch(bar.sxm_circle)
                bar.sxm = self
                try:
                    bar.sxm_dot
                except AttributeError:
                    bar.sxm_dot = True
                    def slide_circle(input_bar = bar):
                        try:
                            input_bar.sxm_circle.center = (input_bar.sxm.transformed_x_values[input_bar.index], input_bar.sxm.transformed_y_values[input_bar.index])
                            input_bar.sxm.sxm_fig.fig.canvas.draw() # TO DO: Blit for speed
                        except KeyError:
                            pass
                    bar.functions.append(slide_circle)
        self.show_image_set = True

    def drag_bar(self, direction = 'horizontal', locator = False, axes = None, color = None):
        
        dbar = super(linecut, self).drag_bar(direction = direction, locator = locator, axes = axes, color = color)

        if (self.sxm_fig is not None) and (dbar.direction[0] == 'h'):
            dbar.sxm_circle = matplotlib.patches.Circle((self.transformed_x_values[dbar.index], self.transformed_y_values[dbar.index]), radius = 0.5, color = dbar.color, zorder = 10)
            self.sxm_fig.ax.add_patch(dbar.sxm_circle)
            dbar.sxm = self
            def slide_circle(input_bar = dbar):
                try:
                    input_bar.sxm_circle.center = (input_bar.sxm.transformed_x_values[input_bar.index], input_bar.sxm.transformed_y_values[input_bar.index])
                    input_bar.sxm.sxm_fig.fig.canvas.draw() # TO DO: Blit for speed
                except KeyError:
                    pass
            dbar.functions.append(slide_circle)
        
        return dbar

def quick_plot(filename, **kwargs):

    loaded_data = nanonis_3ds(filename)
    try:
        return plot(loaded_data, channel='Input 2 (V)', **kwargs)
    except KeyError:
        return plot(loaded_data, channel='Input 2 [AVG] (V)', **kwargs)

# TO DO: Test gap map program.
class gap_map():

    def __init__(self, nanonis_3ds, channel, gap_fit_function):

        self.header = nanonis_3ds.header
        self.spec_data = nanonis_3ds.data[channel]
        self.energy = nanonis_3ds.energy

        self.x_pixels = self.header['x_pixels']
        self.y_pixels = self.header['y_pixels']

        self.gap_data = np.empty([self.x_pixels, self.y_pixels])
        for x_pix in range(self.x_pixels):
            for y_pix in range(self.y_pixels):
                gap_value = gap_fit_function(self.energy, self.spec_data[x_pix, y_pix, :])
                self.gap_data[x_pix, y_pix] = gap_value

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.plot = self.ax.imshow(np.flipud(self.gap_data), extent=[0,self.header['x_size (nm)'],0,self.header['y_size (nm)']], cmap = 'inferno_r')
        self.ax.set_xlabel('X (nm)')
        self.ax.set_ylabel('Y (nm)')
        self.fig.colorbar(self.plot, ax = self.ax)
