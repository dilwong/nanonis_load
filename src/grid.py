#Imports Grid Spectroscopy into Python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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


#Plot Nanonis 3ds data
#Key press UP and DOWN to change energy
class plot():

    def __init__(self, nanonis_3ds, channel, fft = False):

        self.header = nanonis_3ds.header
        self.data = nanonis_3ds.data[channel]
        self.energy = nanonis_3ds.energy

        x_size = self.header['x_size (nm)']
        y_size = self.header['x_size (nm)']
        if fft:
            self.fig = plt.figure(figsize=[2*6.4, 4.8])
            self.ax = self.fig.add_subplot(121)
            self.fft_ax = self.fig.add_subplot(122)
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        self.plot = self.ax.imshow(np.flipud(self.data[:,:,0]), extent=[0,x_size,0,y_size], cmap = 'Blues_r')
        if fft:
            fft_array = np.absolute(np.fft.fft2(np.flipud(self.data[:,:,0])))
            max_fft = np.max(fft_array[1:-1,1:-1])
            fft_array = np.fft.fftshift(fft_array)
            fft_x = -np.pi/x_size
            fft_y = np.pi/y_size
            self.fft_plot = self.fft_ax.imshow(fft_array, extent=[fft_x, -fft_x, -fft_y, fft_y], origin='lower')
            self.fig.colorbar(self.fft_plot, ax = self.fft_ax)
            self.fft_clim(0,max_fft)
        self.ax.set_xlabel('X (nm)')
        self.ax.set_ylabel('Y (nm)')
        self.fig.colorbar(self.plot, ax = self.ax)
        self.free = 0
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
            if fft:
                fft_array = np.absolute(np.fft.fft2(np.flipud(self.data[:,:,self.free])))
                fft_array = np.fft.fftshift(fft_array)
                self.fft_plot.set_data(fft_array)
            title='Energy = ' + str(self.energy[self.free]) + ' eV'
            self.ax.set_title(title)
            self.fig.canvas.draw()

        self.key_press = key_press
        self.fig.canvas.mpl_connect('key_press_event',key_press)

    def clim(self, c_min, c_max):
        self.plot.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.plot.set_cmap(cmap)

    def fft_clim(self, c_min, c_max):
        self.fft_plot.set_clim(c_min, c_max)

    def fft_colormap(self, cmap):
        self.fft_plot.set_cmap(cmap)

#Loads and plots 3DS line cuts
class linecut():

    def __init__(self, filename, channel):

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

        x, y = np.meshgrid(self.bias, self.dist)
        self.pcolor = self.ax.pcolormesh(x, y, self.data, cmap = 'RdYlBu_r')
        self.fig.colorbar(self.pcolor, ax = self.ax)

        self.__draggables__ = []
        self.__drag_h_count__ = 0
        self.__drag_v_count__ = 0
        self.__color_cycle__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.__drag_color_index__ = 0
        self.__drag_rev_legend_map__ = dict()
        self.show_image_set = False

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

    def clim(self, c_min, c_max):
        self.pcolor.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.pcolor.set_cmap(cmap)

    def drag_bar(self, direction = 'horizontal', locator = False):

        drag_index = len(self.__draggables__)
        if direction[0] == 'h':
            if self.__drag_h_count__ == 0:
                self.__drag_h_fig__ = plt.figure()
                self.__drag_h_ax__ = self.__drag_h_fig__.add_subplot(111)
            y_init = self.ax.get_ylim()[1] - (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.1
            line = self.ax.axhline(y_init, color = self.__color_cycle__[self.__drag_color_index__])
            idx, index_value = min(enumerate(self.dist), key = lambda x: abs(x[1] - y_init))
            p_line, = self.__drag_h_ax__.plot(self.bias, self.data[idx,:], label = str(index_value), color = self.__color_cycle__[self.__drag_color_index__])
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
            p_line, = self.__drag_v_ax__.plot(self.dist, self.data[:,idx], label = str(bias_value), color = self.__color_cycle__[self.__drag_color_index__])
            legend = self.__drag_v_ax__.legend()
            count = self.__drag_v_count__
            self.__drag_v_count__ += 1
            if locator:
                v_line = self.__drag_h_ax__.axvline(x_init, color = self.__color_cycle__[self.__drag_color_index__])
        else:
            print('Direction must be "h" for horizontal or "v" for vertical.')
            return
        self.__draggables__.append({'count':count, 'direction': direction[0], 'line':line, 'press':False, 'plot':p_line, 'color':self.__color_cycle__[self.__drag_color_index__], 'index':idx})
        
        if (direction[0] == 'h') and (self.show_image_set == True):
            self.__draggables__[drag_index]['sxm_circle'] = matplotlib.patches.Circle((self.transformed_x_values[idx], self.transformed_y_values[idx]), \
                radius = 0.5, color = self.__color_cycle__[self.__drag_color_index__], zorder = 10)
            self.sxm_fig.ax.add_patch(self.__draggables__[drag_index]['sxm_circle'])
        
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
                idx, index_value = min(enumerate(self.dist), key = lambda x: abs(x[1] - event.ydata))
                self.__draggables__[drag_index]['plot'].set_ydata(self.data[idx,:])
                self.__draggables__[drag_index]['plot'].set_label(str(index_value))
                self.__drag_h_ax__.legend()
                for v in self.__draggables__:
                    v['plot'].axes.get_legend().get_lines()[v['count']].set_visible(True)
                self.__drag_h_fig__.canvas.draw()
                if self.show_image_set == True:
                    try:
                        self.__draggables__[drag_index]['sxm_circle'].center = (self.transformed_x_values[idx], self.transformed_y_values[idx])
                        self.sxm_fig.fig.canvas.draw()
                    except KeyError:
                        pass
            elif direction[0] == 'v':
                self.__draggables__[drag_index]['line'].set_xdata([event.xdata, event.xdata])
                idx, bias_value = min(enumerate(self.bias), key = lambda x: abs(x[1] - event.xdata))
                self.__draggables__[drag_index]['plot'].set_ydata(self.data[:,idx])
                self.__draggables__[drag_index]['plot'].set_label(str(bias_value))
                self.__drag_v_ax__.legend()
                for v in self.__draggables__:
                    v['plot'].axes.get_legend().get_lines()[v['count']].set_visible(True)
                self.__drag_v_fig__.canvas.draw()
            else:
                pass
            self.__draggables__[drag_index]['index'] = idx
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

    def show_image(self, filename):
        
        import sxm

        self.sxm_data = sxm.sxm(filename)
        self.sxm_fig = sxm.plot(self.sxm_data, 'Z (m)')

        def point_transform(x, y):
            theta = np.radians(self.sxm_data.header['angle'])
            R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
            transformed = (x - self.sxm_data.header['x_center (nm)'], y - self.sxm_data.header['y_center (nm)'])
            transformed = R.dot(transformed)
            transformed_x = transformed[0] + self.sxm_data.header['x_range (nm)'] * 0.5
            transformed_y = transformed[1] + self.sxm_data.header['y_range (nm)'] * 0.5
            return (transformed_x, transformed_y)

        transformed_pts = [point_transform(x, y) for x, y in zip(self.x_values, self.y_values)]
        x_values, y_values = zip(*transformed_pts)
        self.transformed_x_values = x_values
        self.transformed_y_values = y_values
        self.sxm_fig.ax.plot(x_values, y_values, color='k')
        self.sxm_fig.ax.set_aspect('equal')
        for draggable in self.__draggables__:
            if draggable['direction'] == 'h':
                draggable['sxm_circle'] = matplotlib.patches.Circle((x_values[draggable['index']], y_values[draggable['index']]), radius = 0.5, color = draggable['color'], zorder = 10)
                self.sxm_fig.ax.add_patch(draggable['sxm_circle'])

        self.show_image_set = True