"""

Plots a dI/dV(Vg, Bz) Landau Fan

Note that this works differently from didv.landau_fan, which takes a list of BASENAMES and magnetic-field values.
This Python module reads the value of the magnetic field directly from the .dat file.

.dat files are basename_LoopIndex_SpectraIndex

Assumes the gate voltages are the same for each loop iteration.

To run:
landauPlot1 = magnetoconductance.landau_fan('BASENAME_') # Loads and plots the Landau Fan
bar1 = landauPlot1.drag_bar(direction='t', color = 'orange') # A mouse-draggable interactive bar
bar2 = landauPlot1.drag_bar(direction='h', error = 0.05) # Default is direction = 'h', error = 0, color = some kind of blue
# It will likely be difficult to find any data using the default error = 0. Use a non-zero, positive value
bar3 = landauPlot1.drag_bar(direction='v', error = 0.1, step = 0.75)
# The drag_bar searches for x/y values within the interval [ bar_location - error, bar_location + error]
# The "step" keyword controls how much bar_location moves per press of the up or down keyboard button
landauPlot1.refresh(10) # Search for new .dat files every 10 seconds to add to the plot.
landauPlot2 = magnetoconductance.landau_fan('BASENAME_', cache = landauPlot1)
# The "cache" keyword allows for a significant speed-up by using spectra already loaded in landauPlot1 instead of reading from disk
# landauPlot2 will still load any spectra on disk that are absent from the cache landauPlot1

"""

import didv
import numpy as np
import numpy.ma as ma
import glob

import time
import traceback
import sys
if sys.version_info.major == 2:
    import thread
    import itertools
elif sys.version_info.major == 3:
    import _thread as thread
else:
    print('Unknown Python version')

# from scipy.spatial import Voronoi, voronoi_plot_2d

# import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# try:
#     from . import interactive_colorplot
# except (ImportError, ValueError):
#     import interactive_colorplot

class landau_fan():

    def __init__(self, *basenames, **kwargs):

        self.basenames = basenames
        self.bias = kwargs['bias'] if ('bias' in kwargs) else 0
        self.channel = kwargs['channel'] if ('channel' in kwargs) else 'Input 2 (V)'
        self.rasterized = kwargs['rasterized'] if ('rasterized' in kwargs) else False
        cmap = kwargs['cmap'] if ('cmap' in kwargs) else 'RdBu'
        cache = kwargs['cache'] if ('cache' in kwargs) else None
        axes = kwargs['axes'] if ('axes' in kwargs) else None

        self.colormap(cmap, set = False)
        self.terminate = False
        self.__draggables__ = []
        self.__moving__ = False
        self.__updating__ = False
        # self.__lock__ = thread.allocate_lock()
        self.__lock__ = self
        
        self.__load_data__(cache = cache)

        if axes is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = axes.figure
            self.ax = axes

        # # Voronoi Tessellation
        # self.pointGrid = [(s.gate, s.Bz) for s in self.spectra_list]
        # self.values = [s.didv_value for s in self.spectra_list]
        # max_val = max(self.values)
        # min_val = min(self.values)
        # norm = matplotlib.colors.Normalize(vmin = min_val, vmax = max_val, clip = True)
        # mapping = cm.ScalarMappable(norm = norm, cmap = cmap)
        # 
        # pointGrid_temp = self.pointGrid[:]
        # pointGrid_temp.append([-999, -999])
        # pointGrid_temp.append([-999, 999])
        # pointGrid_temp.append([999, -999])
        # pointGrid_temp.append([999, 999])
        # self.vor = Voronoi(pointGrid_temp) # Uses the Euclidean metric
        # 
        # for idx in range(self.nSpectra):
        #     vor_idx = self.vor.point_region[idx]
        #     vor_region = self.vor.regions[vor_idx]
        #     if (len(vor_region) != 0) and (-1 not in vor_region):
        #         polygon = self.vor.vertices[vor_region]
        #         self.ax.fill(*zip(*polygon), color = mapping.to_rgba(self.values[idx]))

        # # Use Delaunay Triangulation
        # xt, yt, zt, _, _ = zip(*self.data)
        # self.plot = self.ax.tripcolor(xt, yt, zt, cmap = self.cmap, rasterized = self.rasterized)

        x_temp, y_temp = self.__mesh__()
        self.pcolor = self.ax.pcolormesh(x_temp, y_temp, self.z, cmap = self.cmap, rasterized = self.rasterized)
        self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)

    def __load_data__(self, cache = None):
        
        fileList = []
        for basename in self.basenames:
            file_string = basename + '*.dat'
            fileList += glob.glob(file_string)

        if cache is None:
            self.spectra_list = []
            self.files = []
            cacheFiles = []
            cacheSpectra = []
        else:
            if cache == 'update': # If __load_data is called by self.update()
                cacheFiles = self.files[:]
                cacheSpectra = self.spectra_list[:]
            else:
                cacheFiles = cache.files
                cacheSpectra = cache.spectra_list
            if sys.version_info.major == 2:
                cacheIterator = itertools.izip(cacheFiles, cacheSpectra)
            else:
                cacheIterator = zip(cacheFiles, cacheSpectra)
            self.spectra_list = []
            self.files = []
            for filename, spec in cacheIterator: # Assumes cache.files and cache.spectra_list are in the same order
                if filename in fileList: # This is to remove spectra not found by glob.glob(file_string)
                    self.spectra_list.append(spec)
                    self.files.append(filename)
        
        for filename in fileList: # Insert all spectra not cached into self.files and self.spectra_list
            if filename not in cacheFiles:
                self.files.append(filename)
                spec = didv.spectrum(filename)
                spec.Bz = float(spec.header['Magnetic Field Z (T)'])
                bias_list = spec.data['Bias calc (V)']
                bias_index = min(range(len(bias_list)), key = lambda idx: abs(bias_list[idx] - self.bias))
                spec.didv_value = spec.data[self.channel][bias_index]
                spec.gate = round(spec.gate, 5) # Avoid unlikely floating-point precision issues
                spec.loop_idxs = [int(n) for n in spec.__filename__.split('.')[0].split('_')[-2:]]
                self.spectra_list.append(spec)
        self.nSpectra = len(self.spectra_list)
        self.data = sorted([(s.gate, s.Bz, s.didv_value, s.loop_idxs[0], s.loop_idxs[1]) for s in self.spectra_list])

        # Construct self.x, self.y, and self.z as rectangular arrays with the same dimensions
        x_values = sorted(list(set([elem[0] for elem in self.data])))
        self.x = []
        self.y = []
        self.z = []
        maxnBs = 0
        for gateVoltage in x_values:
            magFields = [elem[1] for elem in self.data if elem[0] == gateVoltage]
            condValues = [elem[2] for elem in self.data if elem[0] == gateVoltage]
            nBs = len(magFields)
            if nBs > maxnBs:
                maxnBs = nBs
            self.x.append([gateVoltage] * nBs)
            self.y.append(magFields)
            self.z.append(condValues)
        for idx, _ in enumerate(self.x):
            nBs = len(self.x[idx])
            if nBs != maxnBs:
                deltaY = self.y[idx][-1] - self.y[idx][-2]
                for r in range(maxnBs - nBs):
                    self.x[idx].append(self.x[idx][0])
                    self.y[idx].append(self.y[idx][-1] + deltaY)
                    self.z[idx].append(np.nan)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = ma.masked_invalid(self.z)

    # Construct the array with quadrilateral vertices such that self.x and self.y are contained in the quadrilaterals
    def __mesh__(self):
        x_temp = (self.x[1:,:]+self.x[:-1,:])*0.5
        x_temp = np.insert(x_temp, 0, self.x[0,:]-0.5*(self.x[1,:]-self.x[0,:]), axis = 0)
        x_temp = np.append(x_temp, np.reshape(self.x[-1,:]+0.5*(self.x[-1,:]-self.x[-2,:]), (1, -1)), axis = 0)
        x_temp = (x_temp[:,1:]+x_temp[:,:-1])*0.5
        x_temp = np.insert(x_temp, 0, x_temp[:,0]-(x_temp[:,1]-x_temp[:,0]), axis = 1)
        x_temp = np.append(x_temp, np.reshape(x_temp[:,-1]+(x_temp[:,-1]-x_temp[:,-2]), (-1, 1)), axis = 1)
        y_temp = (self.y[1:,:]+self.y[:-1,:])*0.5
        y_temp = np.insert(y_temp, 0, self.y[0,:]-0.5*(self.y[1,:]-self.y[0,:]), axis = 0)
        y_temp = np.append(y_temp, np.reshape(self.y[-1,:]+0.5*(self.y[-1,:]-self.y[-2,:]), (1, -1)), axis = 0)
        y_temp = (y_temp[:,1:]+y_temp[:,:-1])*0.5
        y_temp = np.insert(y_temp, 0, y_temp[:,0]-(y_temp[:,1]-y_temp[:,0]), axis = 1)
        y_temp = np.append(y_temp, np.reshape(y_temp[:,-1]+(y_temp[:,-1]-y_temp[:,-2]), (-1, 1)), axis = 1)
        return (x_temp, y_temp)
    
    def xlim(self, x_min, x_max):
        try:
            self.__lock__.acquire()
            self.ax.set_xlim(x_min, x_max)
        except:
            err_detect = traceback.format_exc()
            print(err_detect)
            raise
        finally:
            self.__lock__.release()

    def ylim(self, y_min, y_max):
        try:
            self.__lock__.acquire()
            self.ax.set_ylim(y_min, y_max)
        except:
            err_detect = traceback.format_exc()
            print(err_detect)
            raise
        finally:
            self.__lock__.release()

    def clim(self, c_min, c_max):
        try:
            self.__lock__.acquire()
            self.pcolor.set_clim(c_min, c_max)
        except:
            err_detect = traceback.format_exc()
            print(err_detect)
            raise
        finally:
            self.__lock.release()

    def colormap(self, cmap, set = True):
        if type(cmap) == np.ndarray:
            self.cmap = matplotlib.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap
        if set:
            self.pcolor.set_cmap(self.cmap)

    def acquire(self): # Dummy method that does nothing
        pass

    def release(self): # Dummy method that does nothing
        pass

    def update_loop(self, wait_time):

        while not self.terminate:
            time.sleep(wait_time)
            while(self.__moving__):
                time.sleep(0.5)
            self.update()

    def refresh(self, wait_time = 10):

        def handle_close(event):
            self.terminate = True
        self.fig.canvas.mpl_connect('close_event', handle_close)

        thread.start_new_thread(self.update_loop, (wait_time, ))

    def update(self):
        try:
            self.__lock__.acquire()
            self.__updating__ = True
            self.__load_data__(cache = 'update')
            x_temp, y_temp = self.__mesh__()
            clim_min, clim_max = self.pcolor.get_clim()
            self.colorbar.remove()
            self.pcolor.remove()
            self.pcolor = self.ax.pcolormesh(x_temp, y_temp, self.z, cmap = self.cmap, rasterized = self.rasterized)
            self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
            self.fig.canvas.draw()
        except:
            err_detect = traceback.format_exc()
            print(err_detect)
            raise
        finally:
            self.__updating__ = False
            self.__lock__.release()

    def filter_data(self, axis, value, error):
        if (axis[0].upper() == 'X') or (axis[0].upper() == 'V'):
            f_idx = 0
            vary_idx = 1
        elif (axis[0].upper() == 'Y') or (axis[0].upper() == 'B'):
            f_idx = 1
            vary_idx = 0
        else:
            print('Unknown axis to filter along')
            return
        filtered = [elem for elem in self.data if abs(elem[f_idx] - value) <= error]
        return sorted(filtered, key = lambda elem: elem[vary_idx])

    def drag_bar(self, direction = 'h', axes = None, color = '#1f77b4', initial_value = 0, step = None, error = 0):
        
        if (direction[0] == 'h') or (direction[0] == 't'):
            initial_value = self.ax.get_ylim()[1] - (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.1
        elif direction[0] == 'v':
            initial_value = self.ax.get_xlim()[0] + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.1
        
        return drag_bar(self, direction = direction, axes = axes, color = color, initial_value = initial_value, step = step, error = error)

class drag_bar():

    def __init__(self, parent, direction = 'h', axes = None, color = '#1f77b4', initial_value = 0, step = None, error = 0, marker = True):
        
        self.parent = parent
        self.parent.__draggables__.append(self)
        self.parent.fig.active_drag_bar = self
        self.direction = direction
        self.color = color
        self.current_value = initial_value
        self.error = error

        self.waiting = False
        self.press = False
        self.__autoscale__ = False

        try:
            self.__lock__ = self.parent.__lock__
            self.__lock__.acquire()

            self.typicalDeltaX = np.mean(self.parent.x[1:,:]-self.parent.x[:-1,:])
            self.typicalDeltaY = np.mean(self.parent.y[:,1:]-self.parent.y[:,:-1])
            if step is None:
                if (self.direction[0] == 'h') or (self.direction[0] == 't'):
                    self.step = self.typicalDeltaY
                elif self.direction[0] == 'v':
                    self.step = self.typicalDeltaX
            else:
                self.step = step
            self.minX = np.min(self.parent.x)
            self.maxX = np.max(self.parent.x)
            self.minY = np.min(self.parent.y)
            self.maxY = np.max(self.parent.y)

            if axes is None:
                self.drag_fig = plt.figure()
                self.drag_ax = self.drag_fig.add_subplot(111)
            else:
                self.drag_ax = axes
                self.drag_fig = axes.figure

            indepVar = []
            dependVar = []
            ignoredVar = []
            if self.direction[0] == 'h':
                # axline_function = self.parent.ax.axhline
                data = self.parent.filter_data('Y', self.current_value, self.error)
                if len(data) != 0:
                    indepVar, ignoredVar, dependVar, _, _ = zip(*data)
                    self.parent_line = self.parent.ax.plot(indepVar, ignoredVar, color = self.color)[0]
                else:
                    self.parent_line = self.parent.ax.plot([self.minX, self.maxX], [self.current_value, self.current_value], color = self.color)[0]
            elif self.direction[0] == 'v':
                # axline_function = self.parent.ax.axvline
                data = self.parent.filter_data('X', self.current_value, self.error)
                if len(data) != 0:
                    ignoredVar, indepVar, dependVar, _, _ = zip(*data)
                    self.parent_line = self.parent.ax.plot(ignoredVar, indepVar, color = self.color)[0]
                else:
                    self.parent_line = self.parent.ax.plot([self.current_value, self.current_value], [self.minY, self.maxY], color = self.color)[0]
            elif self.direction[0] == 't':
                # axline_function = self.parent.ax.axhline
                nearestY_tuple = min(self.parent.data, key = lambda elem: abs(elem[1] - self.current_value))
                nearestY = nearestY_tuple[3]
                data = sorted([elem for elem in self.parent.data if elem[3] == nearestY], key = lambda elem: elem[0])
                if len(data) != 0:
                    indepVar, ignoredVar, dependVar, _, _ = zip(*data)
                    self.parent_line = self.parent.ax.plot(indepVar, ignoredVar, color = self.color)[0]
                else:
                    self.parent_line = self.parent.ax.plot([self.minX, self.maxX], [self.current_value, self.current_value], color = self.color)[0]
            else:
                print('Direction must be "h" for horizontal, "v" for vertical, or "t" for tilted.')
                return
            if len(data) == 0:
                legendLabel = str(self.current_value)
            else:
                minIgnoredVar = min(ignoredVar)
                maxIgnoredVar = max(ignoredVar)
                legendLabel = str(minIgnoredVar) + ' to ' + str(maxIgnoredVar)

            self.plot, = self.drag_ax.plot(indepVar, dependVar, label = legendLabel, color = self.color)
            if marker:
                self.plot.set_marker('.')
                self.plot.set_markerfacecolor('black')
                self.plot.set_markeredgewidth('0')
            self.legend = self.drag_ax.legend()

            # self.parent_line = axline_function(self.current_value, color = self.color)
            self.parent_line.set_pickradius(5)

            def on_press(event):
                if self.waiting:
                    return
                if event.inaxes != self.parent.ax:
                    return
                contains, _ = self.parent_line.contains(event)
                if not contains:
                    return
                self.press = True
                self.parent.fig.active_drag_bar = self
                self.parent.__moving__ = True

            def on_motion(event):
                if self.waiting:
                    return
                if event.inaxes != self.parent.ax:
                    return
                if self.press is False:
                    return
                if (self.direction[0] == 'h') or (self.direction[0] == 't'):
                    self.parent.__moving__ = True
                    self.move_to(value = event.ydata)
                elif self.direction[0] == 'v':
                    self.parent.__moving__ = True
                    self.move_to(value = event.xdata)

            def on_release(event):
                try:
                    if self.waiting:
                        return
                    self.press = False
                    while self.parent.__updating__:
                        time.sleep(0.1)
                    self.parent.fig.canvas.draw()
                    self.parent.__moving__ = False
                except:
                    pass
                finally:
                    pass

            def key_press(event):
                if self.parent.fig.active_drag_bar is self:
                    if event.key == 'up':
                        self.parent.__moving__ = True
                        self.move_to(value = self.current_value + self.step)
                    if event.key == 'down':
                        self.parent.__moving__ = True
                        self.move_to(value = self.current_value - self.step)
                    self.parent.__moving__ = False

            self.parent.fig.canvas.mpl_connect('button_press_event', on_press)
            self.parent.fig.canvas.mpl_connect('motion_notify_event', on_motion)
            self.parent.fig.canvas.mpl_connect('button_release_event', on_release)
            self.parent.fig.canvas.mpl_connect('key_press_event', key_press)

        except:
            err_detect = traceback.format_exc()
            print(err_detect)
            raise
        finally:
            self.__lock__.release()

    def move_to(self, value = None):

        if self.parent.__updating__:
            return
        if value is None:
            return
        self.minX = np.min(self.parent.x)
        self.maxX = np.max(self.parent.x)
        self.minY = np.min(self.parent.y)
        self.maxY = np.max(self.parent.y)
        
        try:
            
            self.__lock__.acquire()

            self.current_value = value
            indepVar = []
            dependVar = []
            ignoredVar = []
            if self.direction[0] == 'h':
                # set_data_function = self.parent_line.set_ydata
                data = self.parent.filter_data('Y', self.current_value, self.error)
                if len(data) != 0:
                    indepVar, ignoredVar, dependVar, _, _ = zip(*data)
                    self.parent_line.set_xdata(indepVar)
                    self.parent_line.set_ydata(ignoredVar)
                else:
                    self.parent_line.set_xdata([self.minX, self.maxX])
                    self.parent_line.set_ydata([self.current_value, self.current_value])
            elif self.direction[0] == 'v':
                # set_data_function = self.parent_line.set_xdata
                data = self.parent.filter_data('X', self.current_value, self.error)
                if len(data) != 0:
                    ignoredVar, indepVar, dependVar, _, _ = zip(*data)
                    self.parent_line.set_xdata(ignoredVar)
                    self.parent_line.set_ydata(indepVar)
                else:
                    self.parent_line.set_xdata([self.current_value, self.current_value])
                    self.parent_line.set_ydata([self.minY, self.maxY])
            elif self.direction[0] == 't':
                # set_data_function = self.parent_line.set_ydata
                nearestY_tuple = min(self.parent.data, key = lambda elem: abs(elem[1] - self.current_value))
                nearestY = nearestY_tuple[3]
                self.current_value = nearestY_tuple[1]
                data = sorted([elem for elem in self.parent.data if elem[3] == nearestY], key = lambda elem: elem[0])
                if len(data) != 0:
                    indepVar, ignoredVar, dependVar, _, _ = zip(*data)
                    self.parent_line.set_xdata(indepVar)
                    self.parent_line.set_ydata(ignoredVar)
                else:
                    self.parent_line.set_xdata([self.minX, self.maxX])
                    self.parent_line.set_ydata([self.current_value, self.current_value])
            if len(data) == 0:
                legendLabel = str(self.current_value)
            else:
                minIndepVar = min(indepVar)
                maxIndepVar = max(indepVar)
                minIgnoredVar = min(ignoredVar)
                maxIgnoredVar = max(ignoredVar)
                minDependVar = min(dependVar)
                maxDependVar = max(dependVar)
                legendLabel = str(minIgnoredVar) + ' to ' + str(maxIgnoredVar)
            
            # set_data_function([self.current_value, self.current_value])
            self.parent.fig.canvas.draw()

        except:
            # err_detect = traceback.format_exc()
            # print(err_detect)
            # raise
            pass
        finally:
            self.__lock__.release()

        self.plot.set_xdata(indepVar)
        self.plot.set_ydata(dependVar)
        self.plot.set_label(legendLabel)
        self.drag_ax.legend()
        if (self.__autoscale__) and (len(data) != 0):
            self.xlim(minIndepVar, maxIndepVar)
            self.ylim(minDependVar, maxDependVar)
        self.drag_fig.canvas.draw()

    def xlim(self, x_min, x_max):
        self.drag_ax.set_xlim(x_min, x_max)
        return self

    def ylim(self, y_min, y_max):
        self.drag_ax.set_ylim(y_min, y_max)
        return self

    def autoscale_on(self):
        self.__autoscale__ = True
        return self

    def autoscale_off(self):
        self.__autoscale__ = False
        return self

    def marker(self, symbol = None, color = None, edgewidth = None):
        if symbol is not None:
            self.plot.set_marker(symbol)
        if color is not None:
            self.plot.set_markerfacecolor(color)
        if edgewidth is not None:
            self.plot.set_markeredgewidth(edgewidth)
        self.drag_ax.legend()
        self.drag_fig.canvas.draw()
        return self