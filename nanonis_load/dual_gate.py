r"""
Loading and plotting dI/dV data with two gate voltages.
Tested in Python 2.7 and 3.

To load the data from filename:
    data = dual_gate(filename)

To plot the data (sample bias vs gate vs dI/dV):
    data.plotBiasGate1(gate2) # gate2 is fixed here
    data.plotBiasGate2(gate1) # gate1 is fixed here

To plot the data (gate1 vs gate2 vs dI/dV for a fixed sample bias):
    fixedBiasPlot = data.plotGate1Gate2(data.nearest_bias(bias))
    fixedBiasPlot.linecut() # To create an interactive line profile

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from collections import defaultdict

try:
    from . import didv
except (ImportError, ValueError):
    import didv
try:
    long
except NameError:
    long = int

class dual_gate:

    r"""
    Loads dI/dV data from filename.

    Attributes:
        filename : str
            The name of the file that contains the data.
        spectra_list : list[didv.spectrum]
            A list of dI/dV spectra objects
        gate1 : dict[key, list[didv.spectrum]]
            A dictionary indexed by 'key' that contains lists of dI/dV spectra objects.
            'key' is an immutable object (float or int) that represents a gate voltage.
            For example, 'key' could be the gate voltage itself.
        gate2: dict[key, list[didv.spectrum]]
            Same as gate1 but for the second gate.
        _toKey: callable (float) -> Union[float, int, str]
            A function that maps a gate voltage to a dictionary 'key'
        bias : numpy array
            An array of sample bias voltages.
            It is assumed that all dI/dV spectra are acquired with the same set of sample biases.
        _biasToIndex : dict[float, int]
            A dictionary that maps sample biases to list indices
        _fixedBiasData : list[list[(float, float, float)]]
            A list indexed by sample biases (using _biasToIndex as a map between floating-point sample biases and integer indices).
            The list contains another list containing data for a fixed sample bias.
            (float, float, float) is (first gate, second gate, data from measurement channel - e.g. dI/dV data).
        _dataChannel : str
            The channel with data to plotted, e.g. 'Input 2 (V)'

    Methods:
        plotBiasGate1
        plotBiasGate2
        plotGate1Gate2
        nearest_bias
    """

    def __init__(self, filename):

        self.filename = filename
        self._toKey = lambda x: x # Function to convert gate voltage to dictionary key
        self.spectra_list = didv.parse_arguments(self.filename)
        self.gate1 = defaultdict(list)
        self.gate2 = defaultdict(list)
        self.bias = None
        self._biasToIndex = None
        self._fixedBiasData = None
        self._dataChannel = None
        for idx, spec in enumerate(self.spectra_list):
            spec.gate1 = spec.gate
            del spec.gate
            spec.gate2 = float(spec.header['Gate 2 (V)'])
            self.gate1[self._toKey(spec.gate1)].append(spec)
            self.gate2[self._toKey(spec.gate2)].append(spec)
            spec._dual_gate_order = idx
    
    def _alternateKey(self, gate): # Guard against floating-point comparison errors
        CONVERSIONFACTOR = 1E6 # Convert to microvolts
        return int(CONVERSIONFACTOR * gate)

    def plotBiasGate1(self, gate2, **kwargs):
        r'''
        Plots sample bias on the x-axis, gate1 on the y-axis, and dI/dV on the color axis.

        Arguments:
            gate2:
                The voltage of the second gate.
                If gate2 is a float, plotBiasGate1 will display all data for a fixed second gate equal to gate2.
                Maybe in the future, gate2 can also be a string or callable that filters which voltages on the 
                second gate are looked at.

        Returns:
            didv.colorplot
        '''
        return self._plotBiasGate(gate2, yaxis = 'gate1', **kwargs)

    def plotBiasGate2(self, gate1, **kwargs):
        r'''
        Plots sample bias on the x-axis, gate2 on the y-axis, and dI/dV on the color axis.

        Arguments:
            gate1:
                The voltage of the first gate.
                If gate1 is a float, plotBiasGate2 will display all data for a fixed first gate equal to gate1.
                Maybe in the future, gate1 can also be a string or callable that filters which voltages on the 
                first gate are looked at.
        
        Returns:
            didv.colorplot
        '''
        return self._plotBiasGate(gate1, yaxis= 'gate2', **kwargs)

    def _plotBiasGate(self, gate, yaxis, **kwargs):
        r'''
        Helper method for plotBiasGate1 and plotBiasGate2, since these two methods do the same thing.
        '''
        if yaxis == 'gate1':
            otherGate = 'gate2'
            otherGateDict = self.gate2
            ylabel = 'Gate 1 (V)'
        elif yaxis == 'gate2':
            otherGate = 'gate1'
            otherGateDict = self.gate1
            ylabel = 'Gate 2 (V)'
        else:
            raise Exception('Unknown yaxis parameter')
        if isinstance(gate, (int, long, float)):
            return didv.colorplot(otherGateDict[self._toKey(gate)], yaxis = yaxis, index_label = ylabel, **kwargs)
        elif isinstance(gate, str):
            raise NotImplementedError('Not yet implemented!') # TO DO: Use didv.query or string parsing to specify gate ranges for plotting
        else:
            raise TypeError(otherGate + ' type unrecognized')

    # TO DO: Don't erase the _fixedBiasData when the user decides to do a different channel.
    #        Instead, just cache it somewhere else in case the user decides to return to the original channel.
    def _generateFixedBiasData(self, channel):
        r'''
        Generates _fixedBiasData for plotGate1Gate2.
        '''
        if channel == self._dataChannel:
            return
        if len(self.spectra_list) == 0:
            raise IndexError('No Data!')
        bias_channel = 'Bias calc (V)'
        if self.bias is None:
            self.bias = self.spectra_list[0].data[bias_channel].values
            self._biasToIndex = {bias: idx for idx, bias in enumerate(self.bias)}
        if channel not in self.spectra_list[0].data.columns:
            channel = channel.split()
            channel.insert(-1, '[AVG]')
            channel = ' '.join(channel)
            if channel not in self.spectra_list[0].data.columns:
                raise KeyError('Unknown data channel.')
        if self._fixedBiasData is None:
            self._fixedBiasData = [[] for _ in range(len(self.bias))]
            for spec in self.spectra_list:
                for idx, dataPoint in enumerate(spec.data[channel]):
                    self._fixedBiasData[idx].append((spec.gate1, spec.gate2, dataPoint)) # Maybe use namedtuple
    
    def plotGate1Gate2(self, bias, channel = 'Input 2 (V)', scale1 = 1.0, scale2 = 1.0, shift1 = 0, shift2 = 0):
        r'''
        Plots gate1 on the x-axis, gate2 on the y-axis, and channel on the color axis for a fixed sample bias.

        Arguments:
            bias : float
                The fixed sample bias for which to plot.
            channel : str
                The channel to plot on the color axis.
            scale1 : float
                Scale gate1 by multiplying by scale1
            scale2 : float
                Scale gate2 by multiplying by scale2
            shift1 : float
                Translate gate1 by adding shift1
            shift2 : float
                Translate gate2 by adding shift2

        Return:
            A dual_gate.fixed_bias_plot object.
        '''
        self._generateFixedBiasData(channel)
        return fixed_bias_plot(self, bias, scale1 = scale1, scale2 = scale2, shift1 = shift1, shift2 = shift2)

    def nearest_bias(self, bias):
        r'''
        Since there are floating-point errors, nearest_bias is a helper method that provides the closest sample
        bias to a user-input bias.

        Arguments:
            bias : float
                A sample bias voltage.
        
        Return:
            A float that is the nearest actual sample bias to 'bias'.
        '''
        if self.bias is None:
            if len(self.spectra_list) == 0:
                raise IndexError('No Data!')
            bias_channel = 'Bias calc (V)'
            self.bias = self.spectra_list[0].data[bias_channel].values
            self._biasToIndex = {bias: idx for idx, bias in enumerate(self.bias)}
        return self.bias[np.argmin(np.abs(self.bias - bias))]

class fixed_bias_plot:

    r'''
    A plot of dI/dV(gate1, gate2) for a fixed sample bias voltage returned by dual_gate.dual_gate.plotGate1Gate2.

    Attributes:
        gate1 : numpy array
            A numpy array of voltages from the first gate.
        gate2 : numpy array
            A numpy array of voltages from the second gate.
        bias : float
            The fixed sample bias.
        data : numpy array
            The data being plotted.
        fig : matplotlib figure object
            The figure containing the dI/dV(gate1, gate2) colorplot.
        ax : matplotlib axes object
            The axes containing the dI/dV(gate1, gate2) colorplot.
    '''

    def __init__(self, DG, bias, scale1 = 1.0, scale2 = 1.0, shift1 = 0, shift2 = 0):

        import scipy.interpolate
        self.dualGate = DG
        self.bias = bias

        data = DG._fixedBiasData[DG._biasToIndex[bias]]
        x, y, z = zip(*data)
        #self.gate1 = np.array(list(DG.gate1.keys()))
        #self.gate2 = np.array(list(DG.gate2.keys()))
        self.gate1 = np.array(sorted(DG.gate1))
        self.gate2 = np.array(sorted(DG.gate2)) # Since dictionary keys don't have guaranteed order until Python 3.7
        X, Y = np.meshgrid(self.gate1, self.gate2)
        self.data = scipy.interpolate.griddata(np.array([x,y]).T,np.array(z), (X, Y), method = 'nearest')
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self._scale1 = scale1
        self._scale2 = scale2
        if (scale1 != 1.0) or (scale2 != 1.0) or (shift1 != 0) or (shift2 != 0):
            self.gate1 = scale1 * self.gate1 + shift1
            self.gate2 = scale2 * self.gate2 + shift2
            X, Y = np.meshgrid(self.gate1, self.gate2)
        
        X, Y = _mesh_for_pcolor(X, Y)
        self.pcolor = self.ax.pcolormesh(X, Y, self.data, cmap = 'RdYlBu_r')
        self.colorbar = self.fig.colorbar(self.pcolor, ax = self.ax)
        self.ax.set_xlabel('Gate 1 (V)')
        self.ax.set_ylabel('Gate 2 (V)')

    def xlim(self, x_min, x_max):
        '''Change the x-axis limits'''
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        '''Change the y-axis limits'''
        self.ax.set_ylim(y_min, y_max)

    def clim(self, c_min, c_max):
        '''Change the color-axis limits'''
        self.pcolor.set_clim(c_min, c_max)

    def colormap(self, cmap):
        '''Change the colormap'''
        if type(cmap) == np.ndarray:
            converted_cmap = matplotlib.colors.ListedColormap(cmap)
            self.pcolor.set_cmap(converted_cmap)
        else:
            self.pcolor.set_cmap(cmap)

    # TO DO: Refactor to avoid code duplication with didv.colorplot.linecut
    # TO DO: horizontal_axis = 'distance'
    #        Plot dI/dV line profile against the distance along the interactive line.
    #        This is useful for the event that the user converts gate voltages to charge density or displacement field.
    # TO DO: horizontal_axis = callable function or functor
    #        Plot dI/dV line profile against a user-specified tranformation (gate1 : float, gate2 : float) -> float
    def linecut(self, startPoint = None, endPoint = None, ninterp = 200, axes = None, horizontal_axis = 'gate1'):

        r'''
        Interactive line profile on dI/dV(gate1, gate2) colorplot allowing the user to explore the data.
        This method places a mouse-draggable line on the colorplot, and it creates a new line plot on a separate matplotlib figure.
        Click on the line with the mouse to move the line.
        Click on the endpoints of the line to resize the line.

        Arguments:
            horizontal_axis : str
                Either 'gate1' or 'gate2'.
                This parameter specifies whether 'Gate 1 (V)' or 'Gate 2 (V)' should be plotted on the x-axis of the dI/dV 
                (or other channel) line profile.
        '''

        from matplotlib.collections import LineCollection

        x, y = np.meshgrid(self.gate1, self.gate2)
        x = x.ravel()
        y = y.ravel()
        data = self.data.ravel()

        gate1_range = np.max(self.gate1) - np.min(self.gate1)
        gate2_range = np.max(self.gate2) - np.min(self.gate2)

        fig = self.fig
        ax = self.ax

        first_pt = [None, None]
        last_pt = [None, None]
        line = [None, None, None]
        linecut_points = []

        nState = [0, None, None]
        pickrad = 2.0

        event_handlers = [None, None, None]

        if horizontal_axis == 'gate1':
            selector_index = 0
            horizontal_label = 'Gate 1 (V)'
        elif horizontal_axis == 'gate2':
            selector_index = 1
            horizontal_label = 'Gate 2 (V)'
        else:
            raise Exception('Unknown gate axis.')
        selector = lambda itm: itm[selector_index]

        if axes is None:
            linecut_fig = plt.figure()
            linecut_ax = linecut_fig.add_subplot(111)
        else:
            linecut_ax = axes
            linecut_fig = linecut_ax.figure

        def nearest_pt(gate1_val, gate2_val):
            dist_matrix = ((x - gate1_val)/gate1_range)**2 + ((y - gate2_val)/gate2_range)**2
            return np.argmin(dist_matrix)

        def build_line():
            line[0][0].set_xdata([first_pt[0], last_pt[0]])
            line[0][0].set_ydata([first_pt[1], last_pt[1]])
            fig.canvas.draw()
            x_list = np.linspace(first_pt[0], last_pt[0], ninterp)
            y_list = np.linspace(first_pt[1], last_pt[1], ninterp)
            linecut_points[:] = []
            for idx in range(ninterp):
                closest_pt_idx = nearest_pt(x_list[idx], y_list[idx])
                closest_pt = (x[closest_pt_idx], y[closest_pt_idx], data[closest_pt_idx])
                if closest_pt not in linecut_points:
                    linecut_points.append(closest_pt)
            linecut_points[:] = sorted(linecut_points, key = selector)
            points = np.array(linecut_points)
            horizontal_points = points[:, selector_index]
            vertical_points = points[:, 2]
            if line[1] is None:
                line[1], = linecut_ax.plot(horizontal_points, vertical_points)
                linecut_ax.set_xlabel(horizontal_label)
                line[2], = ax.plot(points[:, 0], points[:, 1], color = 'k', marker = 'x', markersize = 2, linestyle = 'None')
            else:
                line[1].set_xdata(horizontal_points)
                line[1].set_ydata(vertical_points)
                line[2].set_xdata(points[:, 0])
                line[2].set_ydata(points[:, 1])
            try:
                horizontal_min = np.nanmin(horizontal_points)
                horizontal_max = np.nanmax(horizontal_points)
                vertical_min = np.nanmin(vertical_points)
                vertical_max = np.nanmax(vertical_points)
                if horizontal_min != horizontal_max:
                    linecut_ax.set_xlim(horizontal_min, horizontal_max)
                if vertical_min != vertical_max:
                    linecut_ax.set_ylim(vertical_min, vertical_max)
            except ValueError:
                pass
            linecut_fig.canvas.draw()

        def on_click(event):
            if (event.xdata is not None) and (event.ydata is not None):
                if first_pt[0] is None :
                    if startPoint is None:
                        nearest_idx = nearest_pt(event.xdata, event.ydata)
                    else:
                        nearest_idx = nearest_pt(startPoint[0], startPoint[1])
                    first_pt[0] = x[nearest_idx]
                    first_pt[1] = y[nearest_idx]
                    line[0] = ax.plot([first_pt[0], first_pt[0]], [first_pt[1], first_pt[1]], pickradius = pickrad)
                    event_handlers[1] = fig.canvas.mpl_connect('motion_notify_event', on_motion)
                    event_handlers[2] = fig.canvas.mpl_connect('button_release_event', on_release)
                else:
                    try:
                        # Maybe scale with respect to xlim and ylim instead of gate1_range and gate2_range
                        first_dist = ((event.xdata - first_pt[0])/gate1_range)**2 + ((event.ydata - first_pt[1])/gate2_range)**2
                        last_dist = ((event.xdata - last_pt[0])/gate1_range)**2 + ((event.ydata - last_pt[1])/gate2_range)**2
                        distSens = ( 1.0 / 20.0 ) ** 2
                        if first_dist < last_dist:
                            if first_dist < distSens:
                                nState[1] = 0
                                return
                        elif first_dist > last_dist:
                            if last_dist < distSens:
                                nState[1] = 1
                                return
                        else:
                            pass
                        contains, _ = line[0][0].contains(event)
                        if not contains:
                            return
                        else:
                            nState[1] = 2
                            nState[2] = [first_pt[0], first_pt[1], last_pt[0], last_pt[1], event.xdata, event.ydata]
                    except TypeError:
                        pass

        def on_motion(event):
            if event.inaxes != ax:
                return
            if nState[0] == 0:
                if endPoint is None:
                    nearest_idx = nearest_pt(event.xdata, event.ydata)
                else:
                    nearest_idx = nearest_pt(endPoint[0], endPoint[1])
                last_pt[0] = x[nearest_idx]
                last_pt[1] = y[nearest_idx]
                build_line()
            else:
                nearest_idx = nearest_pt(event.xdata, event.ydata)
                if nState[1] == 0:
                    first_pt[0] = x[nearest_idx]
                    first_pt[1] = y[nearest_idx]
                    build_line()
                elif nState[1] == 1:
                    last_pt[0] = x[nearest_idx]
                    last_pt[1] = y[nearest_idx]
                    build_line()
                elif nState[1] == 2:
                    vec_x = event.xdata - nState[2][4]
                    vec_y = event.ydata - nState[2][5]
                    first_pt[0] = nState[2][0] + vec_x
                    first_pt[1] = nState[2][1] + vec_y
                    last_pt[0] = nState[2][2] + vec_x
                    last_pt[1] = nState[2][3] + vec_y
                    build_line()
                else:
                    pass

        def on_release(event):
            nState[0] += 1
            nState[1] = None
            nState[2] = None

        if (startPoint is not None) and (endPoint is not None):
            nearest_idx = nearest_pt(startPoint[0], startPoint[1])
            first_pt[0] = x[nearest_idx]
            first_pt[1] = y[nearest_idx]
            nearest_idx = nearest_pt(endPoint[0], endPoint[1])
            last_pt[0] = x[nearest_idx]
            last_pt[1] = y[nearest_idx]
            line[0] = ax.plot([first_pt[0], last_pt[0]], [first_pt[1], last_pt[1]], pickradius = pickrad)
            build_line()
            event_handlers[1] = fig.canvas.mpl_connect('motion_notify_event', on_motion)
            event_handlers[2] = fig.canvas.mpl_connect('button_release_event', on_release)
            nState[0] += 1

        event_handlers[0] = fig.canvas.mpl_connect('button_press_event', on_click)

def _mesh_for_pcolor(X, Y):

    r'''
    Create a meshgrid with one more row and column for pcolormesh using meshgrid (X, Y).

    Arguments:
        X : numpy float array
        Y : numpy float array
            (X, Y) are produced by numpy.meshgrid

    Returns:
        If the original (X, Y) represented the parameters for which the measurements were taken, 
        the returned (X, Y) represents the coordinates of the corners of the quadrilaterals
        required by matplotlib.pyplot.pcolormesh.
    '''

    X = np.insert(X, 0, X[:, 0] - (X[:, 1] - X[:, 0]), axis = 1)
    X = np.append(X, (X[:, -1] + (X[:, -1] - X[:,-2]))[:, np.newaxis], axis = 1)
    X = np.append(X, X[-1,np.newaxis,:], axis = 0)
    X = (X[:, 1:] + X[:, :-1]) * 0.5
    Y = np.insert(Y, 0, Y[0, :] - (Y[1, :] - Y[0, :]), axis = 0)
    Y = np.append(Y, (Y[-1, :] + (Y[-1, :] - Y[-2, :]))[np.newaxis, :], axis = 0)
    Y = np.append(Y, Y[:,-1,np.newaxis], axis = 1)
    Y = (Y[1:, :] + Y[:-1, :]) * 0.5
    return (X, Y)