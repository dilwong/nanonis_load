r'''
Loads and plots Nanonis .sxm data.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from . import util

from typing import Union, Tuple, Optional

# TO DO: drift correction

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
        if header_line.startswith(':') and header_line.endswith(':'):
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
        header['multipass_biases'] = []
        for row in multipass_rows:
            header['multipass_biases'].append(float(row.split('\t')[6]))
        header['multipass_biases'] = np.array(header['multipass_biases'])
    except (KeyError, IndexError):
        pass
        
    extra_info[1] = idx
    return header

#Loads .sxm files from Nanonis
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

    def __init__(self, filename : str):

        self.filename = filename
        self.data = {}
        extra_info = [None, None]
        self.header = sxm_header(filename, extra_info = extra_info)
        file, idx = extra_info
        raw_data = file[idx+5:]
        size = self.header['x_pixels'] * self.header['y_pixels']
        raw_data = np.frombuffer(raw_data, dtype='>f')
        
        if self.header['direction'] == 'down':
            for idx, channel_name in enumerate(self.header['channels']):
                channel_data = raw_data[idx*size*2:(idx+1)*size*2]
                self.data[channel_name] = [np.nan_to_num(np.flipud(channel_data[0:size].reshape(self.header['y_pixels'], self.header['x_pixels'])))]
                self.data[channel_name].append(np.nan_to_num(np.flip(channel_data[size:2*size].reshape(self.header['y_pixels'], self.header['x_pixels']), axis=(0, 1)))) # Backward channel
        else:
            for idx, channel_name in enumerate(self.header['channels']):
                channel_data = raw_data[idx*size*2:(idx+1)*size*2]
                self.data[channel_name] = [np.nan_to_num(channel_data[0:size].reshape(self.header['y_pixels'], self.header['x_pixels']))]
                self.data[channel_name].append(np.nan_to_num(np.fliplr(channel_data[size:2*size].reshape(self.header['y_pixels'], self.header['x_pixels'])))) # Backward channel

    def get_filename(self) -> str:
        '''
        Returns the filename of the sxm.
        '''
        return self.filename

    def get_scan_pixels(self) -> tuple[int, int]:
        '''
        Returns the number of x pixels and number of y pixels
        '''
        return (self.header['x_pixels'], self.header['y_pixels'])

    def get_gate_voltage(self) -> float:
        '''
        Returns the gate voltage in V.
        '''
        try:
            return float(self.header[':Ext. VI 1>Gate voltage (V):'][0])
        except:
            try:
                split_comment = self.header[':COMMENT:'][0].split()
                return float(split_comment[split_comment.index('V_g') + 2])
            except ValueError:
                return 0.0

    def get_sample_bias(self) -> float:
        '''
        Returns the sample bias in V.
        '''
        return float(self.header[':BIAS:'][0])

    def get_setpoint_current(self) -> float:
        '''
        Returns the setpoint current in pA. 
        '''
        return float(self.header[':Z-CONTROLLER:'][1].split('\t')[3].split()[0])*1e12

    def get_proportional(self) -> float:
        '''
        Returns the proportional gain in pm.
        '''
        return float(self.header[':Z-CONTROLLER:'][1].split('\t')[4].split()[0])*1e12

    def get_proportional_gain(self) -> float:
        return self.get_proportional()

    def get_time_per_line(self) -> tuple[float, float]:
        '''
        Returns
        -------
        forward_time, backward_time : float
            Time per line for forward and backward passes
        '''
        scan_time = self.header[':SCAN_TIME:'][0].split()
        return (float(scan_time[0]), float(scan_time[1]))

    def get_onenote_info_string(self) -> str:
        '''
        Returns
        -------
        info_string : str
            The info string containing info of the image. Paste it into your notes!
        '''
        return f"{self.get_filename()}\n\nPixels = {self.get_scan_pixels()}\nVg = {self.get_gate_voltage()} V\nVs = {round(self.get_sample_bias() * 1000, 2)} mV\nI = {self.get_setpoint_current()} pA\nProportional = {self.get_proportional()} pm\nTime per line = {self.get_time_per_line()} s"

    def get_data(self, channel : str, direction : int = 0) -> np.ndarray:

        return self.data[channel][direction%2]

    def crop_missing_data(self, channel : str, direction : int = 0):
        r"""
        Sets self.y_mask to exclude missing data in the y-direction.
        """
        channel_data = self.data[channel][direction%2]
        if self.y_mask is None:
            self.y_mask = ~(channel_data == 0.0).any(axis=1)
        else:
            self.y_mask = self.y_mask & (~(channel_data == 0.0).any(axis=1))

    @property
    def gate(self):
        return self.get_gate_voltage()

    @property
    def x_range(self):
        return self.header['x_range (nm)']

    @property
    def y_range(self):
        return self.header['y_range (nm)']

    @property
    def xy_range(self):
        return np.array([self.x_range, self.y_range])

    @property
    def x_pixels(self):
        return self.header['x_pixels']

    @property
    def y_pixels(self):
        return self.header['y_pixels']

    @property
    def xy_pixels(self):
        return np.array([self.x_pixels, self.y_pixels])

    @property
    def fft_x_bounds(self):
        return -self.x_pixels / self.x_range / 2, self.x_pixels / self.x_range / 2
    
    @property
    def fft_x_range(self):
        return self.x_pixels / self.x_range

    @property
    def fft_y_bounds(self):
        return -self.y_pixels / self.y_range / 2, self.y_pixels / self.y_range / 2

    @property
    def fft_y_range(self):
        return self.y_pixels / self.y_range

    @property
    def fft_range(self):
        return np.array([self.fft_x_range, self.fft_y_range])

    @property
    def fft_bottom_left_corner(self):
        return np.array([self.fft_x_bounds[0], self.fft_y_bounds[0]])

    def r_to_ij(self, r : np.ndarray, round=False) -> np.ndarray:
        '''
        Convert a real space vector to pixel coordinates. If round is true, the result will be rounded to the
        nearest integer.
        '''
        
        pixel_coords = r / self.xy_range * self.xy_pixels
        if round:
            pixel_coords = np.rint(pixel_coords)
        return pixel_coords

    def ij_to_r(self, ij : np.ndarray) -> np.ndarray:
        '''
        Convert pixel coordinates into a real space vector.
        '''
        return ij / self.xy_pixels * self.xy_range

    def k_to_ij(self, k : np.ndarray, round=False, two_pi=False) -> np.ndarray:
        '''
        Convert a momentum space vector to pixel coordinates of the FFT.
        
        Parameters
        ----------
        k : ndarray
            The momentum space vector to convert to FFT pixel coordinates.
        round : bool, optional
            Whether or not to round the final result. Default is False.
        two_pi : bool, optional
            Whether or not to divide the final result by 2*pi. Default is False.

        Returns
        -------
        pixel_coords : ndarray
            The pixel coordinates of k. 

        '''        
        pixel_coords = (k - self.fft_bottom_left_corner)/self.fft_range * self.xy_pixels

        if two_pi:
            pixel_coords /= 2*np.pi
        if round:
            pixel_coords = np.rint(pixel_coords)

        return pixel_coords

    def ij_to_k(self, ij : np.ndarray, two_pi=False) -> np.ndarray:
        '''
        Convert pixel coordinates to a momentum space vector
        
        Parameters
        ----------
        pixel_coords : ndarray
            Pixel coordinates of a point in the FFT
        two_pi : bool, optional
            Whether or not to multiply the final result by 2*pi. Default is False.

        Returns
        -------
        k : ndarray
            The momentum space vector corresponding to ij. 
        '''
        k = ij / self.xy_pixels * self.fft_range + self.fft_bottom_left_corner
        
        if two_pi:
            k *= 2*np.pi

        return k

    def subtract_plane(self, channel : str, direction : int=0) -> np.ndarray:
        '''
        Returns the specified channel and direction of the data with a plane subtracted.
        '''
        return subtract_plane(self.data[channel][direction])

    def subtract_linear_by_line(self, channel : str, direction : int=0) -> np.ndarray:
        '''
        Returns the specified channel and direction of the data with a linear fit by line subtracted.
        '''
        return subtract_linear_by_line(self.data[channel][direction])

    @staticmethod
    def process_data(data : np.ndarray, process : Union[str, Tuple]) -> np.ndarray:

        if isinstance(process, str):
            process_name = process.lower()
            if process_name not in process_dict:
                if process_name in preprocess_dict:
                    return data
                else:
                    assert False, f"Unknown process {process_name}"
            return process_dict[process_name](data)
        elif isinstance(process, tuple):
            process_name = process[0].lower()
            if process_name not in process_dict:
                if process_name in preprocess_dict:
                    return data
                else:
                    assert False, f"Unknown process {process_name}"
            return process_dict[process_name](data, *process[1:])
        else:
            raise TypeError('processing element must be of type str | tuple.')

    def preprocess_data(self, process):
        
        if isinstance(process, str):
            process_name = process.lower()
            if process_name in preprocess_dict:
                getattr(self, preprocess_dict[process_name])()
            else:
                if process_name not in process_dict:
                    assert False, f"Unknown process {process_name}"
        elif isinstance(process, tuple):
            process_name = process[0].lower()
            if process_name in preprocess_dict:
                getattr(self, preprocess_dict[process_name])(*process[1:])
            else:
                if process_name not in process_dict:
                    assert False, f"Unknown process {process_name}"
        else:
            raise TypeError('processing element must be of type str | tuple.')

    def crop_pixels_left(self, nPixels : int):
        r'''
        Sets self.x_mask to crop nPixels from the data in the x-direction.
        '''
        x_pixels = self.header['x_pixels']
        assert nPixels < x_pixels, "Trying to crop more pixels than is possible."
        mask = np.full(x_pixels, True)
        mask[0:nPixels] = False
        if self.x_mask is None:
            self.x_mask = mask
        else:
            self.x_mask = self.x_mask & mask

    def crop_pixels_right(self, nPixels : int):
        r'''
        Sets self.x_mask to crop nPixels from the data in the x-direction.
        '''
        x_pixels = self.header['x_pixels']
        assert nPixels < x_pixels, "Trying to crop more pixels than is possible."
        mask = np.full(x_pixels, True)
        mask[-nPixels:] = False
        if self.x_mask is None:
            self.x_mask = mask
        else:
            self.x_mask = self.x_mask & mask

    def crop_pixels_top(self, nPixels : int):
        r'''
        Sets self.y_mask to crop nPixels from the data in the y-direction.

        Note that missing data marked in self.y_mask by self.crop_missing_data is included in the cropping
        by self.crop_pixels_top. If the number of missing lines in the STM image is larger than nPixels,
        self.crop_pixels_top will do nothing.

        Also, calling self.crop_pixels_top(nPixels) more than once (say n times) is the same as calling it once.
        It is not the same as calling self.crop_pixels_top(n * nPixels).
        '''
        y_pixels = self.header['y_pixels']
        assert nPixels < y_pixels, "Trying to crop more pixels than is possible."
        mask = np.full(y_pixels, True)
        mask[-nPixels:] = False
        if self.y_mask is None:
            self.y_mask = mask
        else:
            self.y_mask = self.y_mask & mask

    def crop_pixels_bottom(self, nPixels : int):
        r'''
        Sets self.y_mask to crop nPixels from the data in the y-direction.
        '''
        y_pixels = self.header['y_pixels']
        assert nPixels < y_pixels, "Trying to crop more pixels than is possible."
        mask = np.full(y_pixels, True)
        mask[0:nPixels] = False
        if self.y_mask is None:
            self.y_mask = mask
        else:
            self.y_mask = self.y_mask & mask

    def crop_pixels_all(self, nPixels : int):
        self.crop_pixels_left(nPixels)
        self.crop_pixels_right(nPixels)
        self.crop_pixels_top(nPixels)
        self.crop_pixels_bottom(nPixels)

    def crop_pixels_window(self, bottom_left_x : int, bottom_left_y, width : int, height : int):
        r'''
        Crops a window of given width and height with the bottom left corner.
        '''
        x_pixels = self.header['x_pixels']
        y_pixels = self.header['y_pixels']

        self.crop_pixels_left(bottom_left_x)
        self.crop_pixels_bottom(bottom_left_y)
        self.crop_pixels_right(x_pixels - bottom_left_x - width)
        self.crop_pixels_top(y_pixels - bottom_left_y - height)

    def rotate_image(self, angle : float):
        raise NotImplementedError('sxm.rotate_image not yet implemented')

    def set_resolution(self, new_x_resolution : int, new_y_resolution, 
                        channel : Optional[str] = None, direction : Optional[int] = None,
                        method : Optional[str] = 'linear'):
        r'''
        Linear interpolate self.data for a new resolution.

        Parameters
        ----------
        new_x_resolution : int
            The new x-resolution to interpolate to.
        new_y_resolution : int
            The new y-resolution to interpolate to.
        channel : str or None
            The channel of the data to interpolate. If channel == None, then interpolate all channels.
        direction : 0 or 1 or None
            0 is the forward direction. 1 is the backwards direction. None is both directions.

        '''

        from scipy.interpolate import griddata

        try:
            if self.x_mask is not None or self.y_mask is not None:
                print('Warning: x_mask or y_mask is already defined and may not match new resolution.')
        except AttributeError:
            pass
        
        old_x_resolution = self.header["x_pixels"]
        old_y_resolution = self.header['y_pixels']
        x_range = self.header["x_range (nm)"]
        y_range = self.header["y_range (nm)"]

        self.header["x_pixels"] = new_x_resolution
        self.header['y_pixels'] = new_y_resolution
        # Need to edit self.header[':SCAN_PIXELS:']?

        x_new = np.linspace(0, x_range, new_x_resolution)
        y_new = np.linspace(0, y_range, new_y_resolution)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        x_old = np.linspace(0, x_range, old_x_resolution)
        y_old = np.linspace(0, y_range, old_y_resolution)
        X_old, Y_old = np.meshgrid(x_old, y_old)

        old_points = np.array(list(zip(X_old.ravel(), Y_old.ravel())))
        new_points = np.array(list(zip(X_new.ravel(), Y_new.ravel())))
        if channel is None:
            channels = self.data.keys()
        else:
            channels = [channel]
        if direction is None:
            directions = [0, 1]
        else:
            directions = [direction]
        for channel in channels:
            # for idx, _ in enumerate(self.data[channel]):
            for idx in directions:
                interp_image = griddata(old_points, self.data[channel][idx].ravel(), new_points, method=method)
                self.data[channel][idx] = interp_image.reshape((X_new.shape))

def scale(data: np.ndarray, multiply_factor: float) -> np.ndarray:
    '''
    Scale the data.
    '''
    return data * multiply_factor


def subtract_plane(data: np.ndarray) -> np.ndarray:
    '''
    Returns the input but with a plane subtracted from the entire array.
    The input MUST be a 2D array.

    Parameters
    ----------
    data : np.ndarray
        2D numpy array containing data.

    Returns
    -------
    output : ndarray
        The data with a fitted 2D plane subtracted from it.
    '''

    if len(data.shape) != 2:
        raise ValueError("Error: input array is not 2-dimensional.")

    x_dim = data.shape[1]
    y_dim = data.shape[0]

    X, Y = np.meshgrid(np.arange(0, x_dim), np.arange(0, y_dim))
    flattened_X = X.flatten()
    flattened_Y = Y.flatten()
    flattened_data = data.flatten()

    A = np.c_[flattened_X, flattened_Y, np.ones(len(flattened_X))] # Puts flattened_X, flattened_Y, and a column of ones into the columns of a matrix A
    C, _, _, _ = scipy.linalg.lstsq(A, flattened_data) # Finds the least squares solution to Ax = flattened_data where x contains the coefficients of the plane equation

    Z = C[0]*X + C[1]*Y + C[2] # Feeds X and Y into the fitted plane equation

    return data - Z


# TO DO: Refactor to avoid code duplication.


def subtract_parabola(data: np.ndarray) -> np.ndarray:
    '''
    Returns the input but with a parabolic fit over the entire array subtracted.
    The input MUST be a 2D array.

    Parameters
    ----------
    data : np.ndarray
        2D numpy array containing data.

    Returns
    -------
    output : ndarray
        The data with a fitted 2D parabola subtracted from it.
    '''

    if len(data.shape) != 2:
        raise ValueError("Error: input array is not 2-dimensional.")

    x_dim = data.shape[1]
    y_dim = data.shape[0]

    X, Y = np.meshgrid(np.arange(0, x_dim), np.arange(0, y_dim))
    flattened_X = X.flatten()
    flattened_Y = Y.flatten()
    flattened_data = data.flatten()

    A = np.c_[flattened_X**2, flattened_Y**2, flattened_X*flattened_Y, flattened_X, flattened_Y, np.ones(len(flattened_X))] # Puts flattened arrays into the columns of a matrix.
    C, _, _, _ = scipy.linalg.lstsq(A, flattened_data) # Finds the least squares solution to Ax = flattened_data where x contains the coefficients of the plane equation

    Z = C[0]*(X**2) + C[1]*(Y**2) + C[2]*(X*Y) + C[3]*X + C[4]*Y + C[5] # Feeds X and Y into the fitted parabola equation

    return data - Z


def subtract_linear_by_line(data : np.ndarray) -> np.ndarray:
    '''
    Returns the input but with a linear fit subtracted from each row.
    The input MUST be a 2D array.

    Parameters
    ----------
    data: ndarray
        Array to do subtraction on (duh).

    Returns
    -------
    output : ndarray
        The input array with a linear fit subtracted from each line.
    '''
    if len(data.shape) != 2:
        raise ValueError("Input for subtract_linear_by_line must be a 2D array.")
    
    indices = np.arange(0, data.shape[1], 1)
    line_subtracted = data.copy()
    for row in line_subtracted:
        fit = np.polyfit(indices, row, 1)
        row -= fit[0]*indices + fit[1]

    return line_subtracted


def subtract_vertical_linear_by_line(data : np.ndarray) -> np.ndarray:
    '''
    Returns the input but with a linear fit subtracted from each column.
    The input MUST be a 2D array.

    Parameters
    ----------
    data: ndarray
        Array to do subtraction on (duh).

    Returns
    -------
    output : ndarray
        The input array with a linear fit subtracted from each vertical line.
    '''
    data = data.T
    data = subtract_linear_by_line(data)
    return data.T


def subtract_quadratic_by_line(data : np.ndarray) -> np.ndarray:
    '''
    Returns the input but with a quadratic fit subtracted from each row.
    The input MUST be a 2D array.

    Parameters
    ----------
    data : ndarray
        Array to do subtraction on (duh).

    Returns
    -------
    output : ndarray
        The input array with a quadratic fit subtracted from each line.
    '''
    if len(data.shape) != 2:
        raise ValueError("Input for subtract_linear_by_line must be a 2D array.")
    
    indices = np.arange(0, data.shape[1], 1)
    line_subtracted = data.copy()
    for row in line_subtracted:
        fit = np.polyfit(indices, row, 2)
        row -= fit[0]*indices**2 + fit[1]*indices + fit[2]

    return line_subtracted


def subtract_vertical_quadratic_by_line(data : np.ndarray) -> np.ndarray:
    '''
    Returns the input but with a quadratic fit subtracted from each column.
    The input MUST be a 2D array.

    Parameters
    ----------
    data : ndarray
        Array to do subtraction on (duh).

    Returns
    -------
    output : ndarray
        The input array with a quadratic fit subtracted from each vertical line.
    '''
    data = data.T
    data = subtract_quadratic_by_line(data)
    return data.T


def fft2d(data : np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data)))


def fft_abs(data : np.ndarray) -> np.ndarray:
    return np.abs(fft2d(data))


def fft_arg(data : np.ndarray) -> np.ndarray:
    return np.angle(fft2d(data))


def fft_real(data : np.ndarray) -> np.ndarray:
    return np.real(fft2d(data))


def fft_imag(data : np.ndarray) -> np.ndarray:
    return np.imag(fft2d(data))


def moisan_decomposition(data : np.ndarray) -> np.ndarray:
    '''
    Removes the discontinuity at the edges of an image with periodic boundary conditions,
    using Moisan's algorithm (https://doi.org/10.1007/s10851-010-0227-1).

    A "cross" open appears in the Fourier transform of an image. This fixes this issue.

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    output : ndarray
    '''
    X = np.arange(data.shape[1], dtype = int)
    Y = np.arange(data.shape[0], dtype = int)
    v = np.zeros_like(data)
    xedge = data[-1, X] - data[0, X]
    v[0, X] = xedge
    v[-1, X] = -xedge
    yedge = data[Y, -1] - data[Y, 0]
    v[Y, 0] += yedge
    v[Y, -1] += -yedge
    s_denom = (
        2 * np.tile(np.cos(2*np.pi*X/data.shape[1]), (data.shape[0], 1))
        + 2 * np.tile(np.cos(2*np.pi*Y.reshape(-1,1)/data.shape[0]), (1, data.shape[1]))
        - 4)
    s_denom[0, 0] = 1
    s_q = np.fft.fft2(v)/s_denom
    s_q[0, 0] = 0
    s = np.fft.ifft2(s_q).real
    return data - s


def gaussian_blur(data : np.ndarray, sigma : float) -> np.ndarray:
    r'''
    Apply a Gaussian filter (with standard deviation sigma) to the data.

    Parameters
    ----------
    data : np.ndarray
    sigma : float

    Returns
    -------
    output : np.ndarray
    '''
    from scipy.ndimage import fourier_gaussian
    return np.fft.ifft2(fourier_gaussian(np.fft.fft2(data), sigma = sigma)).real

def despike(data : np.ndarray, sigma : float = 2.0) -> np.ndarray:
    r'''
    Removes spikes from image (iterating through vertical lines), using sigma as the threshold.
    Parameters
    ----------
    data : np.ndarray
    sigma : float
    Returns
    -------
    output : np.ndarray
    '''
    data = data.copy()
    for idx in range(data.shape[1]):
        peaks = scipy.signal.find_peaks(data[:, idx] - np.mean(data[:, idx]), threshold = sigma * np.std(data[:, idx]))[0]
        for p in peaks:
            try:
                data[p, idx] = (data[p - 1, idx] + data[p - 2, idx]) / 2.0
            except IndexError:
                pass

        dips = scipy.signal.find_peaks(-(data[:, idx] - np.mean(data[:, idx])), threshold = sigma * np.std(data[:, idx]))[0]
        for d in dips:
            try:
                data[d, idx] = (data[d - 1, idx] + data[d - 2, idx]) / 2.0
            except IndexError:
                pass
    return data


process_dict = {
    'absolute value': np.abs,
    'scale' : scale,
    'subtract plane': subtract_plane,
    'subtract parabola': subtract_parabola,
    'subtract linear fit per line': subtract_linear_by_line, #
    'subtract quadratic fit per line': subtract_quadratic_by_line, ##
    'subtract linear fit by line': subtract_linear_by_line, #
    'subtract quadratic fit by line': subtract_quadratic_by_line, ##
    'subtract linear by line': subtract_linear_by_line, #
    'subtract quadratic by line': subtract_quadratic_by_line, ##
    'subtract vertical linear fit by line': subtract_vertical_linear_by_line,
    'subtract vertical quadratic fit by line': subtract_vertical_quadratic_by_line,
    'moisan': moisan_decomposition,
    'gaussian blur': gaussian_blur,
    'despike': despike
}


preprocess_dict = {
    'crop left pixels' : 'crop_pixels_left',
    'crop right pixels' : 'crop_pixels_right',
    'crop top pixels' : 'crop_pixels_top',
    'crop bottom pixels' : 'crop_pixels_bottom',
    'crop pixels all sides' : 'crop_pixels_all',
    'crop pixels window' : 'crop_pixels_window',
    'rotate' : 'rotate_image',
    'set resolution' : 'set_resolution'
}

# This class probably flips the backwards pass orientation
# because the data storage convention in sxm.sxm.data was changed,
# but this class was not updated to reflect this.
# TO DO: FIX THIS
#
# direction = 0 for forward, direction = 1 for backwards
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

    def __init__(self, sxm_data : sxm, channel : str, direction : int=0, 
                flatten : bool=False, subtract_plane : bool=True,
                cmap=util.get_w_cmap(), rasterized=True, imshow_interpolation='antialiased'):

        self.data = sxm_data

        image_data = np.copy(sxm_data.data[channel][direction])
        avg_dat = image_data[~np.isnan(image_data)].mean()
        image_data[np.isnan(image_data)] = avg_dat
        if (flatten == True) and (subtract_plane == False):
            image_data=scipy.signal.detrend(image_data)

        # Flip upside down if image was taken scanning down
        # if sxm_data.header['direction'] == 'down':
        #     image_data=np.flipud(image_data)

        # Flip left to right if backwards scan
        #
        # THIS PROBABLY SHOULD BE DELETED.
        # if direction:
        #     image_data=np.fliplr(image_data)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        x_range = sxm_data.x_range
        y_range = sxm_data.y_range
        x_pixels = sxm_data.x_pixels
        y_pixels = sxm_data.y_pixels
        y, x = np.mgrid[0:x_range:(x_pixels+1)*1j,0:y_range:(y_pixels+1)*1j]
        #x = x.T
        #y = y.T
        if subtract_plane == True:
            image_data = sxm_data.subtract_plane(channel, direction)
        
        self.im_plot = self.ax.imshow(image_data, origin='lower', extent=(0, sxm_data.x_range, 0, sxm_data.y_range), 
                                        cmap=cmap, rasterized=rasterized, interpolation=imshow_interpolation)
        self.ax.set_aspect('equal')
        self.fig.colorbar(self.im_plot, ax = self.ax)
        self.image_data = image_data

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

    def clim(self, c_min, c_max):
        self.im_plot.set_clim(c_min, c_max)

    def std_clim(self, n_sigma):
        c_min = np.mean(self.image_data) - n_sigma*np.std(self.image_data)
        c_max = np.mean(self.image_data) + n_sigma*np.std(self.image_data)
        self.im_plot.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.im_plot.set_cmap(cmap)

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

    # TO DO: Add window functions.
    def fft(self):
        self.fft_fig = plt.figure()
        self.fft_ax = self.fft_fig.add_subplot(111)
        fft_array = np.absolute(np.fft.fft2(self.image_data))
        max_fft = np.max(fft_array[1:-1,1:-1])
        fft_array = np.fft.fftshift(fft_array)
        fft_x = -np.pi/(self.data.header['x_range (nm)']/self.data.header['x_pixels'])
        fft_y = np.pi/(self.data.header['y_range (nm)']/self.data.header['y_pixels'])
        self.fft_plot = self.fft_ax.imshow(fft_array, extent = [fft_x, -fft_x, -fft_y, fft_y], origin = 'lower')
        self.fft_fig.colorbar(self.fft_plot, ax = self.fft_ax)
        self.fft_clim(0,max_fft)

    def fft_clim(self, c_min, c_max):
        self.fft_plot.set_clim(c_min, c_max)

    def fft_colormap(self, cmap):
        self.fft_plot.set_cmap(cmap)