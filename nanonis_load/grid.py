r"""
Loads and plots Nanonis Grid Spectroscopy (.3ds) data.
"""

import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.signal

from .util import copy_text_to_clipboard

try:
    from . import interactive_colorplot
except ImportError:
    import interactive_colorplot


class Nanonis3ds:
    r"""
    grid.nanonis_3ds loads Nanonis .3ds files.

    Args:
        filename : str

    Attributes:
        header : dict
            A dictionary containing all of the header information from the .3ds file.
        data : dict
            A dictionary indexed by the data channels.
            The items in the dictionary are numpy arrays containing the
            numeric data.
        energy : numpy.ndarray
            A numpy array containing the spectroscopy biases.
    """

    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rb") as f:
            file = f.read()

        header_text = ""
        idx = 0
        while True:
            try:
                header_text += chr(file[idx])  # Python 3
            except TypeError:
                header_text += file[idx]  # Python 2
            idx += 1
            if ":HEADER_END:" in header_text:
                break
        header_text = header_text.split("\r\n")[:-1]
        self.header = dict()
        for entry in header_text:
            entry_array = entry.split("=")
            self.header[entry_array[0]] = entry_array[1]
            if entry_array[0] == "Comment":
                self.header[entry_array[0]] = "=".join(entry_array[1:])

        temp = re.split(' |"', self.header["Grid dim"])
        self.header["x_pixels"] = int(temp[1])
        self.header["y_pixels"] = int(temp[3])
        temp = re.split(";|=", self.header["Grid settings"])
        self.header["x_center (nm)"] = float(temp[0]) * 1e9
        self.header["y_center (nm)"] = float(temp[1]) * 1e9
        self.header["x_size (nm)"] = float(temp[2]) * 1e9
        self.header["y_size (nm)"] = float(temp[3]) * 1e9
        self.header["angle"] = float(temp[4])
        self.header["n_parameters"] = int(self.header["# Parameters (4 byte)"])
        self.header["points"] = int(self.header["Points"])
        channels = re.split('"|;', self.header["Channels"])[1:-1]

        self.data = {}
        raw_data = file[idx + 2 :]
        bpp = self.header["points"] * len(channels) + self.header["n_parameters"]
        self.data_pts = self.header["x_pixels"] * self.header["y_pixels"] * bpp
        numerical_data = np.frombuffer(raw_data, dtype=">f")
        self.header["Start Bias (V)"] = numerical_data[0]
        self.header["End Bias (V)"] = numerical_data[1]

        self.parameters = dict()
        self.parameter_list = []
        for param_name in self.header["Fixed parameters"].strip('"').split(";"):
            self.parameters[param_name] = []
            self.parameter_list.append(param_name)
        for param_name in self.header["Experiment parameters"].strip('"').split(";"):
            self.parameters[param_name] = []
            self.parameter_list.append(param_name)

        predata = [
            [{} for y in range(self.header["y_pixels"])]
            for x in range(self.header["x_pixels"])
        ]
        for i in range(self.header["x_pixels"]):
            for j in range(self.header["y_pixels"]):
                for k in range(len(channels)):
                    start_index = (
                        (i * self.header["y_pixels"] + j) * bpp
                        + self.header["n_parameters"]
                        + k * self.header["points"]
                    )
                    end_index = start_index + self.header["points"]
                    if numerical_data[start_index:end_index].size != 0:
                        predata[i][j][channels[k]] = numerical_data[
                            start_index:end_index
                        ]
                        if k == 0:
                            for idx, param_name in enumerate(self.parameter_list):
                                param_idx = (
                                    (i * self.header["y_pixels"] + j) * bpp
                                    + k * self.header["points"]
                                    + idx
                                )
                                self.parameters[param_name].append(
                                    numerical_data[param_idx]
                                )
                    else:
                        predata[i][j][channels[k]] = np.zeros(end_index - start_index)
                        if k == 0:
                            for param in self.parameters:
                                self.parameters[param].append(0)
        self.biases = np.linspace(
            self.header["Start Bias (V)"],
            self.header["End Bias (V)"],
            self.header["points"],
        )
        for ty in channels:
            self.data[ty] = np.array(
                [
                    [predata[x][y][ty] for y in range(self.header["y_pixels"])]
                    for x in range(self.header["x_pixels"])
                ]
            )


class Grid:

    def __init__(
        self,
        filename,
        channel="Input 2 (V)",
        fft=False,
        transform=None,
        energy_smoothing=None,
        initial_mode="linecut",  # 'linecut' or 'point'
    ):
        """Initializes the Grid object, loading data."""
        self.nanonis_3ds = Nanonis3ds(filename)
        self.filename = filename
        self.header = self.nanonis_3ds.header
        self.data = self.nanonis_3ds.data
        self.biases = self.nanonis_3ds.biases
        self.channel = channel
        self.fft = fft

        # --- Data Transformations ---
        # (Code for transform/smoothing - unchanged from previous version)
        self.transform = transform
        if self.transform == "diff":
            print("Applying numerical differentiation...")
            bias_step = (
                np.abs(self.biases[1] - self.biases[0])
                if self.biases is not None and len(self.biases) > 1
                else 1.0
            )
            if self.channel in self.data:
                self.data[self.channel] = np.gradient(
                    self.data[self.channel], bias_step, axis=-1
                )
            else:
                raise ValueError(
                    f"Channel '{self.channel}' not found for differentiation."
                )
        self.energy_smoothing = energy_smoothing
        if (
            self.energy_smoothing is not None
            and isinstance(self.energy_smoothing, (list, tuple))
            and len(self.energy_smoothing) == 2
        ):
            print(f"Applying Gaussian smoothing (sigma={self.energy_smoothing[1]})...")
            if self.channel in self.data:
                self.data[self.channel] = scipy.ndimage.gaussian_filter1d(
                    self.data[self.channel],
                    sigma=self.energy_smoothing[1],
                    axis=-1,
                    mode="nearest",
                )
            else:
                raise ValueError(f"Channel '{self.channel}' not found for smoothing.")

        # --- Interaction state ---
        self.sweep_index = 0
        self.mode = initial_mode  # 'linecut' or 'point'

        # State for Point Mode
        self.selected_points = []
        self.preset_colors = [
            "red",
            "blue",
            "orange",
            "cyan",
            "magenta",
            "green",
            "pink",
            "olive",
        ]

        # State for Linecut Mode
        self.click_start_pos = None  # Stores (x, y) of mouse press for linecut drag

        # --- Matplotlib objects ---
        self.fig = None
        self.plot_ax = None  # Main grid map
        self.spectrum_ax = None  # For point spectra
        self.linecut_ax = None  # For linecut imshow
        self.fft_ax = None  # Optional FFT map
        # self.fft_linecut_ax = None # Optional FFT linecut (omitted for now)

        self.im = None  # Grid map imshow object
        self.colorbar = None  # Grid map colorbar
        self.fft_plot = None  # FFT map imshow object
        self.fft_colorbar = None  # FFT map colorbar

        # Objects specific to modes
        self.selected_markers = None  # Scatter plot for point mode
        self.linecut_line = None  # Line2D overlay for linecut mode
        self.linecut_plot = None  # Linecut imshow object
        self.linecut_colorbar = None  # Linecut colorbar

        # Event connection IDs
        self._cid_key = None
        self._cid_press = None
        self._cid_motion = None
        self._cid_release = None

        # Final checks
        if self.channel not in self.data:
            raise ValueError(f"Channel '{self.channel}' not found.")
        if self.biases is None or len(self.biases) == 0:
            raise ValueError("Bias data missing.")
        if self.mode not in ["linecut", "point"]:
            print(
                f"Warning: Invalid initial_mode '{self.mode}'. Defaulting to 'linecut'."
            )
            self.mode = "linecut"

    def get_lockin_calibration_factor(
        self, lockin_channel: str = "Input 2 (V)"
    ) -> float:
        return np.linalg.lstsq(
            self.data["Input 2 (V)"][0][0][:, np.newaxis],
            np.gradient(self.data["Current (A)"][0][0], self.biases),
            rcond=None,
        )[0]

    # --- Properties (unchanged) ---
    @property
    def gate_voltage(self):
        try:
            return float(self.header.get("Ext. VI 1>Gate voltage (V)", "NaN"))
        except (ValueError, TypeError):
            return float("nan")

    @property
    def second_gate(self):
        try:
            return float(self.header.get("Ext. VI 1>Second gate voltage (V)", "NaN"))
        except (ValueError, TypeError):
            return float("nan")

    @property
    def x_size(self):
        return float(self.header["x_size (nm)"])

    @property
    def x_center(self):
        """Returns the center x coordinate in nm."""
        return self.header["x_center (nm)"]

    @property
    def xlist(self):
        """Returns a list of the x-coordinates (in nm) where data is taken."""
        x_center = self.header["x_center (nm)"]
        return np.linspace(
            x_center - self.x_size / 2, x_center + self.x_size / 2, self.x_pixels
        )

    @property
    def x_coords(self):
        return self.xlist

    @property
    def x_pixels(self):
        return int(self.header["x_pixels"])

    @property
    def y_size(self):
        return float(self.header["y_size (nm)"])

    @property
    def y_center(self):
        """Returns the center y coordinate in nm."""
        return self.header["y_center (nm)"]

    @property
    def ylist(self):
        """Returns a list of the y-coordinates (in nm) where data is taken."""
        y_center = self.header["y_center (nm)"]
        return np.linspace(
            y_center - self.y_size / 2, y_center + self.y_size / 2, self.y_pixels
        )

    @property
    def y_coords(self):
        return self.ylist

    @property
    def y_pixels(self):
        return int(self.header["y_pixels"])

    @property
    def Z(self):
        if (
            hasattr(self.nanonis_3ds, "parameters")
            and "Z (m)" in self.nanonis_3ds.parameters
        ):
            expected = self.y_pixels * self.x_pixels
            z_data = np.array(self.nanonis_3ds.parameters["Z (m)"])
            if z_data.size == expected:
                return z_data.reshape((self.y_pixels, self.x_pixels))
            else:
                print(f"Warning: Z data size mismatch.")
                return z_data
        else:
            print("Warning: Could not retrieve Z data.")
            return np.full((self.y_pixels, self.x_pixels), np.nan)

    @property
    def bias_range(self):
        return (
            abs(self.biases.max() - self.biases.min())
            if self.biases is not None and len(self.biases) > 0
            else 0
        )

    # --- Equality, Hashing, Comparison (unchanged) ---
    def __eq__(self, other):  # (Unchanged)
        if not isinstance(other, Grid):
            return NotImplemented
        gv_self, gv_other = self.gate_voltage, other.gate_voltage
        if np.isnan(gv_self) and np.isnan(gv_other):
            return True
        return gv_self == gv_other

    def __hash__(self):
        return hash(self.gate_voltage)  # (Unchanged)

    def __lt__(self, other):  # (Unchanged)
        if not isinstance(other, Grid):
            return NotImplemented
        gv_self, gv_other = self.gate_voltage, other.gate_voltage
        if np.isnan(gv_self) or np.isnan(gv_other):
            return np.isnan(gv_self) and not np.isnan(gv_other)
        return self.gate_voltage < other.gate_voltage

    def plot(self, sweep_index=0, mode="linecut", linecut_cmap="RdYlBu_r"):
        """
        Creates the plot layout and initializes the display.

        Parameters:
        - sweep_index: int
            The index of the energy slice to display.
        - mode: str
            The mode to initialize the plot with ('linecut' or 'point').
        """
        if not (0 <= sweep_index < len(self.biases)):
            sweep_index = 0
        self.sweep_index = sweep_index

        # Validate mode input
        if mode not in ["linecut", "point"]:
            print(f"Invalid mode '{mode}'. Defaulting to 'linecut'.")
            mode = "linecut"
        self.mode = mode

        # Close any existing figure
        if self.fig is not None:
            plt.close(self.fig)

        if self.fft:
            self.fig, axs = plt.subplots(2, 2, figsize=(12, 6))
            self.plot_ax = axs[0, 0]
            self.fft_ax = axs[0, 1]
            if self.mode == "linecut":
                self.linecut_ax = axs[1, 0]
                self.spectrum_ax = None
            else:
                self.spectrum_ax = axs[1, 0]
                self.linecut_ax = None
            plt.subplots_adjust(wspace=0.3, hspace=0.3)

        else:  # --- Create Figure and Axes (1x2 layout) ---
            self.fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            self.plot_ax = axs[0]
            if self.mode == "linecut":
                self.linecut_ax = axs[1]
                self.spectrum_ax = None
            else:
                self.spectrum_ax = axs[1]
                self.linecut_ax = None
            plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # --- Plot Initial Grid Image (Left Side) ---
        current_slice = self.data[self.channel][:, :, self.sweep_index]

        vmin = np.mean(current_slice) - 3 * np.std(current_slice)
        vmax = np.mean(current_slice) + 3 * np.std(current_slice)

        self.im = self.plot_ax.imshow(
            current_slice,
            extent=(0, self.x_size, 0, self.y_size),
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
        )
        self.colorbar = self.fig.colorbar(
            self.im, ax=self.plot_ax, label=self.channel, shrink=0.8
        )
        self.plot_ax.set_xlabel("X (nm)")
        self.plot_ax.set_ylabel("Y (nm)")
        self._update_plot_title()  # Updates title with energy

        # --- Plot FFT image (Left Side) ---
        if self.fft:
            fft_array = np.absolute(np.fft.fft2(current_slice))
            max_fft = np.max(fft_array[1:-1, 1:-1])
            fft_array = np.fliplr(
                np.fft.fftshift(fft_array)
            )  # Is this the correct orientation?
            fft_x = -np.pi / self.header["x_size (nm)"]
            fft_y = np.pi / self.header["y_size (nm)"]
            self.fft_plot = self.fft_ax.imshow(
                fft_array, extent=[fft_x, -fft_x, -fft_y, fft_y], origin="lower"
            )
            self.fft_colorbar = self.fig.colorbar(self.fft_plot, ax=self.fft_ax)
            self.fft_clim(0, max_fft)
        else:
            self.fft_plot = None

        # --- Initialize Right-Side Plot Based on Mode ---
        if self.mode == "linecut":
            # Linecut Mode
            self.linecut_ax.set_xlabel("Distance (nm)")
            self.linecut_ax.set_ylabel("Bias (V)")
            self.linecut_ax.set_title("Linecut")
            self.linecut_plot = self.linecut_ax.imshow(
                np.zeros((len(self.biases), 1)),  # Placeholder data
                cmap=linecut_cmap,
                aspect="auto",
                origin="lower",
                extent=(
                    0,
                    1,
                    self.biases.min(),
                    self.biases.max(),
                ),  # Placeholder extent
            )
            self.linecut_colorbar = self.fig.colorbar(
                self.linecut_plot, ax=self.linecut_ax, label=self.channel, shrink=0.8
            )
            # Linecut Mode Line (initially hidden)
            self.linecut_line = matplotlib.lines.Line2D(
                [0, 0], [0, 0], color="magenta", linewidth=3, visible=True
            )
            self.plot_ax.add_line(self.linecut_line)
        else:
            # Point Mode
            self.spectrum_ax.set_xlabel("Bias (V)")
            self.spectrum_ax.set_ylabel(f"Signal ({self.channel})")
            self.spectrum_ax.set_title("Point Spectra")
            self.spectrum_ax.grid(True)
            # Point Mode Markers
            self.selected_markers = self.plot_ax.scatter(
                [],
                [],
                marker="x",
                s=80,
                linewidths=1.5,
                zorder=10,
                edgecolors="black",
                facecolors="none",
                visible=True,
            )

        # --- Connect Event Handlers ---
        self._cid_key = self.fig.canvas.mpl_connect(
            "key_press_event", self._on_key_press
        )
        self._cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_press
        )
        self._cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_motion
        )
        self._cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self._on_release
        )

        # --- Set Initial Mode Visuals ---
        self._set_mode_visuals(
            self.mode
        )  # Ensure correct plots are cleared/titled initially

        plt.show()

    def _calculate_fft(self, data_slice):  # (Unchanged)
        if not np.all(np.isfinite(data_slice)):
            data_slice = np.nan_to_num(data_slice)
        return np.abs(np.fft.fftshift(np.fft.fft2(data_slice)))

    def _update_plot_title(self):  # (Unchanged)
        if (
            self.plot_ax
            and self.biases is not None
            and len(self.biases) > self.sweep_index
        ):
            energy_meV = self.biases[self.sweep_index] * 1000
            self.plot_ax.set_title(
                f"Energy = {energy_meV:.2f} meV (Slice {self.sweep_index})"
            )  # Press m to toggle mode
        elif self.plot_ax:
            self.plot_ax.set_title(f"Mode: {self.mode.capitalize()} ('m' to toggle)")

    def _update_image_slice(self):
        """Updates the main grid map and FFT map based on sweep_index."""
        if (
            self.im
            and 0 <= self.sweep_index < len(self.biases)
            and self.sweep_index < self.data[self.channel].shape[2]
        ):
            current_slice = self.data[self.channel][:, :, self.sweep_index]
            # (Handle non-finite, set data and clim for self.im - unchanged)
            if not np.all(np.isfinite(current_slice)):
                current_slice = np.where(
                    np.isfinite(current_slice), current_slice, np.nan
                )
            self.im.set_data(current_slice)
            vmin = np.mean(current_slice) - 3 * np.std(current_slice)
            vmax = np.mean(current_slice) + 3 * np.std(current_slice)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
                self.im.set_clim(vmin, vmax)
            elif np.isfinite(vmin) and np.isfinite(vmax):
                self.im.set_clim(
                    vmin - 0.1 * abs(vmin) if vmin != 0 else -0.1,
                    vmax + 0.1 * abs(vmax) if vmax != 0 else 0.1,
                )

            # Update FFT plot if enabled
            if self.fft and self.fft_plot and self.fft_ax:
                fft_data = self._calculate_fft(current_slice)
                self.fft_plot.set_data(fft_data)
                # Optionally update FFT clim (e.g., percentile)
                if np.any(fft_data > 0):
                    vmax_fft = np.percentile(fft_data[fft_data > 0], 99.5)
                    if vmax_fft > 0:
                        self.fft_plot.set_clim(0, vmax_fft)
                    else:
                        self.fft_plot.set_clim(
                            0, np.max(fft_data) if np.max(fft_data) > 0 else 1
                        )
                else:
                    self.fft_plot.set_clim(0, 1)

            self._update_plot_title()  # Update title with new energy
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()

    def _update_spectrum_plot(self):
        """Plots individual spectra in point mode."""
        if self.spectrum_ax is None or self.biases is None:
            return

        # Clear previous lines and legend from spectrum_ax
        lines_to_remove = self.spectrum_ax.get_lines()
        for line in lines_to_remove:
            line.remove()
        legend = self.spectrum_ax.get_legend()
        if legend:
            legend.remove()

        marker_coords_list = []
        marker_colors_list = []
        valid_points_count = 0

        if not self.selected_points:
            # Clear markers if no points selected
            if self.selected_markers:
                self.selected_markers.set_offsets(np.empty((0, 2)))
                self.selected_markers.set_facecolors("none")
            self.spectrum_ax.set_title("Point Mode Spectra (no points)")
        else:
            num_preset_colors = len(self.preset_colors)
            for i, point in enumerate(self.selected_points):
                px, py = point
                if 0 <= px < self.x_pixels and 0 <= py < self.y_pixels:
                    point_spectrum = self.data[self.channel][py, px, :]
                    if np.all(np.isfinite(point_spectrum)):
                        color_index = i % num_preset_colors
                        color = self.preset_colors[color_index]
                        # Plot spectrum line
                        self.spectrum_ax.plot(
                            self.biases,
                            point_spectrum,
                            label=f"({px},{py})",
                            color=color,
                        )
                        # Collect marker info
                        x_coord = (px + 0.5) * self.x_size / self.x_pixels
                        y_coord = (py + 0.5) * self.y_size / self.y_pixels
                        marker_coords_list.append((x_coord, y_coord))
                        marker_colors_list.append(color)
                        valid_points_count += 1
                    else:
                        print(f"Warning: Skipping point {point} (non-finite).")
                else:
                    print(f"Warning: Skipping invalid index {point}.")

            # Update markers plot
            if self.selected_markers:
                if marker_coords_list:
                    self.selected_markers.set_offsets(np.array(marker_coords_list))
                    self.selected_markers.set_facecolors(marker_colors_list)
                    self.selected_markers.set_edgecolors("black")
                else:  # Clear if no valid points ended up being plotted
                    self.selected_markers.set_offsets(np.empty((0, 2)))
                    self.selected_markers.set_facecolors("none")

            # Finalize spectrum plot appearance
            if valid_points_count > 0:
                self.spectrum_ax.relim()
                self.spectrum_ax.autoscale_view()
                plural = "s" if valid_points_count > 1 else ""
                self.spectrum_ax.set_title(
                    f"{valid_points_count} Point Spectrum{plural}"
                )
                self.spectrum_ax.legend(fontsize="small", loc="best")
            else:
                self.spectrum_ax.set_title("Point Mode Spectra (no valid points)")

        # Ensure grid and labels are consistent
        self.spectrum_ax.grid(True)
        self.spectrum_ax.set_xlabel("Bias (V)")
        self.spectrum_ax.set_ylabel(f"Signal ({self.channel})")

        if self.fig and self.fig.canvas:
            self.fig.canvas.draw_idle()

    def _update_linecut_plot(self):
        """Calculates and displays the linecut data."""
        if self.linecut_plot is None or self.linecut_ax is None:
            return

        # Get line endpoints in data coordinates
        xydata = self.linecut_line.get_xydata()
        x_start, y_start = xydata[0]
        x_end, y_end = xydata[1]

        # Avoid calculation if line has zero length
        if np.allclose(xydata[0], xydata[1]):
            # Optionally clear the plot or show a message
            dummy_data = np.full((len(self.biases), 1), np.nan)
            self.linecut_plot.set_data(dummy_data)
            self.linecut_plot.set_extent((0, 0.1, self.biases.min(), self.biases.max()))
            self.linecut_ax.set_title("Linecut (zero length)")
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()
            return

        # Convert data coordinates to pixel indices
        px_start = x_start / self.x_size * self.x_pixels
        py_start = y_start / self.y_size * self.y_pixels
        px_end = x_end / self.x_size * self.x_pixels
        py_end = y_end / self.y_size * self.y_pixels

        # Number of points along the line (based on pixel distance)
        num_pts = max(2, int(np.hypot(px_end - px_start, py_end - py_start) + 1))

        # Generate pixel coordinates along the line
        px_coords = np.linspace(px_start, px_end, num_pts)
        py_coords = np.linspace(py_start, py_end, num_pts)

        # Clip coordinates to be within valid pixel range (important!)
        px_coords = np.clip(px_coords, 0, self.x_pixels - 1)
        py_coords = np.clip(py_coords, 0, self.y_pixels - 1)

        # Extract data along the line using integer indexing (nearest neighbor)
        # Note: scipy.ndimage.map_coordinates could do interpolation if needed
        data_cut = self.data[self.channel][
            py_coords.astype(int), px_coords.astype(int), :
        ]

        # Transpose to have distance along x-axis, bias along y-axis for imshow
        data_cut = data_cut.T  # Shape: (num_bias_points, num_pts)

        # Determine line length in data units (nm)
        length = np.hypot(x_end - x_start, y_end - y_start)

        # Update the imshow plot
        self.linecut_plot.set_data(data_cut)
        self.linecut_plot.set_extent((0, length, self.biases.min(), self.biases.max()))

        # Update color limits based on the extracted data
        if np.any(np.isfinite(data_cut)):
            vmin, vmax = np.nanmin(data_cut), np.nanmax(data_cut)
            if vmin < vmax:
                self.linecut_plot.set_clim(vmin, vmax)
            else:  # Handle case where all values are the same
                self.linecut_plot.set_clim(
                    vmin - 0.1 * abs(vmin) if vmin != 0 else -0.1,
                    vmax + 0.1 * abs(vmax) if vmax != 0 else 0.1,
                )
        else:
            # Handle case with no valid data (e.g., all NaNs)
            self.linecut_plot.set_clim(0, 1)

        self.linecut_ax.set_title("Linecut")  # Reset title
        self.linecut_ax.relim()
        self.linecut_ax.autoscale_view()  # Adjust view limits

        if self.fig and self.fig.canvas:
            self.fig.canvas.draw_idle()

    def _set_mode_visuals(self, new_mode):
        """Updates visibility and clears plots when mode changes."""
        self.mode = new_mode
        is_point_mode = self.mode == "point"

        # Update visibility of markers and linecut line
        if self.selected_markers:
            self.selected_markers.set_visible(is_point_mode)
        if self.linecut_line:
            self.linecut_line.set_visible(not is_point_mode)

        # Clear the plot associated with the *inactive* mode
        if is_point_mode:
            # Clear linecut plot
            if self.linecut_plot:
                dummy_data = np.full((len(self.biases), 1), np.nan)
                self.linecut_plot.set_data(dummy_data)
            if self.linecut_ax:
                self.linecut_ax.set_title("Linecut Mode (Inactive)")
        else:
            # Clear spectrum plot
            if self.spectrum_ax:
                lines = self.spectrum_ax.get_lines()
                for line in lines:
                    line.remove()
                legend = self.spectrum_ax.get_legend()
                if legend:
                    legend.remove()
                self.spectrum_ax.relim()
                self.spectrum_ax.autoscale_view()
                self.spectrum_ax.set_title("Point Mode Spectra (Inactive)")
            # Clear point selection state as well
            self.selected_points.clear()
            if self.selected_markers:
                self.selected_markers.set_offsets(np.empty((0, 2)))
                self.selected_markers.set_facecolors("none")

        self._update_plot_title()  # Update main title to reflect mode
        print(f"Switched to {self.mode.capitalize()} mode.")
        if self.fig and self.fig.canvas:
            self.fig.canvas.draw_idle()

    # --- Event Handlers ---
    def _on_key_press(self, event):
        """Handles key presses for slice changes and mode switching."""
        if event.canvas.figure != self.fig:
            return

        # Mode Switch
        if event.key == "m":
            new_mode = "point" if self.mode == "linecut" else "linecut"
            self._set_mode_visuals(new_mode)
            return  # Mode switched, nothing else to do for this key press

        # Slice Change
        if self.biases is None or len(self.biases) == 0:
            return
        if event.key == "down":
            self.sweep_index -= 1
        elif event.key == "up":
            self.sweep_index += 1
        else:
            return  # Ignore other keys

        self.sweep_index %= len(self.biases)
        self._update_image_slice()  # Update map and FFT

    def _on_press(self, event):
        """Handles mouse button press for current mode."""
        if event.canvas.figure != self.fig or event.inaxes != self.plot_ax:
            return
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return  # Click outside axes

        if self.mode == "point":
            # --- Point Mode Press Logic ---
            if (
                self.x_size <= 0
                or self.y_size <= 0
                or self.x_pixels <= 0
                or self.y_pixels <= 0
            ):
                return
            # Calculate pixel index
            px = int(round(xdata / self.x_size * self.x_pixels - 0.5))
            py = int(round(ydata / self.y_size * self.y_pixels - 0.5))
            px = max(0, min(px, self.x_pixels - 1))
            py = max(0, min(py, self.y_pixels - 1))
            point = (px, py)

            is_shift = event.key == "shift"
            is_right_click = event.button == 3

            if is_right_click:
                if self.selected_points:
                    print("Point selection cleared.")
                    self.selected_points.clear()
            elif is_shift:
                if point not in self.selected_points:
                    self.selected_points.append(point)
                    print(f"Added point: {point}. Total: {len(self.selected_points)}")
                else:
                    self.selected_points.remove(point)
                    print(f"Removed point: {point}. Total: {len(self.selected_points)}")
            else:  # Left-click without shift
                if len(self.selected_points) != 1 or self.selected_points[0] != point:
                    self.selected_points = [point]
                    print(f"Selected point: {point}")

            self._update_spectrum_plot()  # Update spectrum and markers

        elif self.mode == "linecut":
            # --- Linecut Mode Press Logic ---
            if event.button == 1:  # Left-click starts line
                self.click_start_pos = (xdata, ydata)
                # Set initial line (zero length)
                self.linecut_line.set_data([xdata, xdata], [ydata, ydata])
                self._update_linecut_plot()  # Update linecut plot (will show zero length message)
                if self.fig and self.fig.canvas:
                    self.fig.canvas.draw_idle()

    def _on_motion(self, event):
        """Handles mouse motion for current mode (only linecut matters)."""
        if event.canvas.figure != self.fig or event.inaxes != self.plot_ax:
            return

        if (
            self.mode == "linecut"
            and event.button == 1
            and self.click_start_pos is not None
        ):
            # --- Linecut Mode Motion Logic ---
            x_start, y_start = self.click_start_pos
            x_end, y_end = event.xdata, event.ydata
            if x_end is None or y_end is None:
                return  # Moved outside axes

            # Update the visual line overlay
            self.linecut_line.set_data([x_start, x_end], [y_start, y_end])

            # Update the linecut imshow plot
            self._update_linecut_plot()
            # No need to redraw here, _update_linecut_plot does it

    def _on_release(self, event):
        """Handles mouse button release for current mode."""
        if event.canvas.figure != self.fig:
            return

        if self.mode == "linecut":
            # Reset click start position when mouse is released
            self.click_start_pos = None

    # --- Public Methods for Control (Unchanged) ---
    def clim(self, c_min, c_max):
        if self.im and self.fig and self.fig.canvas:
            self.im.set_clim(c_min, c_max)
            self.fig.canvas.draw_idle()

    def colormap(self, cmap):
        if self.im and self.fig and self.fig.canvas:
            self.im.set_cmap(cmap)
            self.fig.canvas.draw_idle()

    def fft_clim(self, c_min, c_max):
        if self.fft_plot and self.fig and self.fig.canvas:
            self.fft_plot.set_clim(c_min, c_max)
            self.fig.canvas.draw_idle()

    def fft_colormap(self, cmap):
        if self.fft_plot and self.fig and self.fig.canvas:
            self.fft_plot.set_cmap(cmap)
            self.fig.canvas.draw_idle()

    def plot_spectrum(self, i, j, channel="Input 2 (V)", fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.biases, self.data[channel][i, j, :])

        return fig, ax

    def extract_peak_energies(
        self,
        channel="Input 2 (V)",
        fit_radius=5,
        prominence=0.1,
        width=0,
        std=0.002,
        maxfev=1000,
        fallback="max",
        peak_choosing_method="prominences",
        skip_first_spectrum=False,
        verbose=False,
        full_output=True,
        break_on_exception=False,
        real_space_blur=0,
        energy_blur=0,
    ):
        """
        Extract peak energy as a function of position in the grid

        Parameters
        ----------
        channel : str
            The data channel to do peak energy extraction on.
        fit_radius : int
            The radius of the range of points to do Gaussian fitting on.
            Default is 5.
        prominence : float
            Peak prominence used in scipy.signal.find_peaks().
        width : int
            Peak width used in scipy.signal.find_peaks().
        std : float
            Standard deviation used in the initial guess for Gaussian fitting.
        maxfev : int
            Maximum number of function evaluations used in scipy.optimize.curve_fit().
        fallback : {'max', 'peak_ind'}
            Method of peak finding used if Gaussian fitting fails
            'max': Uses the bias of the maximum value in the spectrum
        peak_choosing_method : {'tracking', 'prominence', 'min', 'max'}
            The method for choosing between several peaks that
            scipy.signal.find_peaks() finds.
        skip_first_spectrum : bool
            Whether or not to skip the first spectrum. If skipped, will take average
            of two neighboring peak energies.
        verbose : bool
            Prints diagnostic information if true.
        full_output : bool
            Returns additional information such as peak indices.
        real_space_blur : float
            Radius in nm to apply Gaussian filter to the data before fitting.
        energy_blur : float
            Radius in V to apply Gaussian filter to the data before fitting.

        Returns
        -------
        peak_energies : ndarray
            Numpy array of peak energies extracted from each spectrum.
        """

        # Define gaussian for fitting
        def gaussian(x, x0, A, sigma, C):
            return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C

        peak_energies = np.zeros((self.y_pixels, self.x_pixels))

        last_peak_ind = 0  # Initalized for peak_tracking

        filter_sigma = (
            real_space_blur / self.y_size * self.y_pixels,
            real_space_blur / self.x_size * self.x_pixels,
            energy_blur / self.bias_range * len(self.biases),
        )
        filtered_data = scipy.ndimage.gaussian_filter(
            self.data[channel], sigma=filter_sigma
        )
        # Loop through all spectrum and extract peaks
        for i in range(self.data[channel].shape[0]):
            for j in range(self.data[channel].shape[1]):
                if i == 0 and j == 0 and skip_first_spectrum:
                    continue  # Skips fitting the first spectrum
                spectrum = filtered_data[i, j, :]
                try:
                    peaks, properties = scipy.signal.find_peaks(
                        spectrum, prominence=prominence, width=width
                    )

                    # Choose between peak choosing methods
                    if peak_choosing_method.lower() in ["track", "tracking"]:
                        peak_ind = peaks[np.argmin(np.abs(peaks - last_peak_ind))]
                    elif peak_choosing_method.lower() in ["prominence", "prominences"]:
                        peak_ind = peaks[np.argmax(properties["prominences"])]
                    elif peak_choosing_method.lower() == "min":
                        peak_ind = np.amin(peaks)
                    elif peak_choosing_method.lower() == "max":
                        peak_ind = np.amax(peaks)
                    else:
                        raise ValueError(
                            f"peak_choosing_method {peak_choosing_method} not recognized."
                        )
                except ValueError as e:
                    if verbose:
                        print(e)
                    peak_ind = np.argmax(spectrum)
                    if break_on_exception:
                        breakpoint()
                last_peak_ind = peak_ind
                fit_start = peak_ind - fit_radius if peak_ind - fit_radius > 0 else 0
                fit_end = (
                    peak_ind + fit_radius
                    if peak_ind + fit_radius < self.header["points"]
                    else self.header["points"]
                )

                p0 = (self.biases[peak_ind], spectrum[peak_ind], std, 0)

                try:
                    fit, cov = scipy.optimize.curve_fit(
                        gaussian,
                        self.biases[fit_start:fit_end],
                        spectrum[fit_start:fit_end],
                        p0,
                        maxfev=maxfev,
                    )
                    peak_energies[i, j] = fit[0]
                except RuntimeError as e:
                    print(
                        f"Error fitting point ({i}, {j}). Falling back to {fallback}."
                    )
                    if verbose:
                        print(e)
                    if fallback == "max":
                        peak_energies[i, j] = self.biases[np.argmax(spectrum)]
                    else:
                        peak_energies[i, j] = self.biases[peak_ind]

        if skip_first_spectrum:
            # Take average of neighboring points for first spectrum
            peak_energies[0, 0] = (peak_energies[1, 0] + peak_energies[0, 1]) / 2

        return peak_energies

    # TO DO : merge with 'extract_peak_energies'
    # we don't need 'animate' function

    # --- Cleanup ---
    def close(self):
        """Closes the matplotlib figure and disconnects callbacks."""
        if self.fig is not None:
            # Disconnect all callbacks
            if self._cid_key:
                self.fig.canvas.mpl_disconnect(self._cid_key)
            if self._cid_press:
                self.fig.canvas.mpl_disconnect(self._cid_press)
            if self._cid_motion:
                self.fig.canvas.mpl_disconnect(self._cid_motion)
            if self._cid_release:
                self.fig.canvas.mpl_disconnect(self._cid_release)
            self._cid_key = self._cid_press = self._cid_motion = self._cid_release = (
                None
            )

            plt.close(self.fig)
            self.fig = None
            print("Plot closed and callbacks disconnected.")

    def copy_onenote_info_string(self):
        gate_voltage = float(self.header["Ext. VI 1>Gate voltage (V)"])
        second_gate_voltage = float(self.header["Ext. VI 1>Second gate voltage (V)"])
        lockin_amplitude = float(self.header["Ext. VI 2>Amplitude (V)"])
        lockin_frequency = float(self.header["Ext. VI 2>Frequency (Hz)"])
        lockin_sensitivity = self.header["Ext. VI 2>Sensitivity"]
        lockin_time_constant = self.header["Ext. VI 2>Time constant"]
        lockin_phase = float(self.header["Ext. VI 2>Phase"])
        Cx_temp = float(self.header["Ext. VI 3>STM Cx Temp (K)"])
        Rx_temp = float(self.header["Ext. VI 3>STM Rx Temp (K)"])
        return_string = f"{self.filename}\n\nGate voltage = {gate_voltage} V\nSecond gate voltage = {second_gate_voltage} V\nLockin amplitude = {lockin_amplitude} (V)\nLockin frequency = {lockin_frequency} Hz\nLockin sensitivity = {lockin_sensitivity}\nLockin time constant = {lockin_time_constant}\nLockin phase = {lockin_phase}\nCx temperature = {Cx_temp}\nRx temperature = {Rx_temp}"
        copy_text_to_clipboard(return_string)


# TO DO: Copy data to clipboard
class older_Grid:
    r"""
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
            Change the colormap to cmap for the real-space image, where cmap is an acceptable
            matplotlib colormap.
        fft_clim(c_min : float, c_max : float) : None
            Set the color axis limits on the Fourier transform. c_min < c_max
        fft_colormap(cmap) : None
            Change the colormap to cmap for the Fourier transform, where cmap is an acceptable
            matplotlib colormap.
        show_spectra() : None
            If the user has clicked in the real-space image, show_spectra() will plot the
            spectra corresponding to the clicked pixel in a new matplotlib Figure.
    """

    def __init__(
        self,
        filename,
        channel="Input 2 (V)",
        fft=False,
        transform=None,
        energy_smoothing=None,
        plot=False,
        test=False,
        comparison_mode="gate",
    ):
        """ """
        self.nanonis_3ds = Nanonis3ds(filename)
        self.filename = filename
        self.header = self.nanonis_3ds.header
        self.data = self.nanonis_3ds.data
        self.transform = transform
        if self.transform == "diff":
            self.data = np.gradient(self.data, axis=-1)
        self.energy_smoothing = energy_smoothing
        if self.energy_smoothing is not None:
            self.data = scipy.ndimage.gaussian_filter(
                self.data, self.energy_smoothing[1], axes=-1
            )
        self.biases = self.nanonis_3ds.biases
        self.channel = channel
        self.press = None
        self.click = None
        self.fft = fft
        self._nanonis_3ds = self.nanonis_3ds

        self.auto_contrast = True
        self.comparison_mode = comparison_mode

    def get_lockin_calibration_factor(
        self, lockin_channel: str = "Input 2 (V)"
    ) -> float:
        return np.linalg.lstsq(
            self.data["Input 2 (V)"][0][0][:, np.newaxis],
            np.gradient(self.data["Current (A)"][0][0], self.biases),
            rcond=None,
        )[0]

    @property
    def gate_voltage(self):
        return float(self.header["Ext. VI 1>Gate voltage (V)"])

    @property
    def second_gate(self):
        return float(self.header["Ext. VI 1>Second gate voltage (V)"])

    @property
    def x_size(self):
        """Returns the x size in nm."""
        return float(self.header["x_size (nm)"])

    @property
    def x_center(self):
        """Returns the center x coordinate in nm."""
        return self.header["x_center (nm)"]

    @property
    def xlist(self):
        """Returns a list of the x-coordinates (in nm) where data is taken."""
        x_center = self.header["x_center (nm)"]
        return np.linspace(
            x_center - self.x_size / 2, x_center + self.x_size / 2, self.x_pixels
        )

    @property
    def x_coords(self):
        return self.xlist

    @property
    def x_pixels(self):
        """Returns the number of pixels in the x direction."""
        return int(self.header["x_pixels"])

    @property
    def y_size(self):
        """Returns the y size in nm"""
        return float(self.header["y_size (nm)"])

    @property
    def y_center(self):
        """Returns the center y coordinate in nm."""
        return self.header["y_center (nm)"]

    @property
    def y_pixels(self):
        """Returns the number of pixels in the y direction."""
        return int(self.header["y_pixels"])

    @property
    def ylist(self):
        """Returns a list of the y-coordinates (in nm) where data is taken."""
        y_center = self.header["y_center (nm)"]
        return np.linspace(
            y_center - self.y_size / 2, y_center + self.y_size / 2, self.y_pixels
        )

    @property
    def y_coords(self):
        return self.ylist

    @property
    def Z(self):
        """Returns the topography from the initial Z of each spectrum."""
        # TODO: MAKE SURE THE INDICES ARE CORRECT IF THE GRID IS NON SQUARE
        return np.array(self.nanonis_3ds.parameters["Z (m)"]).reshape(
            (self.y_pixels, self.x_pixels)
        )

    @property
    def bias_range(self):
        return abs(self.biases.max() - self.biases.min())

    def __eq__(self, other):
        if self.comparison_mode == "gate":
            return self.gate_voltage == other.gate_voltage
        elif self.comparison_mode == "second_gate":
            return self.second_gate == other.second_gate
        else:
            raise ValueError(f"{self.comparison_mode} is not a valid comparison mode.")

    def __lt__(self, other):
        if self.comparison_mode == "gate":
            return self.gate_voltage < other.gate_voltage
        elif self.comparison_mode == "second_gate":
            return self.second_gate < other.second_gate
        else:
            raise ValueError(f"{self.comparison_mode} is not a valid comparison mode.")

    def bias_slice(self, bias: float, channel="Input 2 (V)"):
        bias_index = np.argmin(np.abs(self.biases - bias))
        return self.data[channel][:, :, bias_index]

    def plot(self, sweep_index=0, channel="Input 2 (V)"):
        # Create axes for plotting
        if self.fft:
            self.fig = plt.figure(figsize=[2 * 6.4, 4.8])
            self.plot_ax = self.fig.add_subplot(221)
            self.fft_ax = self.fig.add_subplot(222)
            self.linecut_ax = self.fig.add_subplot(223)  # Axes for linecut through grid
            self.fft_linecut_ax = self.fig.add_subplot(
                224
            )  # Axes for linecut through fft
            self.linecut_ax.set_aspect("auto")
            self.fft_linecut_ax.set_aspect("auto")

        else:
            self.fig = plt.figure(figsize=[2 * 6.4, 4.8])
            self.plot_ax = self.fig.add_subplot(121)
            self.linecut_ax = self.fig.add_subplot(122)  # Axes for linecut through grid
            self.linecut_ax.set_aspect("auto")
            plt.subplots_adjust(wspace=0.3)

        # Plot grid
        self.im = self.plot_ax.imshow(
            np.flipud(self.data[channel][:, :, sweep_index]),
            extent=(0, self.header["x_size (nm)"], 0, self.header["y_size (nm)"]),
            cmap="Blues_r",
        )  # Check to make sure x_size and y_size aren't mixed up
        if self.fft:
            fft_array = np.absolute(np.fft.fft2(np.flipud(self.data[channel][:, :, 0])))
            max_fft = np.max(fft_array[1:-1, 1:-1])
            fft_array = np.fliplr(
                np.fft.fftshift(fft_array)
            )  # Is this the correct orientation?
            fft_x = -np.pi / self.header["x_size (nm)"]
            fft_y = np.pi / self.header["y_size (nm)"]
            self.fft_plot = self.fft_ax.imshow(
                fft_array, extent=[fft_x, -fft_x, -fft_y, fft_y], origin="lower"
            )
            self.fft_colorbar = self.fig.colorbar(self.fft_plot, ax=self.fft_ax)
            self.fft_clim(0, max_fft)
        else:
            self.fft_plot = None

        # Line representing the linecut will be drawn here
        self.linecut_line = matplotlib.lines.Line2D(
            [0, 0], [0, 0], color="r", linewidth=3
        )
        self.plot_ax.add_line(self.linecut_line)
        # Empty linecut plot as placeholder first
        self.linecut_plot = self.linecut_ax.imshow(
            np.zeros((1, 1)), cmap="RdYlBu_r", aspect="auto"
        )
        self.fig.colorbar(self.linecut_plot)
        self.linecut_ax.set_xlabel("Distance (nm)")
        self.linecut_ax.set_ylabel("Bias (V)")

        if self.fft:
            self.fft_linecut_line = matplotlib.lines.Line2D([0, 0], [0, 0], color="r")
            self.fft_ax.add_line(self.fft_linecut_line)
            self.fft_linecut_plot = self.fft_linecut_ax.imshow(
                np.zeros((1, 1)), cmap="RdYlBu_r", aspect="auto"
            )

        self.plot_ax.set_xlabel("X (nm)")
        self.plot_ax.set_ylabel("Y (nm)")
        self.colorbar = self.fig.colorbar(self.im, ax=self.plot_ax)
        self.free = 0
        title = "Energy = " + str(round(self.biases[sweep_index] * 1000, 4)) + "meV"
        self.plot_ax.set_title(title)

        def update_linecut():
            # Convert line endpoints to pixel units
            xydata = self.linecut_line.get_xydata()
            p0 = xydata[0] * self.header["x_pixels"] / self.header["x_size (nm)"]
            p1 = xydata[1] * self.header["y_pixels"] / self.header["y_size (nm)"]
            x0, y0 = p0.round().astype(int)
            x1, y1 = p1.round().astype(int)
            num_pts = int(np.hypot(x1 - x0, y1 - y0))  # Number of pixels
            # Create list of x and y pixel coordinates
            x, y = np.linspace(x0, x1, num_pts), np.linspace(y0, y1, num_pts)
            # Create linecut. Is the indexing correct?
            data_cut = self.data[channel][y.astype(int), x.astype(int), :].T

            min_bias = np.amin(
                [
                    float(self.header["Start Bias (V)"]),
                    float(self.header["End Bias (V)"]),
                ]
            )
            max_bias = np.amax(
                [
                    float(self.header["Start Bias (V)"]),
                    float(self.header["End Bias (V)"]),
                ]
            )

            if float(self.header["Start Bias (V)"]) < float(
                self.header["End Bias (V)"]
            ):
                data_cut = np.flipud(data_cut)

            self.linecut_plot.set_data(data_cut)
            length = np.hypot(xydata[1, 0] - xydata[0, 0], xydata[0, 1] - xydata[1, 1])
            self.linecut_plot.set_extent((0, length, min_bias, max_bias))
            try:
                self.linecut_plot.set_clim(0, data_cut.max())
            except:
                pass

        def key_press(event):
            if event.key[0:4] == "alt+":
                key = event.key[4:]
            else:
                key = event.key

            if key == "down":
                self.free -= 1
            elif key == "up":
                self.free += 1
            if self.free < 0:
                self.free = len(self.biases) - 1
            elif self.free >= len(self.biases):
                self.free = 0
            data = np.flipud(self.data[channel][:, :, self.free])
            self.im.set_data(data)
            if self.fft:
                fft_array = np.absolute(np.fft.fft2(data))
                fft_array = np.fliplr(
                    np.fft.fftshift(fft_array)
                )  # Is this the correct orientation?
                self.fft_plot.set_data(fft_array)
            self.im.set_clim(data.min(), data.max())
            title = "Energy = " + str(self.biases[self.free]) + " eV"
            self.plot_ax.set_title(title)
            self.fig.canvas.draw()

        def on_press(event):
            if event.inaxes == self.plot_ax and event.button == 1:
                self.click = (event.xdata, event.ydata)
                self.linecut_line.set_xdata([event.xdata, event.xdata])
                self.linecut_line.set_ydata([event.ydata, event.ydata])
            else:
                return

        def on_motion(event):
            if event.inaxes == self.plot_ax and event.button == 1:
                self.linecut_line.set_xdata([self.click[0], event.xdata])
                self.linecut_line.set_ydata([self.click[1], event.ydata])
                update_linecut()
                self.fig.canvas.draw()

        def on_release(event):
            return

        self.key_press = key_press
        self.fig.canvas.mpl_connect("key_press_event", key_press)
        self.fig.canvas.mpl_connect("button_press_event", on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", on_motion)
        self.fig.canvas.mpl_connect("button_release_event", on_release)

    def clim(self, c_min, c_max):
        self.im.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.im.set_cmap(cmap)

    def fft_clim(self, c_min, c_max):
        self.fft_plot.set_clim(c_min, c_max)

    def fft_colormap(self, cmap):
        self.fft_plot.set_cmap(cmap)

    def plot_spectrum(self, i, j, channel="Input 2 (V)", fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.biases, self.data[channel][i, j, :])

        return fig, ax

    # TO DO: Implement channel keyword
    def show_spectra(self, channel=None, ax=None):

        if self.click is None:
            return
        x_pixel = int(
            np.floor(
                self.click[0] * self.header["x_pixels"] / self.header["x_size (nm)"]
            )
        )
        y_pixel = int(
            np.floor(
                self.click[1] * self.header["y_pixels"] / self.header["y_size (nm)"]
            )
        )
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.biases, self.data[channel][y_pixel, x_pixel, :])

        # TO DO: test this
        x = (x_pixel + 0.5) * self.header["x_size (nm)"] / self.header["x_pixels"]
        y = (y_pixel + 0.5) * self.header["y_size (nm)"] / self.header["x_pixels"]
        theta = -np.radians(self.header["angle"])
        R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        xy_vec = (
            x - self.header["x_size (nm)"] * 0.5,
            y - self.header["y_size (nm)"] * 0.5,
        )
        transformed_vec = R.dot(xy_vec)
        transformed_x = transformed_vec[0] + self.header["x_center (nm)"]
        transformed_y = transformed_vec[1] + self.header["y_center (nm)"]
        print("x = " + str(transformed_x) + " nm")
        print("y = " + str(transformed_y) + " nm")

    # dpi does not work... why?
    # Needs a MovieWriter
    def animate(self, key, wait_time=200, filename=None, dpi=80, writer="pillow"):
        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(
            self.fig,
            self.increment_energy,
            frames=[key] * (len(self.biases) - 0),
            interval=wait_time,
        )  # Fix ordering
        if filename is not None:
            anim.save(filename, dpi=dpi, writer=writer)

    def extract_peak_energies(
        self,
        channel="Input 2 (V)",
        fit_radius=5,
        prominence=0.1,
        width=0,
        std=0.002,
        maxfev=1000,
        fallback="max",
        peak_choosing_method="prominences",
        skip_first_spectrum=False,
        verbose=False,
        full_output=True,
        break_on_exception=False,
        real_space_blur=0,
        energy_blur=0,
    ):
        """
        Extract peak energy as a function of position in the grid

        Parameters
        ----------
        channel : str
            The data channel to do peak energy extraction on.
        fit_radius : int
            The radius of the range of points to do Gaussian fitting on.
            Default is 5.
        prominence : float
            Peak prominence used in scipy.signal.find_peaks().
        width : int
            Peak width used in scipy.signal.find_peaks().
        std : float
            Standard deviation used in the initial guess for Gaussian fitting.
        maxfev : int
            Maximum number of function evaluations used in scipy.optimize.curve_fit().
        fallback : {'max', 'peak_ind'}
            Method of peak finding used if Gaussian fitting fails
            'max': Uses the bias of the maximum value in the spectrum
        peak_choosing_method : {'tracking', 'prominence', 'min', 'max'}
            The method for choosing between several peaks that
            scipy.signal.find_peaks() finds.
        skip_first_spectrum : bool
            Whether or not to skip the first spectrum. If skipped, will take average
            of two neighboring peak energies.
        verbose : bool
            Prints diagnostic information if true.
        full_output : bool
            Returns additional information such as peak indices.
        real_space_blur : float
            Radius in nm to apply Gaussian filter to the data before fitting.
        energy_blur : float
            Radius in V to apply Gaussian filter to the data before fitting.

        Returns
        -------
        peak_energies : ndarray
            Numpy array of peak energies extracted from each spectrum.
        """

        # Define gaussian for fitting
        def gaussian(x, x0, A, sigma, C):
            return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C

        peak_energies = np.zeros((self.y_pixels, self.x_pixels))

        last_peak_ind = 0  # Initalized for peak_tracking

        filter_sigma = (
            real_space_blur / self.y_size * self.y_pixels,
            real_space_blur / self.x_size * self.x_pixels,
            energy_blur / self.bias_range * len(self.biases),
        )
        filtered_data = scipy.ndimage.gaussian_filter(
            self.data[channel], sigma=filter_sigma
        )
        # Loop through all spectrum and extract peaks
        for i in range(self.data[channel].shape[0]):
            for j in range(self.data[channel].shape[1]):
                if i == 0 and j == 0 and skip_first_spectrum:
                    continue  # Skips fitting the first spectrum
                spectrum = filtered_data[i, j, :]
                try:
                    peaks, properties = scipy.signal.find_peaks(
                        spectrum, prominence=prominence, width=width
                    )

                    # Choose between peak choosing methods
                    if peak_choosing_method.lower() in ["track", "tracking"]:
                        peak_ind = peaks[np.argmin(np.abs(peaks - last_peak_ind))]
                    elif peak_choosing_method.lower() in ["prominence", "prominences"]:
                        peak_ind = peaks[np.argmax(properties["prominences"])]
                    elif peak_choosing_method.lower() == "min":
                        peak_ind = np.amin(peaks)
                    elif peak_choosing_method.lower() == "max":
                        peak_ind = np.amax(peaks)
                    else:
                        raise ValueError(
                            f"peak_choosing_method {peak_choosing_method} not recognized."
                        )
                except ValueError as e:
                    if verbose:
                        print(e)
                    peak_ind = np.argmax(spectrum)
                    if break_on_exception:
                        breakpoint()
                last_peak_ind = peak_ind
                fit_start = peak_ind - fit_radius if peak_ind - fit_radius > 0 else 0
                fit_end = (
                    peak_ind + fit_radius
                    if peak_ind + fit_radius < self.header["points"]
                    else self.header["points"]
                )

                p0 = (self.biases[peak_ind], spectrum[peak_ind], std, 0)

                try:
                    fit, cov = scipy.optimize.curve_fit(
                        gaussian,
                        self.biases[fit_start:fit_end],
                        spectrum[fit_start:fit_end],
                        p0,
                        maxfev=maxfev,
                    )
                    peak_energies[i, j] = fit[0]
                except RuntimeError as e:
                    print(
                        f"Error fitting point ({i}, {j}). Falling back to {fallback}."
                    )
                    if verbose:
                        print(e)
                    if fallback == "max":
                        peak_energies[i, j] = self.biases[np.argmax(spectrum)]
                    else:
                        peak_energies[i, j] = self.biases[peak_ind]

        if skip_first_spectrum:
            # Take average of neighboring points for first spectrum
            peak_energies[0, 0] = (peak_energies[1, 0] + peak_energies[0, 1]) / 2

        return peak_energies

    def copy_onenote_info_string(self):
        gate_voltage = float(self.header["Ext. VI 1>Gate voltage (V)"])
        second_gate_voltage = float(self.header["Ext. VI 1>Second gate voltage (V)"])
        lockin_amplitude = float(self.header["Ext. VI 2>Amplitude (V)"])
        lockin_frequency = float(self.header["Ext. VI 2>Frequency (Hz)"])
        lockin_sensitivity = self.header["Ext. VI 2>Sensitivity"]
        lockin_time_constant = self.header["Ext. VI 2>Time constant"]
        lockin_phase = float(self.header["Ext. VI 2>Phase"])
        Cx_temp = float(self.header["Ext. VI 3>STM Cx Temp (K)"])
        Rx_temp = float(self.header["Ext. VI 3>STM Rx Temp (K)"])
        return_string = f"{self.filename}\n\nGate voltage = {gate_voltage} V\nSecond gate voltage = {second_gate_voltage} V\nLockin amplitude = {lockin_amplitude} (V)\nLockin frequency = {lockin_frequency} Hz\nLockin sensitivity = {lockin_sensitivity}\nLockin time constant = {lockin_time_constant}\nLockin phase = {lockin_phase}\nCx temperature = {Cx_temp}\nRx temperature = {Rx_temp}"
        copy_text_to_clipboard(return_string)


# Loads and plots 3DS line cuts
class Linecut(interactive_colorplot.Colorplot):
    r"""
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
    """

    def __init__(
        self,
        filename,
        channel,
        normalize=False,
        calibrate_didv=True,
        fig=None,
        ax=None,
        use_millivolts=True,
        rasterized=False,
    ):

        interactive_colorplot.Colorplot.__init__(self)
        self.sxm_fig = None
        self.sxm_data = None

        # Load data with filename
        self.nanonis_3ds = Nanonis3ds(filename)
        self.n_positions = self.nanonis_3ds.header["x_pixels"]
        if self.nanonis_3ds.header["y_pixels"] != 1:
            print("WARNING: " + filename + " IS NOT A LINE CUT")
            print("         grid.linecut MAY NOT WORK AS EXPECTED")
        self.n_energies = self.nanonis_3ds.header["points"]
        self.bias = self.nanonis_3ds.biases
        self.x_values = np.array(self.nanonis_3ds.parameters["X (m)"]) * 1e9
        self.y_values = np.array(self.nanonis_3ds.parameters["Y (m)"]) * 1e9
        self.dist = np.sqrt(
            (self.x_values - self.x_values[0]) ** 2
            + (self.y_values - self.y_values[0]) ** 2
        )

        self.data = np.array(
            [
                self.nanonis_3ds.data[channel][site].flatten()
                for site in range(self.n_positions)
            ]
        )
        # Calibrate units to be in siemens
        if calibrate_didv:
            # Current of the first point for calibration purposes
            first_point_current = self.nanonis_3ds.data["Current (A)"][0]
            current_gradient = np.gradient(np.squeeze(first_point_current), self.bias)
            calibration_factor = np.linalg.lstsq(
                self.data[0][:, np.newaxis], current_gradient, rcond=None
            )[0]
            self.data *= calibration_factor * 1e9  # Convert to nS
        if normalize == "integral":
            self.data = (self.data.T / np.sum(self.data, axis=-1)).T
        elif normalize == "max":
            self.data = (self.data.T / np.amax(self.data, axis=-1)).T
        if fig is not None:
            self.fig = fig
        else:
            self.fig = plt.figure()
        if ax is not None:
            self.ax = ax
        else:
            self.ax = self.fig.add_subplot(111)

        self.ax.set_xlabel("Distance (nm)")
        self.ax.set_ylabel("Sample bias (V)")

        new_bias = (self.bias[1:] + self.bias[:-1]) * 0.5
        new_bias = np.insert(
            new_bias, 0, self.bias[0] - (self.bias[1] - self.bias[0]) * 0.5
        )
        new_bias = np.append(
            new_bias, self.bias[-1] + (self.bias[-1] - self.bias[-2]) * 0.5
        )
        new_dist = (self.dist[1:] + self.dist[:-1]) * 0.5
        new_dist = np.insert(
            new_dist, 0, self.dist[0] - (self.dist[1] - self.dist[0]) * 0.5
        )
        new_dist = np.append(
            new_dist, self.dist[-1] + (self.dist[-1] - self.dist[-2]) * 0.5
        )
        x, y = np.meshgrid(new_dist, new_bias)
        x = x.T
        y = y.T
        # self.data = self.data.T
        self.pcolor = self.ax.pcolormesh(
            x, y, self.data, cmap="RdYlBu_r", rasterized=rasterized
        )
        self.fig.colorbar(self.pcolor, ax=self.ax)

        self.show_image_set = False
        self.xlist = self.bias
        self.ylist = self.dist

    def point_transform(self, x, y):
        if self.sxm_data is None:
            return None
        theta = np.radians(self.sxm_data.header["angle"])
        R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        transformed = (
            x - self.sxm_data.header["x_center (nm)"],
            y - self.sxm_data.header["y_center (nm)"],
        )
        transformed = R.dot(transformed)
        transformed_x = transformed[0] + self.sxm_data.header["x_range (nm)"] * 0.5
        transformed_y = transformed[1] + self.sxm_data.header["y_range (nm)"] * 0.5
        return (transformed_x, transformed_y)

    def show_image(self, filename, flatten=True, subtract_plane=False):

        try:
            from . import sxm
        except ImportError:
            import sxm

        self.sxm_data = sxm.sxm(filename)
        self.sxm_fig = sxm.Plot(
            self.sxm_data, "Z (m)", flatten=flatten, subtract_plane=subtract_plane
        )

        transformed_pts = [
            self.point_transform(x, y) for x, y in zip(self.x_values, self.y_values)
        ]
        x_values, y_values = zip(*transformed_pts)
        self.transformed_x_values = x_values
        self.transformed_y_values = y_values
        self.sxm_fig.ax.plot(x_values, y_values, color="k")
        self.sxm_fig.ax.set_aspect("equal")
        for bar in self._draggables:
            if bar.direction[0] == "h":
                bar.sxm_circle = matplotlib.patches.Circle(
                    (x_values[bar.index], y_values[bar.index]),
                    radius=0.5,
                    color=bar.color,
                    zorder=10,
                )
                self.sxm_fig.ax.add_patch(bar.sxm_circle)
                bar.sxm = self
                try:
                    bar.sxm_dot
                except AttributeError:
                    bar.sxm_dot = True

                    def slide_circle(input_bar=bar):
                        try:
                            input_bar.sxm_circle.center = (
                                input_bar.sxm.transformed_x_values[input_bar.index],
                                input_bar.sxm.transformed_y_values[input_bar.index],
                            )
                            input_bar.sxm.sxm_fig.fig.canvas.draw()  # TO DO: Blit for speed
                        except KeyError:
                            pass

                    bar.functions.append(slide_circle)
        self.show_image_set = True

    def drag_bar(self, direction="horizontal", locator=False, axes=None, color=None):

        dbar = super(Linecut, self).drag_bar(
            direction=direction, locator=locator, axes=axes, color=color
        )

        if (self.sxm_fig is not None) and (dbar.direction[0] == "h"):
            dbar.sxm_circle = matplotlib.patches.Circle(
                (
                    self.transformed_x_values[dbar.index],
                    self.transformed_y_values[dbar.index],
                ),
                radius=0.5,
                color=dbar.color,
                zorder=10,
            )
            self.sxm_fig.ax.add_patch(dbar.sxm_circle)
            dbar.sxm = self

            def slide_circle(input_bar=dbar):
                try:
                    input_bar.sxm_circle.center = (
                        input_bar.sxm.transformed_x_values[input_bar.index],
                        input_bar.sxm.transformed_y_values[input_bar.index],
                    )
                    input_bar.sxm.sxm_fig.fig.canvas.draw()  # TO DO: Blit for speed
                except KeyError:
                    pass

            dbar.functions.append(slide_circle)

        return dbar

    def get_onenote_info_string(self) -> str:
        """
        Returns an info string to paste into your notes
        """

        filename = self.nanonis_3ds.filename
        header = self.nanonis_3ds.header

        try:
            gate_voltage = header["Ext. VI 1>Gate voltage (V)"]
        except:
            second_gate_voltage = "Not recorded"
        try:
            second_gate_voltage = header["Ext. VI 1>Second gate voltage (V)"]
        except:
            second_gate_voltage = "Not recorded"

        bias_range_float = (
            float(header["Start Bias (V)"]),
            float(header["End Bias (V)"]),
        )
        if np.abs(bias_range_float[0]) < 1 or np.abs(bias_range_float[1]) < 1:
            bias_range = f"{(round(bias_range_float[0]*1000, 2), round(bias_range_float[1]*1000, 2))} mV"
        else:
            bias_range = (
                f"{(round(bias_range_float[0], 2), round(bias_range_float[1], 2))} V"
            )

        try:
            lockin_amplitude = header["Ext. VI 2>Amplitude (V)"]
        except:
            lockin_amplitude = "Not recorded"
        try:
            lockin_frequency = f"{header['Ext. VI 2>Frequency (Hz)']} Hz"
        except:
            lockin_frequency = "Not recorded"
        try:
            lockin_sensitivity = header["Ext. VI 2>Sensitivity"]
        except:
            lockin_sensitivity = "Not recorded"
        try:
            lockin_time_constant = header["Ext. VI 2>Time constant"]
        except:
            lockin_time_constant = "Not recorded"

        return f"{filename}\n\nGate voltage = {gate_voltage} V\nSecond gate = {second_gate_voltage} V\nBias range = {bias_range}\nLockin amplitude = {lockin_amplitude}\nLockin frequency = {lockin_frequency}\nLockin sensitivity = {lockin_sensitivity}\nLockin time constant = {lockin_time_constant}"

    def copy_onenote_info_string(self):
        """
        Copies the string returned by get_onenote_info_string() onto the clipboard.
        """
        copy_text_to_clipboard(self.get_onenote_info_string())


def quick_plot(filename, **kwargs):

    loaded_data = Nanonis3ds(filename)
    try:
        return Grid(loaded_data, channel="Input 2 (V)", **kwargs)
    except KeyError:
        return Grid(loaded_data, channel="Input 2 [AVG] (V)", **kwargs)
