# nanonis_load/gapmap.py
from __future__ import annotations
from . import didv
from . import dual_gate as dg

import glob
import itertools
import copy
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.widgets


class Gapmap:
    """
    Build a gap-map from a collection of didv.Spectrum objects or from files matching
    a filename pattern.

    Typical notebook usage converted:
        gm = GapMap(filename_pattern="CS116_tip4_gapmap4", wildcard=True)
        fig, ax = gm.plot(vmin=10, vmax=60, cmap='RdBu_r')

    Parameters
    ----------
    spectra_or_pattern :
        Either a sequence of didv.Spectrum objects OR a filename pattern string
        (e.g. "CS116_tip4_gapmap4" which will be globbed as "CS116_tip4_gapmap4*.dat")
    channel :
        channel name used in Spectrum.data (kept for compatibility; not used directly)
    gaussian_filter_order :
        passed to get_gap_size if you want to customize (default 2 in your notebook)
    current_threshold :
        passed to get_gap_size (default 2e-12 as in your notebook)
    """

    def __init__(
        self,
        spectra_or_pattern: Union[str, Sequence[didv.Spectrum]],
        *,
        channel: str = "Current (A)",
        gaussian_filter_order: int = 2,
        current_threshold: float = 2e-12,
        glob_wildcard: bool = True,
    ):
        self.channel = channel
        self.gaussian_filter_order = gaussian_filter_order
        self.current_threshold = current_threshold

        self.delta_V_g = []
        self.delta_V_m = []
        self.V_g_offset = 0
        self.V_m_offset = 0

        # Accept either a list of Spectrum objects or a filename pattern string
        if isinstance(spectra_or_pattern, str):
            pattern = f"{spectra_or_pattern}*.dat" if glob_wildcard else spectra_or_pattern
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {pattern}")
            # instantiate didv.Spectrum for each file
            self.spectra: List[didv.Spectrum] = [didv.Spectrum(fn) for fn in files]
        else:
            # assume it's an iterable of Spectrum objects
            self.spectra = list(spectra_or_pattern)
            if not self.spectra:
                raise ValueError("Provided spectra iterable is empty")

        # Precompute arrays used for mapping (mirrors your notebook workflow)
        self._compute_basic_arrays()

    def _compute_basic_arrays(self):
        """Compute arrays: current, Z, V_g, V_m, gap_sizes and mask/filter used later."""
        # current array (not used directly for plotting but preserved for API)
        self.current = np.array([spectrum.data[self.channel] for spectrum in self.spectra])

        # Bias array from the first spectrum. Assumes all spectra were taken at the same bias values
        self.biases = np.array(self.spectra[0].data['Bias calc (V)'])

        # Z (header 'Z (m)')
        self.Z = np.array([float(spectrum.header.get("Z (m)", np.nan)) for spectrum in self.spectra])

        # mask similar to your notebook: Z < 0e-9
        self._filter_mask = self.Z < 0e-9

        # gate voltages (first and second gates) filtered
        self.V_g = np.array([spectrum.gate_voltage for spectrum in self.spectra])[self._filter_mask]
        self.V_m = np.array([spectrum.second_gate for spectrum in self.spectra])[self._filter_mask]

        # compute gap sizes using didv.Spectrum.get_gap_size
        # we compute for all spectra, then apply the same filter
        gap_list = np.array(
            [
                spectrum.get_gap_size(current_threshold=self.current_threshold, gaussian_filter_order=self.gaussian_filter_order)
                for spectrum in self.spectra
            ]
        )
        self.gap_sizes = gap_list[self._filter_mask]

        # unique sorted grids
        self.unique_V_g = np.array(sorted(list(set(self.V_g))))
        self.unique_V_m = np.array(sorted(list(set(self.V_m))))

    def compute_grid_interpolation(self, method: str = "nearest", rescale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate gap_sizes onto cartesian grid (V_m, V_g) and return:
            unique_V_g (x axis), unique_V_m (y axis), interp_values (flattened in product order)
        """
        # build pairs exactly as notebook: product(unique_V_m, unique_V_g)
        pairs = list(itertools.product(self.unique_V_m, self.unique_V_g))
        unique_pairs = sorted(list(set(pairs)))
        # perform interpolation
        interp_values = scipy.interpolate.griddata(
            np.c_[self.V_m, self.V_g],  # points (V_m, V_g)
            self.gap_sizes,             # values
            unique_pairs,                      # query points
            method=method,
            rescale=rescale,
        )
        return self.unique_V_g, self.unique_V_m, np.array(interp_values)
    
    def compute_bias_slice_interpolation(self, bias, gradient=False, method: str = 'nearest'):
        """
        Returns the current (or its gradient) at a bias slice
        """
        pairs = list(itertools.product(self.unique_V_m, self.unique_V_g))
        unique_pairs = sorted(list(set(pairs)))
        bias_index = np.argmin(np.abs(self.biases - bias))
        if gradient:
            current_slice = np.gradient(self.current, axis=-1)[:, bias_index]
        else:
            current_slice = self.current[:, bias_index]
        interp_values = scipy.interpolate.griddata(
            np.c_[self.V_m, self.V_g],  # points (V_m, V_g)
            current_slice,             # values
            unique_pairs,                      # query points
            method=method,
            rescale=True,
        )
        return interp_values
    
    def capacitance_calculator(self):
        delta_V_g = self.delta_V_g
        delta_V_m = self.delta_V_m
        c_g = 1 / abs(delta_V_g[0] - delta_V_g[1])
        c_t = 1 / abs(delta_V_m[0] - delta_V_m[1])

        self.c_g = c_g
        self.c_t = c_t

        return self.c_g, self.c_t
        
    
    def filling_factor_convert(self, V_g, V_m):
        reshaped_V_g = V_g - self.V_g_offset
        reshaped_V_m = V_m - self.V_m_offset

        c_g, c_t = self.capacitance_calculator()

        total_filling_factor = c_g * reshaped_V_g
        delta_filling_factor = 2 * c_t * reshaped_V_m - c_g * reshaped_V_g

        v_t = (total_filling_factor + delta_filling_factor) / 2
        v_m = (total_filling_factor - delta_filling_factor) / 2

        return total_filling_factor, delta_filling_factor, v_t, v_m, c_g, c_t
    
    
    def invert_filling_factor_calculator(self, constant: list, variable: list, N: int = 50):
        """
        Build V_g and V_m ranges given one fixed quantity (constant) and the variable range (variable).
        'constant' should be like ['total_filling_factor', value] or ['v_t', value]
        'variable' should be a 2-element iterable [start, end] specifying the range for the other axis.
        Returns V_g, V_m arrays of length N.
        """
        if not (isinstance(constant, (list, tuple)) and len(constant) >= 2):
            raise ValueError("constant must be a list or tuple like ['name', value].")

        name = constant[0]
        val = constant[1]

        if name in ["total_filling_factor", "total"]:
            total_filling_factor = [val, val]
            delta_filling_factor = list(variable)

        elif name in ["delta_filling_factor", "delta"]:
            delta_filling_factor = [val, val]
            total_filling_factor = list(variable)

        elif name in ["v_t", "top_filling_factor"]:
            v_t_fixed = [val, val]
            v_m_var = list(variable)
            total_filling_factor = [vt + vm for vt, vm in zip(v_t_fixed, v_m_var)]
            delta_filling_factor = [vt - vm for vt, vm in zip(v_t_fixed, v_m_var)]

        elif name in ["v_m", "middle_filling_factor"]:
            v_m_fixed = [val, val]
            v_t_var = list(variable)
            total_filling_factor = [vt + vm for vt, vm in zip(v_t_var, v_m_fixed)]
            delta_filling_factor = [vt - vm for vt, vm in zip(v_t_var, v_m_fixed)]

        else:
            raise ValueError(f"Unknown constant name: {name}")

        c_g, c_t = self.capacitance_calculator()

        total = np.linspace(float(total_filling_factor[0]), float(total_filling_factor[1]), N)
        delta = np.linspace(float(delta_filling_factor[0]), float(delta_filling_factor[1]), N)

        V_g = total / c_g
        V_m = (delta + c_g * V_g) / (2.0 * c_t)

        return V_g, V_m   


    def plot_gap_size(
        self,
        ax: Optional[plt.Axes] = None,
        vmin: Optional[float] = 10.0,
        vmax: Optional[float] = 60.0,
        cmap: str = "RdBu_r",
        colorbar: bool = True,
        xlabel: str = "$V_g$ (V)",
        ylabel: str = "$V_m$ (mV)",
        figsize: Tuple[int, int] = (6, 4),
        shading: str = "nearest",
        interp_method: str = "nearest",
        interp_rescale: bool = True,
        total_filling_factor : bool = False,
        top_filling_factor : bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a pcolormesh gap map like in your notebook.

        Returns (fig, ax)
        """
        uniq_Vg, uniq_Vm, interp_vals = self.compute_grid_interpolation(method=interp_method, rescale=interp_rescale)

        # reshape into rows = len(unique_V_m), cols = len(unique_V_g)
        Z = interp_vals.reshape(-1, len(uniq_Vg)) * 1000
                
        v_g_2d, v_m_2d = np.meshgrid(uniq_Vg, uniq_Vm)

        self.v_g_2d = v_g_2d
        self.v_m_2d = v_m_2d
        self.gaps = Z

        # convert V_m to mV for plotting
        uniq_Vm_mV = uniq_Vm * 1000.0

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if total_filling_factor == True :
            total, delta, _, _, _, _ = self.filling_factor_convert(self.v_g_2d, self.v_m_2d)
            pcm = ax.pcolormesh(total, delta, Z , shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            xlabel = "Total filling factor"
            ylabel = r"$\Delta \nu$"

        elif top_filling_factor == True :
            _, _, v_t, v_m, _, _ = self.filling_factor_convert(self.v_g_2d, self.v_m_2d)
            pcm = ax.pcolormesh(v_t, v_m, Z , shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
            xlabel = r"$\nu top$"
            ylabel = r"$\nu middle$"

        else :
            pcm = ax.pcolormesh(sorted(uniq_Vg), sorted(uniq_Vm_mV), Z , shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(pcm, ax=ax) if colorbar else None

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if cbar is not None:
            cbar.set_label("Gap size (mV)")

        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

        return fig, ax


    def plot_bias_slice(self, fig=None, ax=None, vmin=None, vmax=None, gradient=True, cmap='RdBu_r',  xlabel: str = "$V_g$ (V)",  ylabel: str = "$V_m$ (V)", auto_clim=True):
        if ax is None:
            fig, ax = plt.subplots()

        fig.subplots_adjust(bottom=0.25)
        
        slider_ax = fig.add_axes([0.25, 0.01, 0.65, 0.03])
        self.bias_slider = matplotlib.widgets.Slider(
            ax=slider_ax,
            label="Bias (V)",
            valmin=np.amin(self.biases),
            valmax=np.amax(self.biases),
            valinit=np.amin(self.biases)
        )

        bias_slice = self.compute_bias_slice_interpolation(np.amin(self.biases), gradient=gradient).reshape((-1, len(self.unique_V_g)))
        
        pcolor = ax.pcolormesh(sorted(self.unique_V_g), sorted(self.unique_V_m), bias_slice, vmin=vmin if not auto_clim else np.quantile(bias_slice, 0.01), vmax=vmax if not auto_clim else np.quantile(bias_slice, 0.99), cmap=cmap)
        cbar = fig.colorbar(pcolor)
        def slider_update(bias):
            new_slice = self.compute_bias_slice_interpolation(bias, gradient=gradient)
            pcolor.set_array(new_slice)
            if auto_clim:
                pcolor.set_clim(np.quantile(new_slice, 0.01), np.quantile(new_slice, 0.99))
            fig.canvas.draw_idle()

        self.bias_slider.on_changed(slider_update)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

        if cbar is not None:
            cbar.set_label("Current (A)")


        return fig, ax