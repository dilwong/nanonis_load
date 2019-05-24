# Interactive colorplot base class
# Drag bar user interface in didv.colorplot

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors

import numpy as np
import pandas as pd

class colorplot():

    def __init__(self):
        self.__draggables__ = []
        self.__drag_h_count__ = 0
        self.__drag_v_count__ = 0
        self.__color_cycle__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.__drag_color_index__ = 0
        self.__colorbar_rectangles__ = []

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

    def clim(self, c_min, c_max):
        self.pcolor.set_clim(c_min, c_max)
        if len(self.__colorbar_rectangles__) != 0:
            self.update_colormap()

    def colormap(self, cmap, change_original = True):
        if type(cmap) == np.ndarray:
            converted_cmap = matplotlib.colors.ListedColormap(cmap)
            self.pcolor.set_cmap(converted_cmap)
        else:
            self.pcolor.set_cmap(cmap)
        if change_original:
            self.original_cmap = self.pcolor.cmap
            self.stop_define_colormap()
            self.__colorbar_rectangles__ = []

    def contour(self):
        c_x, c_y = np.meshgrid(self.bias, self.index_list)
        c_x = c_x.T
        c_y = c_y.T
        self.ax.contour(c_x, c_y, self.data, cmap = 'jet')

    def drag_bar(self, direction = 'horizontal', locator = False, axes = None, color = None):

        if direction[0] == 'h':
            if axes is None:
                if self.__drag_h_count__ == 0:
                    self.__drag_h_fig__ = plt.figure()
                    self.__drag_h_ax__ = self.__drag_h_fig__.add_subplot(111)
                axes = self.__drag_h_ax__
            initial_value = self.ax.get_ylim()[1] - (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.1
            self.__drag_h_count__ += 1
            if locator:
                locator_axes = self.__drag_v_ax__
            else:
                locator_axes = None
        elif direction[0] == 'v':
            if axes is None:
                if self.__drag_v_count__ == 0:
                    self.__drag_v_fig__ = plt.figure()
                    self.__drag_v_ax__ = self.__drag_v_fig__.add_subplot(111)
                axes = self.__drag_v_ax__
            initial_value = self.ax.get_xlim()[0] + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.1
            self.__drag_v_count__ += 1
            if locator:
                locator_axes = self.__drag_h_ax__
            else:
                locator_axes = None
        else:
            print('Direction must be "h" for horizontal or "v" for vertical.')
            return
        if color is None:
            color = self.__color_cycle__[self.__drag_color_index__]
            self.__drag_color_index__ += 1
            self.__drag_color_index__ = self.__drag_color_index__ % len(self.__color_cycle__)

        return drag_bar(self, direction, axes, color, initial_value, self.xlist, self.ylist, locator_axes = locator_axes)

    def define_colormap(self):

        def on_click_ax(event):
            if (event.inaxes == self.ax) and (event.button == 3):
                self.add_colorbar_rectangle(event.xdata, event.ydata)
        
        def on_click_bar(event):
            if event.inaxes == self.color_select_ax:
                for rect in self.__colorbar_rectangles__:
                    if rect.active:
                        rect.orig_norm_value = self.color_select_bar.norm(event.ydata)
                        rect.color = self.original_cmap(rect.orig_norm_value)
                        rect.rect.set_facecolor(rect.color)
                        self.update_colormap()
                        self.fig.canvas.draw()

        def key_press(event):
            if event.key == 'backspace':
                remove_mark = None
                for n, rect in enumerate(self.__colorbar_rectangles__):
                    if rect.active:
                        rect.rect.remove()
                        remove_mark = n
                        break
                if remove_mark is not None:
                    del self.__colorbar_rectangles__[n]
                self.update_colormap()
                self.fig.canvas.draw()

        for rect in self.__colorbar_rectangles__:
            self.ax.add_patch(rect.rect)
        self.__define_colormap_event__ = self.fig.canvas.mpl_connect('button_press_event', on_click_ax)

        self.color_select_fig = plt.figure(figsize=(0.5,5))
        self.color_select_ax = self.color_select_fig.add_subplot(111)
        self.color_select_bar = self.color_select_ax.imshow(np.vstack((np.linspace(0,1,256),np.linspace(0,1,256))).T, cmap = self.original_cmap, origin='lower', extent=(0,.1,0,1))
        self.color_select_ax.set_xticks([],[])
        self.color_select_ax.set_yticks([],[])
        self.color_select_fig.canvas.mpl_connect('button_press_event', on_click_bar)
        self.__colormap_keypress_event__ = self.fig.canvas.mpl_connect('key_press_event', key_press)

    def stop_define_colormap(self):
        try:
            self.fig.canvas.mpl_disconnect(self.__define_colormap_event__)
            self.fig.canvas.mpl_disconnect(self.__colormap_keypress_event__)
            plt.close(self.color_select_fig)
            for rect in self.__colorbar_rectangles__:
                rect.rect.remove()
        except AttributeError:
            pass

    def update_colormap(self):

        cmap_points = [[rect.c_value, rect.orig_norm_value] for rect in self.__colorbar_rectangles__]
        c_min, c_max = self.pcolor.get_clim()
        cmap_points.append([c_min, 0])
        cmap_points.append([c_max, 1])
        cmap_points = sorted(cmap_points)
        new_cmap_array = np.interp(np.linspace(c_min,c_max,256), *list(zip(*cmap_points)))
        new_cmap = matplotlib.colors.ListedColormap(self.original_cmap(new_cmap_array))
        self.colormap(new_cmap, change_original = False)

    def add_colorbar_rectangle(self, x_value, y_value):
        rect = colorbar_rectangle(x_value, y_value, self)
        self.__colorbar_rectangles__.append(rect)
        self.ax.add_patch(rect.rect)
        self.fig.canvas.draw()

    def export_colormap(self):
        c_min, c_max = self.pcolor.get_clim()
        print('C_MIN: ' + str(c_min))
        for n, rect in enumerate(self.__colorbar_rectangles__):
            print('RECT ' + str(n) + ': ' + str(rect.x_tuple[1]) + ', ' + str(rect.y_tuple[1]))
        print('C_MAX: ' + str(c_max))
        return self.pcolor.cmap(np.linspace(0,1,256))

class drag_bar():

    def __init__(self, colorplot, direction, axes, color, initial_value, xlist, ylist, locator_axes = None):

        self.colorplot = colorplot
        self.direction = direction
        self.drag_ax = axes
        self.drag_fig = axes.figure
        self.color = color
        self.xlist = xlist
        self.ylist = ylist
        self.data = colorplot.data
        self.press = False
        self.linked_bars = []
        self.functions = []
        self.fast = False

        self.colorplot.__draggables__.append(self)

        if direction[0] == 'h':
            self.tune_list = self.ylist
            self.indep_list = self.xlist
            axline_function = self.colorplot.ax.axhline
            self.slice_dict = {'left' : slice(None)}
            self.slice_const = 'right'
        elif direction[0] == 'v':
            self.tune_list = self.xlist
            self.indep_list = self.ylist
            axline_function = self.colorplot.ax.axvline
            self.slice_dict = {'right' : slice(None)}
            self.slice_const = 'left'
        else:
            print('Direction must be "h" for horizontal or "v" for vertical.')
            return
        self.index, self.index_value = min(enumerate(self.tune_list), key = lambda x: abs(x[1] - initial_value))
        self.phantom_index_value = self.index_value
        self.slice_dict[self.slice_const] = self.index
        self.colorplot_line = axline_function(self.index_value, color = self.color)
        self.plot, = self.drag_ax.plot(self.indep_list, self.data[self.slice_dict['left'],self.slice_dict['right']], label = str(self.index_value), color = self.color)
        legend = self.drag_ax.legend()
        self.legend_order = len(legend.get_lines()) - 1

        if locator_axes is not None:
            self.locator_axes = locator_axes
            self.locator_line = self.locator_axes.axvline(self.index_value, color = color)
        else:
            self.locator_axes = None

        self.colorplot_line.set_picker(5)
        self.refresh_legend()

        def on_press(event):
            if event.inaxes != self.colorplot.ax:
                return
            contains, _ = self.colorplot_line.contains(event)
            if not contains:
                return
            self.press = True
            self.colorplot.fig.active_drag_bar = self

        def on_motion(event):
            if event.inaxes != self.colorplot.ax:
                return
            if self.press is False:
                return
            if direction[0] == 'h':
                self.move_to(value = event.ydata)
            elif direction[0] == 'v':
                self.move_to(value = event.xdata)

        def on_release(event):
            self.refresh_legend()
            self.press = False
            if not self.fast:
                self.colorplot.fig.canvas.draw()
            self.phantom_index_value = self.index_value

        def key_press(event):
            try:
                if (self.colorplot.fig.active_drag_bar is not None) and (self.colorplot.fig.active_drag_bar is self):
                    if (event.key == 'up') or (event.key == 'down'):
                        try:
                            if event.key == 'up':
                                if self.tune_list[1] > self.tune_list[0]:
                                    step = 1
                                else:
                                    step = -1
                            if event.key == 'down':
                                if self.tune_list[1] > self.tune_list[0]:
                                    step = -1
                                else:
                                    step = 1
                            self.index = self.index + step
                            self.move_to(index = self.index)
                        except IndexError:
                            pass
            except AttributeError:
                pass

        self.colorplot.fig.canvas.mpl_connect('button_press_event', on_press)
        self.colorplot.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.colorplot.fig.canvas.mpl_connect('button_release_event', on_release)
        self.colorplot.fig.canvas.mpl_connect('key_press_event', key_press)

        def pick_line(event):
            if self.legend is event.artist:
                plot_line = self.plot
            else:
                return
            visibility = not plot_line.get_visible()
            plot_line.set_visible(visibility)
            plot_line.figure.canvas.draw()

        self.drag_fig.canvas.mpl_connect('pick_event', pick_line)

    def move_to(self, index = None, value = None, skip_draw = False):
        past_index_value = self.index_value
        if index is None:
            if value is not None:
                self.index, self.index_value = min(enumerate(self.tune_list), key = lambda x: (abs(x[1] - value), -x[0]))
            else:
                print('Error: No index or value supplied to move_to')
        else:
            self.index_value = self.tune_list[self.index]
        if self.direction[0] == 'h':
            set_data_function = self.colorplot_line.set_ydata
        elif self.direction[0] == 'v':
            set_data_function = self.colorplot_line.set_xdata
        set_data_function([self.index_value, self.index_value])
        self.slice_dict[self.slice_const] = self.index
        self.plot.set_ydata(self.data[self.slice_dict['left'],self.slice_dict['right']])
        self.plot.set_label(str(self.index_value))
        if (len(self.linked_bars) == 0) and (not skip_draw):
            self.drag_ax.legend()
            self.refresh_legend()
            self.drag_fig.canvas.draw()
            self.colorplot.fig.canvas.draw()
        if self.locator_axes is not None:
                self.locator_line.set_xdata([self.index_value, self.index_value])
                self.locator_axes.figure.canvas.draw()
        index_value_diff = self.index_value - past_index_value
        if len(self.linked_bars) != 0:
            for bar in self.linked_bars:
                bar.phantom_index_value = bar.phantom_index_value + index_value_diff
                bar.move_to(value = bar.phantom_index_value, skip_draw = True)
            self.drag_ax.legend()
            self.refresh_legend()
            self.drag_fig.canvas.draw()
            if self.fast:
                self.colorplot.fig.canvas.restore_region(self.background)
                self.colorplot.ax.draw_artist(self.colorplot_line)
                for bar in self.linked_bars:
                    self.colorplot.ax.draw_artist(bar.colorplot_line)
                self.colorplot.fig.canvas.blit(self.colorplot.fig.bbox)
            else:
                self.colorplot.fig.canvas.draw()
        if len(self.functions) != 0:
            for func in self.functions:
                func()

    def refresh_legend(self):
        for bar in self.colorplot.__draggables__:
            if bar.drag_ax is self.drag_ax:
                bar.drag_ax.get_legend().get_lines()[bar.legend_order].set_visible(True)
                bar.drag_ax.get_legend().get_lines()[bar.legend_order].set_picker(5)
                bar.legend = bar.drag_ax.get_legend().get_lines()[bar.legend_order]

    def join_drag_bars(self, *drag_bar_list): # Only works if colorplots on the same figure, line traces on the same axes
        for bar in drag_bar_list:
            self.linked_bars.append(bar)

    def to_clipboard(self):
        pd.DataFrame([self.indep_list, self.data[self.slice_dict['left'],self.slice_dict['right']]]).transpose().to_clipboard(index=False, header =False)

class colorbar_rectangle():

    def __init__(self, x_value, y_value, colorplot):
        
        self.colorplot = colorplot
        x_tuple = min(enumerate(self.colorplot.xlist), key = lambda x: (abs(x[1] - x_value), -x[0]))
        y_tuple = min(enumerate(self.colorplot.ylist), key = lambda x: (abs(x[1] - y_value), -x[0]))
        c_value = self.colorplot.data[x_tuple[0], y_tuple[0]]
        
        self.x_tuple = x_tuple
        self.y_tuple = y_tuple
        self.c_value = c_value
        self.ax = colorplot.ax
        self.colorbar = colorplot.colorbar
        self.press = False

        self.color = self.colorplot.colorbar.cmap(self.colorplot.colorbar.norm(c_value)) # set_norm methods are bugged in matplotlib, so don't mess with self.colorplot.pcolor.norm directly
        c_min, c_max = self.colorplot.pcolor.get_clim()
        if c_value >= c_max:
            self.orig_norm_value = 1.0
        elif c_value <= c_min:
            self.orig_norm_value = 0.0
        elif np.isnan(c_value):
            self.orig_norm_value = 0.0
        else:
            self.orig_norm_value = np.interp(c_value, self.colorplot.pcolor.get_clim(), (0,1))
        
        width, height = self.ax.transData.inverted().transform(self.ax.transAxes.transform((0.025,0.025))) - self.ax.transData.inverted().transform(self.ax.transAxes.transform((0,0)))
        self.rect = matplotlib.patches.Rectangle((x_tuple[1] - width * 0.5, y_tuple[1] - height * 0.5), width, height, edgecolor='k', facecolor=self.color)
        self.width = width
        self.height = height

        for rect in self.colorplot.__colorbar_rectangles__:
            rect.active = False
        self.active = True

        def on_press(event):
            if event.inaxes != self.rect.axes:
                return
            contains, _ = self.rect.contains(event)
            if not contains:
                return
            self.press = True
            for rect in self.colorplot.__colorbar_rectangles__:
                rect.active = False
            self.active = True

        def on_motion(event):
            if not self.press:
                return
            if event.inaxes != self.rect.axes:
                return
            self.x_tuple = min(enumerate(self.colorplot.xlist), key = lambda x: (abs(x[1] - event.xdata), -x[0]))
            self.y_tuple = min(enumerate(self.colorplot.ylist), key = lambda x: (abs(x[1] - event.ydata), -x[0]))
            self.c_value = self.colorplot.data[self.x_tuple[0], self.y_tuple[0]]
            self.rect.set_x(self.x_tuple[1] - self.width * 0.5)
            self.rect.set_y(self.y_tuple[1] - self.height * 0.5)
            self.colorplot.update_colormap()
            self.colorplot.fig.canvas.draw()

        def on_release(event):
            self.press = False
            self.colorplot.fig.canvas.draw()

        self.colorplot.fig.canvas.mpl_connect('button_press_event', on_press)
        self.colorplot.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.colorplot.fig.canvas.mpl_connect('button_release_event', on_release)