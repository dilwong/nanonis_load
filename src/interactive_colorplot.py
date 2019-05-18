# Interactive colorplot base class
# Drag bar user interface in didv.colorplot
# TO DO: change grid.linecut to use this interface

import matplotlib.pyplot as plt

class colorplot():

    def __init__(self):
        pass

    def xlim(self, x_min, x_max):
        self.ax.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.ax.set_ylim(y_min, y_max)

    def clim(self, c_min, c_max):
        self.pcolor.set_clim(c_min, c_max)

    def colormap(self, cmap):
        self.pcolor.set_cmap(cmap)

    def contour(self):
        c_x, c_y = np.meshgrid(self.bias, self.index_list)
        c_x = c_x.T
        c_y = c_y.T
        self.ax.contour(c_x, c_y, self.data, cmap = 'jet')

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
            self.colorplot.fig.canvas.draw() # TO DO: Optimize using blitting
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
            self.colorplot.fig.canvas.draw()

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
        import pandas as pd
        pd.DataFrame([self.indep_list, self.data[self.slice_dict['left'],self.slice_dict['right']]]).transpose().to_clipboard(index=False, header =False)

