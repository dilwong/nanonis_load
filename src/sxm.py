#Imports Nanonis .sxm files into Python

import numpy
import os.path
import re

import matplotlib
import matplotlib.pyplot

import scipy
import scipy.signal

#Loads .sxm files into Nanonis
#load_sxm(filename).header contains Nanonis header information
#load_sxm(filename).data contains Numpy arrays with data
class load_sxm():

    #This class is a modified version of
    #https://github.com/fbianco/thoth/blob/master/nanonisfile.py
    #but with some serious bugs fixed.
    def __init__(self, filename):

        self.data = []
        self.header = {}
        self.header['filename'] = filename

        self.file = open(os.path.normpath(self.header['filename']),'rb')

        s1 = self.file.readline()
        if not re.match(':NANONIS_VERSION:',s1):
                print "The file %s does not have the Nanonis SXM" % self.header['filename']
                return

        self.header['version'] = int(self.file.readline())

        while True:
            line = self.file.readline().strip()

            if re.match('^:.*:$', line):
                tagname = line[1:-1]
            else:
                if 'Z-CONTROLLER' == tagname:
                    keys = line.split('\t')
                    values = self.file.readline().strip().split('\t')
                    self.header['z-controller'] = dict(zip(keys,values))
                elif tagname in ('BIAS','REC_TEMP','ACQ_TIME','SCAN_ANGLE'):
                    self.header[tagname.lower()] = float(line)
                elif tagname in ('SCAN_PIXELS','SCAN_TIME','SCAN_RANGE','SCAN_OFFSET'):
                    self.header[tagname.lower()] = [float(i) for i in re.split('\s+', line)]
                elif 'DATA_INFO' == tagname:
                    if 1 == self.header['version']:
                        keys = re.split('\s\s+',line)
                    else:
                        keys = line.split('\t')
                    self.header['data_info'] = []

                    while True:
                        line = self.file.readline().strip()
                        if not line:
                            break
                        values = line.strip().split('\t')
                        self.header['data_info'].append(dict(zip(keys,values)))
                elif tagname in ('SCANIT_TYPE','REC_DATE','REC_TIME','SCAN_FILE','SCAN_DIR'):
                    self.header[tagname.lower()] = line
                elif 'SCANIT_END' == tagname:
                    break
                else:
                    if not self.header.has_key(tagname.lower()):
                        self.header[tagname.lower()] = line
                    else:
                        self.header[tagname.lower()] += '\n' + line

        s = '\x00\x00'
        while '\x1A\x04' != s:
            s = s[1] + self.file.read(1)

        size = int(self.header['scan_pixels'][0]*self.header['scan_pixels'][1]*4) # 4 Bytes/px

        nchannels = len(self.header['data_info'])
        supp = 0
        for n in range(nchannels):
            supp += ('both' == self.header['data_info'][n]['Direction'])
        nchannels+=supp

        for i in range(nchannels):
            data_buffer = self.file.read(size)
            self.header['channel'] = i
            self.data.append(numpy.ndarray(shape=numpy.flipud(self.header['scan_pixels']),dtype='>f',buffer=data_buffer))

        return

#plot_sxm(filename,channel,direction)
#channel can be 'z', 'current', or whatever data is stored
#direction = 0 for forward, direction = 1 for backwards
class plot_sxm():

    #This class is named after plot_sxm.m
    def __init__(self,filename,channel,direction = 0,flatten = 1):

        self.header = {}
        
        #If input filename is integer xxx, the read 'mxxx.sxm'
        if type(filename) == str:
            new_filename = filename
        elif type(filename) == int:
            new_filename = 'm' + '%0*d' % (3, filename) + '.sxm'
        else:
            return

        #Load .sxm file data
        nanonis = load_sxm(new_filename)
        self.header['x_center (nm)']=nanonis.header['scan_offset'][0]*1e9
        self.header['y_center (nm)']=nanonis.header['scan_offset'][1]*1e9
        data_index = [data_type['Name'].lower() for data_type in nanonis.header['data_info']].index(channel.lower())
        channel_index = 2*data_index + direction
        image_data=nanonis.data[channel_index]

        #Flip upside down if image was taken scanning down
        if nanonis.header['scan_dir'] == 'down':
            image_data=numpy.flipud(image_data)
        
        #Flatten
        image_data.flags.writeable = True
        avg_dat = image_data[~numpy.isnan(image_data)].mean()
        #image_data = numpy.nan_to_num(image_data)
        image_data[numpy.isnan(image_data)] = avg_dat
        if flatten == 1:
            image_data=scipy.signal.detrend(image_data)

        #Plot it!
        self.header['x_range (nm)']=nanonis.header['scan_range'][0]*1e9
        self.header['y_range (nm)']=nanonis.header['scan_range'][1]*1e9
        image_x=numpy.linspace(0,self.header['y_range (nm)'],nanonis.header['scan_pixels'][1])
        image_y=numpy.linspace(0,self.header['x_range (nm)'],nanonis.header['scan_pixels'][0])
        ax=self.figure=matplotlib.pyplot.figure()
        self.plot=self.figure.add_subplot(111, aspect="equal")
        self.plot.set_xlabel('X (nm)')
        self.plot.set_ylabel('Y (nm)')
        title = new_filename + ', Vs = ' + str(nanonis.header['bias']) + 'V, I = ' \
                + str(float(nanonis.header['z-controller>setpoint'])*1e9) + 'nA'
        self.plot.set_title(title)
        ax=self.plot.pcolor(image_y,image_x,image_data)
        
        cbar=matplotlib.pyplot.colorbar(ax)
        self.cycle = sorted([i for i in dir(matplotlib.pyplot.cm) \
                             if hasattr(getattr(matplotlib.pyplot.cm,i),'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)
        self.free=0
        self.press=None
        def key_press(event):
            if event.key[0:4] == 'alt+':
                key=event.key[4:]
            else:
                key=event.key
            #color_schemes=['jet','pink','gray','Accent','BuGn', \
            #               'BuPu','PiYG','Purples','Greens', \
            #               'Spectral','cool','copper','hot', \
            #               'spectral','spring','summer','winter']
            if key == 'right':
                self.index += 1
            elif key == 'left':
                self.index -= 1
            if self.index < 0:
                self.index = len(self.cycle)-1
            elif self.index >= len(self.cycle):
                self.index = 0
            cmap = self.cycle[self.index]
            cbar.set_cmap(cmap)
            cbar.draw_all()
            ax.set_cmap(cmap)
            ax.get_axes().set_title(title)
            cbar.patch.figure.canvas.draw()
        def on_press(event):
            if event.inaxes != cbar.ax:
                return
            self.press = event.x, event.y
        def on_motion(event):
            if self.press is None: return
            if event.inaxes != cbar.ax: return
            xprev, yprev = self.press
            dx = event.x - xprev
            dy = event.y - yprev
            self.press = event.x,event.y
            scale = cbar.norm.vmax - cbar.norm.vmin
            perc = 0.03
            if event.button==1:
                cbar.norm.vmin -= (perc*scale)*numpy.sign(dy)
                cbar.norm.vmax -= (perc*scale)*numpy.sign(dy)
            elif event.button==3:
                cbar.norm.vmin -= (perc*scale)*numpy.sign(dy)
                cbar.norm.vmax += (perc*scale)*numpy.sign(dy)
            cbar.draw_all()
            ax.set_norm(cbar.norm)
            cbar.patch.figure.canvas.draw()
        def on_release(event):
            self.press = None
            ax.set_norm(cbar.norm)
            cbar.patch.figure.canvas.draw()
        self.figure.canvas.mpl_connect('button_press_event',on_press)
        self.figure.canvas.mpl_connect('key_press_event',key_press)
        self.figure.canvas.mpl_connect('button_release_event',on_release)
        self.figure.canvas.mpl_connect('motion_notify_event',on_motion)
        self.figure.show()
        
        return
