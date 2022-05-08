# nanonis_load

A library for loading, plotting, and analyzing data from Nanonis SPM files. This library is written to be compatible with both Python 2.7 and Python 3.

The following file types are supported:  
> *.dat Bias Spectroscopy files  
> *.3ds Grid Spectroscopy files  
> *.sxm Image files  

To load and plot .dat files:
```
spec = didv.spectrum('FILENAME.dat')
specPlot = didv.plot(spec, channel = 'Input 2 (V)')
```
didv.plot takes individual spectrum objects or a list of spectrum objects.  
You can click on the line symbol on the legend to show or hide an individual spectrum.

To plot $dI/dV(V_s, V_g)$ colorplots, where $V_s$ is the sample bias and $V_g$ is gate voltage annotated in the header of each .dat file:
```
cPlot = didv.colorplot(BASENAME)
bar = cPlot.drag_bar(direction = 'horizontal')
cPlot.xlim(XMIN, XMAX)
cPlot.ylim(YMIN, YMAX)
cPlot.clim(CMIN, MAX)
```

Nanonis image files (.sxm) can be loaded and plotted as follows:
```
imageData = sxm.sxm('FILENAME.SXM')
imagePlot = sxm.plot(imageData, channel = 'Z (m)')
imagePlot.fft()
```

And 2D Nanonis binary files (.3ds) can be loaded and plotted using the 'grid' module:
```
gridData = grid.nanonis_3ds('FILENAME.3ds')
gridPlot = grid.plot(gridData, channel = 'Input 2 (V)', fft = True)
```
UP and DOWN arrow keys on the keyboard can be used to iterate over different energies.  
1D line grids can be loaded and plotted using the grid.linecut class.

There are two different ways of creating Landau fans in this library: didv.landau_fan and magnetoconductance.landau_fan.