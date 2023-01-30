
# MagPyPlot

[MagPyPlot](https://github.com/pacesm/MagPyPlot) is an extension of the
Python [Matplotlib](https://matplotlib.org/) providing extra components
to make easier plotting of geomagnetic field and related variables.

## Installation

```
git clone https://github.com/pacesm/MagPyPlot
pip install ./MagPyPlot
```

## Usage


## Magnetic Local Time Polar Axes

```python
from matplotlib.pyplot import subplot, show
from magpyplot import register_projections

# register MagPyPlot projections to Matplotlib
register_projections()


# MLT - North Pole
ax = subplot(1, 2, 1, projection="mlt_north")
ax.plot_night_patch() # optional dimming of the night half of the plot
ax.plot(mlt, lat, '.')


# MLT - South Pole
ax = subplot(1, 2, 1, projection="mlt_south")
ax.plot_night_patch() # optional dimming of the night half of the plot
ax.plot(mlt, lat, '.')

show()
...
