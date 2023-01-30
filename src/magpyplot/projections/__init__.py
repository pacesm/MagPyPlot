"""
   MagPyPlot custom axes types and projections.
"""

from numpy import asarray
from matplotlib.projections import register_projection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from .polar import PolarAxes


class PolarAxesMLT(PolarAxes):
    """ Base polar Magnetic Local Time axes. """
    name = "mlt"
    theta_max = 24 # hours
    theta_unit = "h"
    theta_label = "MLT"
    r_unit = "\N{DEGREE SIGN}"
    r_label = "lat"

    def __init__(self, *args, theta_offset=-6, theta_direction=+1,
                 rlabel_position=112.5, **kwargs):
        super().__init__(
            *args, theta_offset=theta_offset, theta_direction=theta_direction,
            rlabel_position=rlabel_position, **kwargs)

    def plot_night_patch(self, color="whitesmoke", **kwargs):
        """ Add dimming for the night half for the MLT plot.

        To get the correct side of the arc patch set the limit of the r-axis
        before calling this method.

        Parameters
        ----------
        color : patch color

        Returns
        -------
        patch: created `matplotlib.patch.PathPatch` object

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `matplotlib.patch.Patch` properties.

        """
        r_origin = self.get_rorigin()
        _, r_max = self.get_ylim()
        vertices = asarray([
            (-6, r_origin), (-6, r_max), (6, r_max), (6, r_origin),
        ], dtype="float64")
        # set _interpolation_steps > 1 to draw an arch
        return self.add_patch(PathPatch(
            Path(vertices, _interpolation_steps=2), color=color, **kwargs))

register_projection(PolarAxesMLT)


class PolarAxesMLTNorth(PolarAxesMLT):
    """ North polar Magnetic Local Time axes. """
    name = "mlt_north"
    r_min = +90.0
    r_max = +35.0

register_projection(PolarAxesMLTNorth)


class PolarAxesMLTSouth(PolarAxesMLT):
    """ South polar Magnetic Local Time axes. """
    name = "mlt_south"
    r_min = -90.0
    r_max = -35.0


def register_projections():
    """ Register MagPyPlot custom projections. """

    for projection in [
        PolarAxesMLT,
        PolarAxesMLTNorth,
        PolarAxesMLTSouth,
    ]:
        register_projection(projection)
