"""
Custom polar projection class designed to be extensible.

Derived from matplotlib/lib/matplotlib/projections/polar.py
(commit 0ec308d7fad16055f1abe5648712e7b8d8043d14)

See https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE

Copyright (c) 2012-2023 Matplotlib Development Team
Copyright (c) 2002-2011 John D. Hunter
"""

import math
import types

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine


class PolarTransform(mtransforms.Transform):
    r"""
    The base polar transform.

    This transform maps polar coordinates :math:`\theta, r` into Cartesian
    coordinates :math:`x, y = r \cos(\theta), r \sin(\theta)`
    (but does not fully transform into Axes coordinates or
    handle positioning in screen space).

    This transformation is designed to be applied to data after any scaling
    along the radial axis (e.g. log-scaling) has been applied to the input
    data.

    Path segments at a fixed radius are automatically transformed to circular
    arcs as long as ``path._interpolation_steps > 1``.
    """

    input_dims = output_dims = 2

    def __init__(self, axis=None, use_rmin=True,
                 _apply_theta_transforms=True, *, scale_transform=None):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`, optional
            Axis associated with this transform. This is used to get the
            minimum radial limit.
        use_rmin : `bool`, optional
            If ``True``, subtract the minimum radial axis limit before
            transforming to Cartesian coordinates. *axis* must also be
            specified for this to take effect.
        """
        super().__init__()
        self._axis = axis
        self._use_rmin = use_rmin
        self._apply_theta_transforms = _apply_theta_transforms
        self._scale_transform = scale_transform

    __str__ = mtransforms._make_str_method(
        "_axis",
        use_rmin="_use_rmin",
        _apply_theta_transforms="_apply_theta_transforms")

    def _get_rorigin(self):
        # Get lower r limit after being scaled by the radial scale transform
        return self._scale_transform.transform(
            (0, self._axis.get_rorigin()))[1]

    def transform_non_affine(self, tr):
        # docstring inherited
        theta, r = np.transpose(tr)

        # Convert theta units to radians.
        theta = self._axis.units2radians(theta)
        # PolarAxes does not use the theta transforms here, but apply them for
        # backwards-compatibility if not being used by it.
        if self._apply_theta_transforms and self._axis is not None:
            theta *= self._axis.get_theta_direction()
            theta += self._axis.get_theta_offset()
        if self._use_rmin and self._axis is not None:
            r = (r - self._get_rorigin()) * self._axis.get_rsign()
        r = np.where(r >= 0, r, np.nan)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    def transform_path_non_affine(self, path):
        print(path)
        # docstring inherited
        if not len(path) or path._interpolation_steps == 1:
            return Path(self.transform_non_affine(path.vertices), path.codes)
        xys = []
        codes = []
        last_t = last_r = None
        for trs, c in path.iter_segments():
            trs = trs.reshape((-1, 2))
            if c == Path.LINETO:
                (t, r), = trs
                if t == last_t:  # Same angle: draw a straight line.
                    xys.extend(self.transform_non_affine(trs))
                    codes.append(Path.LINETO)
                elif r == last_r:  # Same radius: draw an arc.
                    # The following is complicated by Path.arc() being
                    # "helpful" and unwrapping the angles, but we don't want
                    # that behavior here.
                    last_td = np.rad2deg(self._axis.units2radians(last_t))
                    td = np.rad2deg(self._axis.units2radians(t))
                    if self._use_rmin and self._axis is not None:
                        r = ((r - self._get_rorigin())
                             * self._axis.get_rsign())
                    if last_td <= td:
                        while td - last_td > 360:
                            arc = Path.arc(last_td, last_td + 360)
                            xys.extend(arc.vertices[1:] * r)
                            codes.extend(arc.codes[1:])
                            last_td += 360
                        arc = Path.arc(last_td, td)
                        xys.extend(arc.vertices[1:] * r)
                        codes.extend(arc.codes[1:])
                    else:
                        # The reverse version also relies on the fact that all
                        # codes but the first one are the same.
                        while last_td - td > 360:
                            arc = Path.arc(last_td - 360, last_td)
                            xys.extend(arc.vertices[::-1][1:] * r)
                            codes.extend(arc.codes[1:])
                            last_td -= 360
                        arc = Path.arc(td, last_td)
                        xys.extend(arc.vertices[::-1][1:] * r)
                        codes.extend(arc.codes[1:])
                else:  # Interpolate.
                    trs = cbook.simple_linear_interpolation(
                        np.row_stack([(last_t, last_r), trs]),
                        path._interpolation_steps)[1:]
                    xys.extend(self.transform_non_affine(trs))
                    codes.extend([Path.LINETO] * len(trs))
            else:  # Not a straight line.
                xys.extend(self.transform_non_affine(trs))
                codes.extend([c] * len(trs))
            last_t, last_r = trs[-1]
        return Path(np.asarray(xys), codes)

    def inverted(self):
        # docstring inherited
        return PolarAxes.InvertedPolarTransform(self._axis, self._use_rmin,
                                                self._apply_theta_transforms)


class PolarAffine(mtransforms.Affine2DBase):
    r"""
    The affine part of the polar projection.

    Scales the output so that maximum radius rests on the edge of the axes
    circle and the origin is mapped to (0.5, 0.5). The transform applied is
    the same to x and y components and given by:

    .. math::

        x_{1} = 0.5 \left [ \frac{x_{0}}{(r_{\max} - r_{\min})} + 1 \right ]

    :math:`r_{\min}, r_{\max}` are the minimum and maximum radial limits after
    any scaling (e.g. log scaling) has been removed.
    """
    def __init__(self, scale_transform, limits):
        """
        Parameters
        ----------
        scale_transform : `~matplotlib.transforms.Transform`
            Scaling transform for the data. This is used to remove any scaling
            from the radial view limits.
        limits : `~matplotlib.transforms.BboxBase`
            View limits of the data. The only part of its bounds that is used
            is the y limits (for the radius limits).
        """
        super().__init__()
        self._scale_transform = scale_transform
        self._limits = limits
        self.set_children(scale_transform, limits)
        self._mtx = None

    __str__ = mtransforms._make_str_method("_scale_transform", "_limits")

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            limits_scaled = self._limits.transformed(self._scale_transform)
            yscale = limits_scaled.ymax - limits_scaled.ymin
            affine = mtransforms.Affine2D() \
                .scale(0.5 / yscale) \
                .translate(0.5, 0.5)
            self._mtx = affine.get_matrix()
            self._inverted = None
            self._invalid = 0
        return self._mtx


class InvertedPolarTransform(mtransforms.Transform):
    """
    The inverse of the polar transform, mapping Cartesian
    coordinate space *x* and *y* back to *theta* and *r*.
    """
    input_dims = output_dims = 2

    def __init__(self, axis=None, use_rmin=True,
                 _apply_theta_transforms=True):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`, optional
            Axis associated with this transform. This is used to get the
            minimum radial limit.
        use_rmin : `bool`, optional
            If ``True`` add the minimum radial axis limit after
            transforming from Cartesian coordinates. *axis* must also be
            specified for this to take effect.
        """
        super().__init__()
        self._axis = axis
        self._use_rmin = use_rmin
        self._apply_theta_transforms = _apply_theta_transforms

    __str__ = mtransforms._make_str_method(
        "_axis",
        use_rmin="_use_rmin",
        _apply_theta_transforms="_apply_theta_transforms")

    def transform_non_affine(self, xy):
        # docstring inherited
        x, y = xy.T
        r = np.hypot(x, y)
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        # PolarAxes does not use the theta transforms here, but apply them for
        # backwards-compatibility if not being used by it.
        if self._apply_theta_transforms and self._axis is not None:
            theta -= self._axis.get_theta_offset()
            theta *= self._axis.get_theta_direction()
            theta %= 2 * np.pi
        if self._use_rmin and self._axis is not None:
            r += self._axis.get_rorigin()
            r *= self._axis.get_rsign()
        # Convert theta radians to theta units.
        theta = self._axis.radians2units(theta)
        return np.column_stack([theta, r])

    def inverted(self):
        # docstring inherited
        return PolarAxes.PolarTransform(self._axis, self._use_rmin,
                                        self._apply_theta_transforms)


class ThetaFormatter(mticker.Formatter):
    """
    Used to format the *theta* tick labels.
    """

    def __call__(self, x, pos=None):
        unit = self.axis.axes.theta_unit or ""
        if self.axis.axes.theta_format:
            value = self.axis.axes.theta_format.format(x)
        else:
            vmin, vmax = self.axis.get_view_interval()
            digits = max(-int(np.log10(abs(vmax - vmin)) - 1.5), 0)
            value = f"{x:0.{digits:d}f}"
        return f"{value}{unit}"


class ThetaLocator(mticker.Locator):
    """
    Used to locate theta ticks.

    This will work the same as the base locator except in the case that the
    view spans the entire circle. In such cases, the previously used default
    locations of every 45 degrees are returned.
    """

    def __init__(self, base):
        self.base = base
        self.axis = self.base.axis

    def set_axis(self, axis):
        self.axis = axis
        self.base.set_axis(self.axis)

    def __call__(self):
        axes = self.axis.axes
        lim = self.axis.get_view_interval()
        if _is_full_circle_rad(*axes.units2radians(lim)):
            # in case of non-zero lower bound prefer the upper bound tick
            offset = 0 if lim[0] == 0 else 1
            return axes.theta_max * np.arange(offset, 8 + offset) / 8 + lim[0]
        else:
            return self.base()

    def view_limits(self, vmin, vmax):
        return self.base.view_limits(vmin, vmax)


class ThetaTick(maxis.XTick):
    """
    A theta-axis tick.

    This subclass of `.XTick` provides angular ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    tick location. This results in ticks that are correctly perpendicular to
    the arc spine.

    When 'auto' rotation is enabled, labels are also rotated to be parallel to
    the spine. The label padding is also applied here since it's not possible
    to use a generic axes transform to produce tick-specific padding.
    """

    def __init__(self, axes, *args, **kwargs):
        self._text1_translate = mtransforms.ScaledTranslation(
            0, 0, axes.figure.dpi_scale_trans)
        self._text2_translate = mtransforms.ScaledTranslation(
            0, 0, axes.figure.dpi_scale_trans)
        super().__init__(axes, *args, **kwargs)
        self.label1.set(
            rotation_mode='anchor',
            transform=self.label1.get_transform() + self._text1_translate)
        self.label2.set(
            rotation_mode='anchor',
            transform=self.label2.get_transform() + self._text2_translate)

    def _apply_params(self, **kwargs):
        super()._apply_params(**kwargs)
        # Ensure transform is correct; sometimes this gets reset.
        trans = self.label1.get_transform()
        if not trans.contains_branch(self._text1_translate):
            self.label1.set_transform(trans + self._text1_translate)
        trans = self.label2.get_transform()
        if not trans.contains_branch(self._text2_translate):
            self.label2.set_transform(trans + self._text2_translate)

    def _update_padding(self, pad, angle):
        angle = self.axes.units2radians(angle)
        padx = pad * np.cos(angle) / 72
        pady = pad * np.sin(angle) / 72
        self._text1_translate._t = (padx, pady)
        self._text1_translate.invalidate()
        self._text2_translate._t = (-padx, -pady)
        self._text2_translate.invalidate()

    def update_position(self, loc):
        super().update_position(loc)
        axes = self.axes
        angle = axes.units2radians(
            loc * axes.get_theta_direction() + axes.get_theta_offset())
        text_angle = np.rad2deg(angle) % 360 - 90
        angle -= np.pi / 2

        marker = self.tick1line.get_marker()
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick1line._marker._transform
        self.tick1line._marker._transform = trans

        marker = self.tick2line.get_marker()
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # Don't modify custom tick line markers.
            trans = self.tick2line._marker._transform
        self.tick2line._marker._transform = trans

        mode, user_angle = self._labelrotation
        if mode == 'default':
            text_angle = user_angle
        else:
            if text_angle > 90:
                text_angle -= 180
            elif text_angle < -90:
                text_angle += 180
            text_angle += user_angle
        self.label1.set_rotation(text_angle)
        self.label2.set_rotation(text_angle)

        # This extra padding helps preserve the look from previous releases but
        # is also needed because labels are anchored to their center.
        pad = self._pad + 7
        self._update_padding(
            pad,
            self._loc * axes.get_theta_direction() + axes.get_theta_offset())


class ThetaAxis(maxis.XAxis):
    """
    A theta Axis.

    This overrides certain properties of an `.XAxis` to provide special-casing
    for an angular axis.
    """
    __name__ = 'thetaaxis'
    axis_name = 'theta'  #: Read-only name identifying the axis.
    _tick_class = ThetaTick

    def _get_tick(self, major):
        # required by old API
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return self._tick_class(self.axes, 0, major=major, **tick_kw)

    def _wrap_locator_formatter(self):
        self.set_major_locator(self.axes.ThetaLocator(self.get_major_locator()))
        self.set_major_formatter(self.axes.ThetaFormatter())
        self.isDefault_majloc = True
        self.isDefault_majfmt = True

    def clear(self):
        # docstring inherited
        super().clear()
        self.set_ticks_position('none')
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        if value != 'linear':
            raise NotImplementedError(
                "The xscale cannot be set on a polar plot")
        super()._set_scale(value, **kwargs)
        # LinearScale.set_default_locators_and_formatters just set the major
        # locator to be an AutoLocator, so we customize it here to have ticks
        # at sensible degree multiples.
        self.get_major_locator().set_params(steps=[1, 1.5, 3, 4.5, 9, 10])
        self._wrap_locator_formatter()

    def _copy_tick_props(self, src, dest):
        """Copy the props from src tick to dest tick."""
        if src is None or dest is None:
            return
        super()._copy_tick_props(src, dest)

        # Ensure that tick transforms are independent so that padding works.
        trans = dest._get_text1_transform()[0]
        dest.label1.set_transform(trans + dest._text1_translate)
        trans = dest._get_text2_transform()[0]
        dest.label2.set_transform(trans + dest._text2_translate)


class RadialFormatter(mticker.ScalarFormatter):
    """
    Used to format the radial tick labels.
    """

    def __call__(self, x, pos=None):
        unit = self.axis.axes.r_unit or ""
        if self.axis.axes.r_format:
            value = self.axis.axes.r_format.format(x)
        else:
            value = super().__call__(x, pos)
        return f"{value}{unit}"


class RadialLocator(mticker.Locator):
    """
    Used to locate radius ticks.

    Ensures that all ticks are strictly positive.  For all other tasks, it
    delegates to the base `.Locator` (which may be different depending on the
    scale of the *r*-axis).
    """

    def __init__(self, base, axes=None):
        self.base = base
        self._axes = axes

    def set_axis(self, axis):
        self.base.set_axis(axis)

    def __call__(self):
        # Ensure previous behaviour with full circle non-annular views.
        if self._axes:
            bounds = self._axes.units2radians(self._axes.viewLim.intervalx)
            if _is_full_circle_rad(*bounds):
                rorigin = self._axes.get_rorigin() * self._axes.get_rsign()
                if self._axes.get_rmin() <= rorigin:
                    return [tick for tick in self.base() if tick > rorigin]
        return self.base()

    def _zero_in_bounds(self):
        """
        Return True if zero is within the valid values for the
        scale of the radial axis.
        """
        vmin, vmax = self._axes.yaxis._scale.limit_range_for_scale(0, 1, 1e-5)
        return vmin == 0

    def nonsingular(self, vmin, vmax):
        # docstring inherited
        if self._zero_in_bounds() and (vmin, vmax) == (-np.inf, np.inf):
            # Initial view limits
            return (0, 1)
        else:
            return self.base.nonsingular(vmin, vmax)

    def view_limits(self, vmin, vmax):
        vmin, vmax = self.base.view_limits(vmin, vmax)
        if self._zero_in_bounds() and vmax > vmin:
            # this allows inverted r/y-lims
            vmin = min(0, vmin)
        return mtransforms.nonsingular(vmin, vmax)


class _ThetaShift(mtransforms.ScaledTranslation):
    """
    Apply a padding shift based on axes theta limits.

    This is used to create padding for radial ticks.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        The owning axes; used to determine limits.
    pad : float
        The padding to apply, in points.
    mode : {'min', 'max', 'rlabel'}
        Whether to shift away from the start (``'min'``) or the end (``'max'``)
        of the axes, or using the rlabel position (``'rlabel'``).
    """
    def __init__(self, axes, pad, mode):
        super().__init__(pad, pad, axes.figure.dpi_scale_trans)
        self.set_children(axes._realViewLim)
        self.axes = axes
        self.mode = mode
        self.pad = pad

    __str__ = mtransforms._make_str_method("axes", "pad", "mode")

    def get_matrix(self):
        if self._invalid:
            if self.mode == 'rlabel':
                angle = (
                    np.deg2rad(self.axes.get_rlabel_position()) *
                    self.axes.get_theta_direction() +
                    self.axes.units2radians(self.axes.get_theta_offset())
                )
            else:
                if self.mode == 'min':
                    angle = self.axes.units2radians(self.axes._realViewLim.xmin)
                elif self.mode == 'max':
                    angle = self.axes.units2radians(self.axes._realViewLim.xmax)

            if self.mode in ('rlabel', 'min'):
                padx = np.cos(angle - np.pi / 2)
                pady = np.sin(angle - np.pi / 2)
            else:
                padx = np.cos(angle + np.pi / 2)
                pady = np.sin(angle + np.pi / 2)

            self._t = (self.pad * padx / 72, self.pad * pady / 72)
        return super().get_matrix()


class RadialTick(maxis.YTick):
    """
    A radial-axis tick.

    This subclass of `.YTick` provides radial ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    axes limits.  This results in ticks that are correctly perpendicular to
    the spine. Labels are also rotated to be perpendicular to the spine, when
    'auto' rotation is enabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label1.set_rotation_mode('anchor')
        self.label2.set_rotation_mode('anchor')

    def _determine_anchor(self, mode, angle, start):
        # Note: angle is the (spine angle - 90) because it's used for the tick
        # & text setup, so all numbers below are -90 from (normed) spine angle.
        if mode == 'auto':
            if start:
                if -90 <= angle <= 90:
                    return 'left', 'center'
                else:
                    return 'right', 'center'
            else:
                if -90 <= angle <= 90:
                    return 'right', 'center'
                else:
                    return 'left', 'center'
        else:
            if start:
                if angle < -68.5:
                    return 'center', 'top'
                elif angle < -23.5:
                    return 'left', 'top'
                elif angle < 22.5:
                    return 'left', 'center'
                elif angle < 67.5:
                    return 'left', 'bottom'
                elif angle < 112.5:
                    return 'center', 'bottom'
                elif angle < 157.5:
                    return 'right', 'bottom'
                elif angle < 202.5:
                    return 'right', 'center'
                elif angle < 247.5:
                    return 'right', 'top'
                else:
                    return 'center', 'top'
            else:
                if angle < -68.5:
                    return 'center', 'bottom'
                elif angle < -23.5:
                    return 'right', 'bottom'
                elif angle < 22.5:
                    return 'right', 'center'
                elif angle < 67.5:
                    return 'right', 'top'
                elif angle < 112.5:
                    return 'center', 'top'
                elif angle < 157.5:
                    return 'left', 'top'
                elif angle < 202.5:
                    return 'left', 'center'
                elif angle < 247.5:
                    return 'left', 'bottom'
                else:
                    return 'center', 'bottom'

    def update_position(self, loc):
        super().update_position(loc)
        axes = self.axes
        units2degrees = lambda v: np.rad2deg(axes.units2radians(v))
        thetamin = units2degrees(axes.get_thetamin())
        thetamax = units2degrees(axes.get_thetamax())
        direction = axes.get_theta_direction()
        offset = units2degrees(axes.get_theta_offset())
        rlabel_position = axes.get_rlabel_position()
        full = _is_full_circle_deg(thetamin, thetamax)

        if full:
            angle = (rlabel_position * direction + offset) % 360 - 90
            tick_angle = 0
        else:
            angle = (thetamin * direction + offset) % 360 - 90
            if direction > 0:
                tick_angle = angle
            else:
                tick_angle = angle + 180
        text_angle = (angle + 90) % 180 - 90  # between -90 and +90.
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle

        if full:
            ha = self.label1.get_horizontalalignment()
            va = self.label1.get_verticalalignment()
        else:
            ha, va = self._determine_anchor(mode, angle, direction > 0)
        self.label1.set_horizontalalignment(ha)
        self.label1.set_verticalalignment(va)
        self.label1.set_rotation(text_angle)

        marker = self.tick1line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(np.deg2rad(tick_angle))
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(np.deg2rad(tick_angle + 90))
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1)\
                    .rotate(np.deg2rad(tick_angle))
        else:
            # Don't modify custom tick line markers.
            trans = self.tick1line._marker._transform
        self.tick1line._marker._transform = trans

        if full:
            self.label2.set_visible(False)
            self.tick2line.set_visible(False)
        angle = (thetamax * direction + offset) % 360 - 90
        if direction > 0:
            tick_angle = angle
        else:
            tick_angle = angle + 180
        text_angle = (angle + 90) % 180 - 90  # between -90 and +90.
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle

        ha, va = self._determine_anchor(mode, angle, direction < 0)
        self.label2.set_ha(ha)
        self.label2.set_va(va)
        self.label2.set_rotation(text_angle)

        marker = self.tick2line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(np.deg2rad(tick_angle))
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(np.deg2rad(tick_angle + 90))
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1)\
                    .rotate(np.deg2rad(tick_angle))
        else:
            # Don't modify custom tick line markers.
            trans = self.tick2line._marker._transform
        self.tick2line._marker._transform = trans


class RadialAxis(maxis.YAxis):
    """
    A radial Axis.

    This overrides certain properties of a `.YAxis` to provide special-casing
    for a radial axis.
    """
    __name__ = 'radialaxis'
    axis_name = 'radius'  #: Read-only name identifying the axis.
    _tick_class = RadialTick

    def _get_tick(self, major):
        # required by old API
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return self._tick_class(self.axes, 0, major=major, **tick_kw)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sticky_edges.y.append(0)

    def _wrap_locator_formatter(self):
        self.set_major_locator(
            self.axes.RadialLocator(self.get_major_locator(), self.axes))
        self.set_major_formatter(self.axes.RadialFormatter())
        self.isDefault_majloc = True
        self.isDefault_majfmt = True

    def clear(self):
        # docstring inherited
        super().clear()
        self.set_ticks_position('none')
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        super()._set_scale(value, **kwargs)
        self._wrap_locator_formatter()


def _is_full_circle_deg(thetamin, thetamax):
    """
    Determine if a wedge (in degrees) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(abs(thetamax - thetamin) - 360.0) < 1e-12


def _is_full_circle_rad(thetamin, thetamax):
    """
    Determine if a wedge (in radians) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(abs(thetamax - thetamin) - 2 * np.pi) < 1.74e-14


def _exceeds_full_circle_rad(thetamin, thetamax):
    """
    Determine if a wedge (in radians) exceeds the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    return abs(thetamax - thetamin) - 2 * np.pi >= 1.74e-14


class _WedgeBbox(mtransforms.Bbox):
    """
    Transform (theta, r) wedge Bbox into axes bounding box.

    Parameters
    ----------
    center : (float, float)
        Center of the wedge
    viewLim : `~matplotlib.transforms.Bbox`
        Bbox determining the boundaries of the wedge
    originLim : `~matplotlib.transforms.Bbox`
        Bbox determining the origin for the wedge, if different from *viewLim*
    """
    def __init__(self, axes, center, viewLim, originLim, **kwargs):
        super().__init__([[0, 0], [1, 1]], **kwargs)
        self._axes = axes
        self._center = center
        self._viewLim = viewLim
        self._originLim = originLim
        self.set_children(viewLim, originLim)

    __str__ = mtransforms._make_str_method("_center", "_viewLim", "_originLim")

    def get_points(self):
        # docstring inherited
        if self._invalid:
            points = self._viewLim.get_points().copy()
            # convert angular limits to degrees to work with Wedge.
            points[:, 0] = np.rad2deg(self._axes.units2radians(points[:, 0]))
            if points[0, 0] > points[1, 0]:
                points[:, 0] = points[::-1, 0]

            # Scale radial limits based on origin radius.
            points[:, 1] -= self._originLim.y0

            # Scale radial limits to match axes limits.
            rscale = 0.5 / points[1, 1]
            points[:, 1] *= rscale
            width = min(points[1, 1] - points[0, 1], 0.5)

            # Generate bounding box for wedge.
            wedge = mpatches.Wedge(self._center, points[1, 1],
                                   points[0, 0], points[1, 0],
                                   width=width)
            self.update_from_path(wedge.get_path())

            # Ensure equal aspect ratio.
            w, h = self._points[1] - self._points[0]
            deltah = max(w - h, 0) / 2
            deltaw = max(h - w, 0) / 2
            self._points += np.array([[-deltaw, -deltah], [deltaw, deltah]])

            self._invalid = 0

        return self._points


class PolarSpine(Spine):
    """
    Custom spine object handling gracefully the theta angular units.
    """

    def _adjust_location(self):
        if self._patch_type != 'arc':
            super()._adjust_location()
            return

        # fix arc spine angles
        thetamin, thetamax = np.rad2deg(self.axes.units2radians(
            self.axes._realViewLim.intervalx))
        if thetamin > thetamax:
            thetamin, thetamax = thetamax, thetamin
        self._path = Path.arc(thetamin, thetamax)

        # scale inner arc spine
        if self.spine_type == 'bottom':
            rorigin = self.axes.get_rorigin()
            rmin, rmax = self.axes._realViewLim.intervaly
            scale = (rmin - rorigin) / (rmax - rorigin)
            self._height = scale
            self._width = scale


class PolarAxes(Axes):
    """
    A polar graph projection, where the input dimensions are *theta*, *r*.

    Theta starts pointing east and goes anti-clockwise.
    """
    name = 'polar_plus'

    # Maximum (full circle) theta value.
    theta_max = 360 # degrees

    # Use Unicode rather than mathtext with \circ, so that it will work
    # correctly with any arbitrary font (assuming it has a degree sign),
    # whereas $5\circ$ will only work correctly with one of the supported
    # math fonts (Computer Modern and STIX).
    theta_unit = "\N{DEGREE SIGN}"
    theta_label = "\N{GREEK SMALL LETTER THETA}"
    theta_format = None
    r_unit = ""
    r_label = "r"
    r_format = None
    r_min = 0.0
    r_max = 1.0
    r_origin = None

    coord_format = "{theta_label}={theta}{theta_unit}, {r_label}={r}{r_unit}"

    def units2radians(self, value):
        """ Convert theta angular units to radians. """
        return value * 2.0 * np.pi / self.theta_max

    def radians2units(self, value):
        """ Convert radians to theta angular units. """
        return value * self.theta_max * 0.5 / np.pi

    def __init__(self, *args,
                 theta_offset=0, theta_direction=1, rlabel_position=22.5,
                 **kwargs):
        # docstring inherited
        self._default_theta_offset = theta_offset
        self._default_theta_direction = theta_direction
        self._default_rlabel_position = rlabel_position
        super().__init__(*args, **kwargs)
        self.use_sticky_edges = True
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.clear()

    def clear(self):
        # docstring inherited
        super().clear()

        self.title.set_y(1.05)

        start = self.spines.get('start', None)
        if start:
            start.set_visible(False)
        end = self.spines.get('end', None)
        if end:
            end.set_visible(False)

        self.set_xlim(0.0, self.theta_max)
        self.set_ylim(self.r_min, self.r_max)

        self.grid(mpl.rcParams['polaraxes.grid'])
        inner = self.spines.get('inner', None)
        if inner:
            inner.set_visible(False)

        self.set_rorigin(self.r_origin)
        self.set_theta_offset(self._default_theta_offset)
        self.set_theta_direction(self._default_theta_direction)

    def _init_axis(self):
        # This is moved out of __init__ because non-separable axes don't use it
        self.xaxis = self.ThetaAxis(self)
        self.yaxis = self.RadialAxis(self)
        # Calling polar_axes.xaxis.clear() or polar_axes.yaxis.clear()
        # results in weird artifacts. Therefore we disable this for now.
        # self.spines['polar'].register_axis(self.yaxis)

    def _set_lim_and_transforms(self):
        # A view limit where the minimum radius can be locked if the user
        # specifies an alternate origin.
        self._originViewLim = mtransforms.LockableBbox(self.viewLim)

        # Handle angular offset and direction.
        self._direction = mtransforms.Affine2D() \
            .scale(self._default_theta_direction, 1.0)
        self._theta_offset = mtransforms.Affine2D() \
            .translate(self._default_theta_offset, 0.0)
        self.transShift = self._direction + self._theta_offset
        # A view limit shifted to the correct location after accounting for
        # orientation and offset.
        self._realViewLim = mtransforms.TransformedBbox(self.viewLim,
                                                        self.transShift)

        # Transforms the x and y axis separately by a scale factor
        # It is assumed that this part will have non-linear components
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # Scale view limit into a bbox around the selected wedge. This may be
        # smaller than the usual unit axes rectangle if not plotting the full
        # circle.
        self.axesLim = _WedgeBbox(
            self, (0.5, 0.5), self._realViewLim, self._originViewLim)

        # Scale the wedge to fill the axes.
        self.transWedge = mtransforms.BboxTransformFrom(self.axesLim)

        # Scale the axes to fill the figure.
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # A (possibly non-linear) projection on the (already scaled)
        # data.  This one is aware of rmin
        self.transProjection = self.PolarTransform(
            self,
            _apply_theta_transforms=False,
            scale_transform=self.transScale
        )
        # Add dependency on rorigin.
        self.transProjection.set_children(self._originViewLim)

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transProjectionAffine = self.PolarAffine(self.transScale,
                                                      self._originViewLim)

        # The complete data transformation stack -- from data all the
        # way to display coordinates
        #
        # 1. Remove any radial axis scaling (e.g. log scaling)
        # 2. Shift data in the theta direction
        # 3. Project the data from polar to cartesian values
        #    (with the origin in the same place)
        # 4. Scale and translate the cartesian values to Axes coordinates
        #    (here the origin is moved to the lower left of the Axes)
        # 5. Move and scale to fill the Axes
        # 6. Convert from Axes coordinates to Figure coordinates
        self.transData = (
            self.transScale +
            self.transShift +
            self.transProjection +
            (
                self.transProjectionAffine +
                self.transWedge +
                self.transAxes
            )
        )

        # This is the transform for theta-axis ticks.  It is
        # equivalent to transData, except it always puts r == 0.0 and r == 1.0
        # at the edge of the axis circles.
        self._xaxis_transform = (
            mtransforms.blended_transform_factory(
                mtransforms.IdentityTransform(),
                mtransforms.BboxTransformTo(self.viewLim)) +
            self.transData)
        # The theta labels are flipped along the radius, so that text 1 is on
        # the outside by default. This should work the same as before.
        flipr_transform = mtransforms.Affine2D() \
            .translate(0.0, -0.5) \
            .scale(1.0, -1.0) \
            .translate(0.0, 0.5)
        self._xaxis_text_transform = flipr_transform + self._xaxis_transform

        # This is the transform for r-axis ticks.  It scales the theta
        # axis so the gridlines from 0.0 to 1.0, now go from thetamin to
        # thetamax.
        self._yaxis_transform = (
            mtransforms.blended_transform_factory(
                mtransforms.BboxTransformTo(self.viewLim),
                mtransforms.IdentityTransform()) +
            self.transData)
        # The r-axis labels are put at an angle and padded in the r-direction
        self._r_label_position = mtransforms.Affine2D()
        self.set_rlabel_position(self._default_rlabel_position)
        self._yaxis_text_transform = mtransforms.TransformWrapper(
            self.transData)

    def get_xaxis_transform(self, which='grid'):
        _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text_transform, 'center', 'center'

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text_transform, 'center', 'center'

    def get_yaxis_transform(self, which='grid'):
        if which in ('tick1', 'tick2'):
            return self._yaxis_text_transform
        elif which == 'grid':
            return self._yaxis_transform
        else:
            _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)

    def get_yaxis_text1_transform(self, pad):
        thetamin, thetamax = (self.units2radians(self._realViewLim.intervalx))
        if _is_full_circle_rad(thetamin, thetamax):
            return self._yaxis_text_transform, 'bottom', 'left'
        elif self.get_theta_direction() > 0:
            halign = 'left'
            pad_shift = _ThetaShift(self, pad, 'min')
        else:
            halign = 'right'
            pad_shift = _ThetaShift(self, pad, 'max')
        return self._yaxis_text_transform + pad_shift, 'center', halign

    def get_yaxis_text2_transform(self, pad):
        if self.get_theta_direction() > 0:
            halign = 'right'
            pad_shift = _ThetaShift(self, pad, 'max')
        else:
            halign = 'left'
            pad_shift = _ThetaShift(self, pad, 'min')
        return self._yaxis_text_transform + pad_shift, 'center', halign

    def draw(self, renderer):
        self._unstale_viewLim()
        thetamin, thetamax = (
            np.rad2deg(self.units2radians(self._realViewLim.intervalx)))
        if thetamin > thetamax:
            thetamin, thetamax = thetamax, thetamin
        rmin, rmax = ((self._realViewLim.intervaly - self.get_rorigin()) *
                      self.get_rsign())
        if isinstance(self.patch, mpatches.Wedge):
            # Backwards-compatibility: Any subclassed Axes might override the
            # patch to not be the Wedge that PolarAxes uses.
            center = self.transWedge.transform((0.5, 0.5))
            self.patch.set_center(center)
            self.patch.set_theta1(thetamin)
            self.patch.set_theta2(thetamax)

            edge, _ = self.transWedge.transform((1, 0))
            radius = edge - center[0]
            width = min(radius * (rmax - rmin) / rmax, radius)
            self.patch.set_radius(radius)
            self.patch.set_width(width)

            inner_width = radius - width
            inner = self.spines.get('inner', None)
            if inner:
                inner.set_visible(inner_width != 0.0)

        visible = not _is_full_circle_deg(thetamin, thetamax)
        # For backwards compatibility, any subclassed Axes might override the
        # spines to not include start/end that PolarAxes uses.
        start = self.spines.get('start', None)
        end = self.spines.get('end', None)
        if start:
            start.set_visible(visible)
        if end:
            end.set_visible(visible)
        if visible:
            yaxis_text_transform = self._yaxis_transform
        else:
            yaxis_text_transform = self._r_label_position + self.transData
        if self._yaxis_text_transform != yaxis_text_transform:
            self._yaxis_text_transform.set(yaxis_text_transform)
            self.yaxis.reset_ticks()
            self.yaxis.set_clip_path(self.patch)

        super().draw(renderer)

    def _gen_axes_patch(self):
        return mpatches.Wedge((0.5, 0.5), 0.5, 0.0, 360.0)

    def _gen_axes_spines(self):
        spines = {
            'polar': PolarSpine.arc_spine(self, 'top', (0.5, 0.5), 0.5, 0, 360),
            'start': Spine.linear_spine(self, 'left'),
            'end': Spine.linear_spine(self, 'right'),
            'inner': PolarSpine.arc_spine(self, 'bottom', (0.5, 0.5), 0.0, 0, 360),
        }
        spines['polar'].set_transform(self.transWedge + self.transAxes)
        spines['inner'].set_transform(self.transWedge + self.transAxes)
        spines['start'].set_transform(self._yaxis_transform)
        spines['end'].set_transform(self._yaxis_transform)
        return spines

    def set_thetamax(self, thetamax):
        """Set the maximum theta limit in the theta angular units."""
        self.viewLim.x1 = thetamax

    def get_thetamax(self):
        """Return the maximum theta limit in the theta angular units."""
        return self.viewLim.xmax

    def set_thetamin(self, thetamin):
        """Set the minimum theta limit in the theta angular units."""
        self.viewLim.x0 = thetamin

    def get_thetamin(self):
        """Get the minimum theta limit in the theta angular units."""
        return self.viewLim.xmin

    def set_thetalim(self, *args, **kwargs):
        r"""
        Set the minimum and maximum theta values.

        Can take the following signatures:

        - ``set_thetalim(minval, maxval)``
        - ``set_thetalim(thetamin=minval, thetamax=maxval)``

        Both setting the limits in the theta angular units.

        where minval and maxval are the minimum and maximum limits. Values are
        wrapped in to the range :math:`[0, theta_max]`, so for example it is
        possible to do ``set_thetalim(-theta_max / 2,  theta_max/ 2)`` to have
        an axis symmetric around 0. A ValueError is raised if the absolute
        angle difference is larger than a full circle.
        """
        orig_lim = self.get_xlim()
        if 'thetamin' in kwargs:
            kwargs['xmin'] = kwargs.pop('thetamin')
        if 'thetamax' in kwargs:
            kwargs['xmax'] = kwargs.pop('thetamax')
        new_min, new_max = self.set_xlim(*args, **kwargs)
        # Parsing all permutations of *args, **kwargs is tricky; it is simpler
        # to let set_xlim() do it and then validate the limits.
        if _exceeds_full_circle_rad(self.units2radians(new_max), self.units2radians(new_min)):
            self.set_xlim(orig_lim)  # un-accept the change
            raise ValueError("The angle range must be less than a full circle")
        return new_min, new_max

    def set_theta_offset(self, offset):
        """
        Set the offset for the location of 0 in the theta angular units.
        """
        mtx = self._theta_offset.get_matrix()
        mtx[0, 2] = offset
        self._theta_offset.invalidate()

    def get_theta_offset(self):
        """
        Get the offset for the location of 0 in the theta angular units.
        """
        return self._theta_offset.get_matrix()[0, 2]

    def set_theta_zero_location(self, loc, offset=0.0):
        """
        Set the location of theta's zero.

        This simply calls `set_theta_offset` with the correct value in radians.

        Parameters
        ----------
        loc : str
            May be one of "N", "NW", "W", "SW", "S", "SE", "E", or "NE".
        offset : float, default: 0
            An offset in degrees to apply from the specified *loc*. **Note:**
            this offset is *always* applied counter-clockwise regardless of
            the direction setting.
        """
        mapping = {
            'E': 0,
            'NE': 0.125,
            'N': 0.25,
            'NW': 0.375,
            'W': 0.5,
            'SW': 0.625,
            'S': 0.75,
            'SE': 0.875}
        return self.set_theta_offset(
            self.theta_max * (mapping[loc] + offset/360.0))

    def set_theta_direction(self, direction):
        """
        Set the direction in which theta increases.

        clockwise, -1:
           Theta increases in the clockwise direction

        counterclockwise, anticlockwise, 1:
           Theta increases in the counterclockwise direction
        """
        mtx = self._direction.get_matrix()
        if direction in ('clockwise', -1):
            mtx[0, 0] = -1
        elif direction in ('counterclockwise', 'anticlockwise', 1):
            mtx[0, 0] = 1
        else:
            _api.check_in_list(
                [-1, 1, 'clockwise', 'counterclockwise', 'anticlockwise'],
                direction=direction)
        self._direction.invalidate()

    def get_theta_direction(self):
        """
        Get the direction in which theta increases.

        -1:
           Theta increases in the clockwise direction

        1:
           Theta increases in the counterclockwise direction
        """
        return self._direction.get_matrix()[0, 0]

    def set_rmax(self, rmax):
        """
        Set the outer radial limit.

        Parameters
        ----------
        rmax : float
        """
        self.viewLim.y1 = rmax

    def get_rmax(self):
        """
        Returns
        -------
        float
            Outer radial limit.
        """
        return self.viewLim.ymax

    def set_rmin(self, rmin):
        """
        Set the inner radial limit.

        Parameters
        ----------
        rmin : float
        """
        self.viewLim.y0 = rmin

    def get_rmin(self):
        """
        Returns
        -------
        float
            The inner radial limit.
        """
        return self.viewLim.ymin

    def set_rorigin(self, rorigin):
        """
        Update the radial origin.

        Parameters
        ----------
        rorigin : float
        """
        self._originViewLim.locked_y0 = rorigin

    def get_rorigin(self):
        """
        Returns
        -------
        float
        """
        return self._originViewLim.y0

    def get_rsign(self):
        return np.sign(self._originViewLim.y1 - self._originViewLim.y0)

    @_api.make_keyword_only("3.6", "emit")
    def set_rlim(self, bottom=None, top=None, emit=True, auto=False, **kwargs):
        """
        Set the radial axis view limits.

        This function behaves like `.Axes.set_ylim`, but additionally supports
        *rmin* and *rmax* as aliases for *bottom* and *top*.

        See Also
        --------
        .Axes.set_ylim
        """
        if 'rmin' in kwargs:
            if bottom is None:
                bottom = kwargs.pop('rmin')
            else:
                raise ValueError('Cannot supply both positional "bottom"'
                                 'argument and kwarg "rmin"')
        if 'rmax' in kwargs:
            if top is None:
                top = kwargs.pop('rmax')
            else:
                raise ValueError('Cannot supply both positional "top"'
                                 'argument and kwarg "rmax"')
        return self.set_ylim(bottom=bottom, top=top, emit=emit, auto=auto,
                             **kwargs)

    def get_rlabel_position(self):
        """
        Get the position of the radius labels.

        Returns
        -------
        float
            The angular position of the full-circle radius labels
            wrt. theta origin in degrees.
        """
        return np.rad2deg(self.units2radians(
            self._r_label_position.get_matrix()[0, 2]))

    def set_rlabel_position(self, value):
        """
        Update the theta position of the radius labels.

        Parameters
        ----------
        value : number
            The angular position of the full-circle radius labels
            wrt. theta origin in degrees.
        """
        self._r_label_position.clear().translate(
                self.radians2units(np.deg2rad(value)), 0.0)

    def set_yscale(self, *args, **kwargs):
        super().set_yscale(*args, **kwargs)
        self.yaxis.set_major_locator(
            self.RadialLocator(self.yaxis.get_major_locator(), self))

    def set_rscale(self, *args, **kwargs):
        return Axes.set_yscale(self, *args, **kwargs)

    def set_rticks(self, *args, **kwargs):
        return Axes.set_yticks(self, *args, **kwargs)

    def set_thetagrids(self, angles, labels=None, fmt=None, **kwargs):
        """
        Set the theta gridlines in a polar plot.

        Parameters
        ----------
        angles : tuple with floats, theta units
            The angles of the theta gridlines.

        labels : tuple with strings or None
            The labels to use at each theta gridline. The
            `.projections.polar.ThetaFormatter` will be used if None.

        fmt : str or None
            Format string used in `matplotlib.ticker.FormatStrFormatter`.
            For example '%f'. Note that the angle that is used is in
            radians.

        Returns
        -------
        lines : list of `.lines.Line2D`
            The theta gridlines.

        labels : list of `.text.Text`
            The tick labels.

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `.Text` properties for the labels.

        See Also
        --------
        .PolarAxes.set_rgrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """

        # Make sure we take into account unitized data
        angles = self.convert_yunits(angles) # FIXME: is it correct?
        self.set_xticks(angles)
        if labels is not None:
            self.set_xticklabels(labels)
        elif fmt is not None:
            self.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        for t in self.xaxis.get_ticklabels():
            if hasattr(t, "_internal_update"):
                t._internal_update(kwargs)
            else:
                t.update(kwargs)
        return self.xaxis.get_ticklines(), self.xaxis.get_ticklabels()

    def set_rgrids(self, radii, labels=None, angle=None, fmt=None, **kwargs):
        """
        Set the radial gridlines on a polar plot.

        Parameters
        ----------
        radii : tuple with floats
            The radii for the radial gridlines

        labels : tuple with strings or None
            The labels to use at each radial gridline. The
            `matplotlib.ticker.ScalarFormatter` will be used if None.

        angle : float
            The angular position of the radius labels in degrees.

        fmt : str or None
            Format string used in `matplotlib.ticker.FormatStrFormatter`.
            For example '%f'.

        Returns
        -------
        lines : list of `.lines.Line2D`
            The radial gridlines.

        labels : list of `.text.Text`
            The tick labels.

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `.Text` properties for the labels.

        See Also
        --------
        .PolarAxes.set_thetagrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """
        # Make sure we take into account unitized data
        radii = self.convert_xunits(radii) # FIXME: is this correct?
        radii = np.asarray(radii)

        self.set_yticks(radii)
        if labels is not None:
            self.set_yticklabels(labels)
        elif fmt is not None:
            self.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        if angle is None:
            angle = self.get_rlabel_position()
        self.set_rlabel_position(angle)
        for t in self.yaxis.get_ticklabels():
            if hasattr(t, "_internal_update"):
                t._internal_update(kwargs)
            else:
                t.update(kwargs)
        return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()

    def format_coord(self, theta, r):
        # docstring inherited
        screen_xy = self.transData.transform((theta, r))
        screen_xys = screen_xy + np.stack(
            np.meshgrid([-1, 0, 1], [-1, 0, 1])).reshape((2, -1)).T
        ts, rs = self.transData.inverted().transform(screen_xys).T
        delta_t = abs(
            (ts - theta + self.theta_max*0.5) % self.theta_max
            - self.theta_max*0.5
        ).max()
        delta_t_halfturns = delta_t / np.pi
        delta_r = abs(rs - r).max()
        if theta < 0:
            theta += self.theta_max
        theta_halfturns = theta * 2 / self.theta_max

        # See ScalarFormatter.format_data_short.  For r, use g-formatting
        # (as for linear axes), but for theta, use f-formatting as scientific
        # notation doesn't make sense and the trailing dot is ugly.
        def format_sig(value, delta, opt, fmt):
            # For "f", only count digits after decimal point.
            prec = (max(0, -math.floor(math.log10(delta))) if fmt == "f" else
                    cbook._g_sig_digits(value, delta))
            return f"{value:-{opt}.{prec}{fmt}}"

        return self.coord_format.format(
            theta_label=self.theta_label,
            theta_unit=self.theta_unit,
            theta=format_sig(theta, delta_t_halfturns, "", "f"),
            r_label=self.r_label,
            r_unit=self.r_unit,
            r=format_sig(r, delta_t_halfturns, "", "g"),
        )

    def get_data_ratio(self):
        """
        Return the aspect ratio of the data itself.  For a polar plot,
        this should always be 1.0
        """
        return 1.0

    # # # Interactive panning

    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.

        A polar Axes does not support zoom boxes.
        """
        return False

    def can_pan(self):
        """
        Return whether this Axes supports the pan/zoom button functionality.

        For a polar Axes, this is slightly misleading. Both panning and
        zooming are performed by the same button. Panning is performed
        in azimuth while zooming is done along the radial.
        """
        return True

    def start_pan(self, x, y, button):
        angle = np.deg2rad(self.get_rlabel_position())
        mode = ''
        if button == 1:
            epsilon = np.pi / 45.0
            t, r = self.transData.inverted().transform((x, y))
            if angle - epsilon <= t <= angle + epsilon:
                mode = 'drag_r_labels'
        elif button == 3:
            mode = 'zoom'

        self._pan_start = types.SimpleNamespace(
            rmax=self.get_rmax(),
            trans=self.transData.frozen(),
            trans_inverse=self.transData.inverted().frozen(),
            r_label_angle=self.get_rlabel_position(),
            x=x,
            y=y,
            mode=mode)

    def end_pan(self):
        del self._pan_start

    def drag_pan(self, button, key, x, y):
        p = self._pan_start

        if p.mode == 'drag_r_labels':
            (startt, startr), (t, r) = p.trans_inverse.transform(
                [(p.x, p.y), (x, y)])

            # Deal with theta
            dt = np.rad2deg(startt - t)
            self.set_rlabel_position(p.r_label_angle - dt)

            trans, vert1, horiz1 = self.get_yaxis_text1_transform(0.0)
            trans, vert2, horiz2 = self.get_yaxis_text2_transform(0.0)
            for t in self.yaxis.majorTicks + self.yaxis.minorTicks:
                t.label1.set_va(vert1)
                t.label1.set_ha(horiz1)
                t.label2.set_va(vert2)
                t.label2.set_ha(horiz2)

        elif p.mode == 'zoom':
            (startt, startr), (t, r) = p.trans_inverse.transform(
                [(p.x, p.y), (x, y)])

            # Deal with r
            scale = r / startr
            self.set_rmax(p.rmax / scale)


# To keep things all self-contained, we can put aliases to the Polar classes
# defined above. This isn't strictly necessary, but it makes some of the
# code more readable, and provides a backwards compatible Polar API. In
# particular, this is used by the :doc:`/gallery/specialty_plots/radar_chart`
# example to override PolarTransform on a PolarAxes subclass, so make sure that
# that example is unaffected before changing this.
PolarAxes.PolarTransform = PolarTransform
PolarAxes.PolarAffine = PolarAffine
PolarAxes.InvertedPolarTransform = InvertedPolarTransform
PolarAxes.RadialFormatter = RadialFormatter
PolarAxes.ThetaFormatter = ThetaFormatter
PolarAxes.RadialLocator = RadialLocator
PolarAxes.ThetaLocator = ThetaLocator
PolarAxes.ThetaAxis = ThetaAxis
PolarAxes.RadialAxis = RadialAxis
