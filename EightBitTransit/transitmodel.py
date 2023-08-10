import numpy as np
from .TransitingImage import TransitingImage
from .cGridFunctions import pixelate_image, lowres_grid
from .mats import CIRCLEMAT
from numba import jit  # makes things faster
from warnings import warn
__all__ = ["scale_transit", "signal_fit", "TransitModel", "CircleTransit"]


@jit(nopython=True)
def scale_transit(model, trange, depth):
    """Interpolate the reference signal onto our desired properties
    The time range determines width and the depth scales the depth.

    A depth of 0.4 will result in a transit with a minimum flux of 1-0.4

    The result of this function will return an array of flux values for a dip
    placed at a given trange.
    """
    # Generating a reference signal
    model_invert = 1-model  # invert to scale depth of transit
    model_invert = model_invert*(1/max(model_invert))  # scale 0 to 1 for ref

    tmin = min(trange)
    tmax = max(trange)
    signal_times = np.linspace(tmin, tmax, len(model_invert))

    xs = signal_times
    signal_invert = model_invert*depth
    ys = 1-signal_invert
    sig_flux = np.interp(trange, xs, ys)
    return sig_flux


@jit(nopython=True)
def signal_fit(model, times, width, depth, tref):
    # Make an evenly sampled grid to put our signal on with appropriate width
    eventimes = np.linspace(np.min(times), np.max(times), len(times)*5)
    t0 = tref-width/2
    i0 = np.arange(len(eventimes))[t0 < eventimes][0]
    dt = (max(eventimes)-min(eventimes))/len(eventimes)
    ntsteps = int(np.floor(width/dt))

    # put the signal on the evenly spaced time grid
    signal = np.ones_like(eventimes)
    signal[i0:i0 +
           ntsteps] = scale_transit(model, eventimes[i0:i0+ntsteps], depth)

    xs = eventimes
    ys = signal

    # interopolate that  to the actual LC times
    signal = np.interp(times, xs, ys)

    return signal


# initialize above functions
signal_fit(np.array([1., 1., 1., 0.5, 0., 0.5, 1., 1., 1.]),
           np.linspace(0., 1., 8),
           0.3,
           0.3,
           0.5)


class TransitModel(TransitingImage):
    def __init__(self,
                 lowres=None,
                 lowrestype="mean",  # or: "mode",
                 lowresround=False,  # or: True,
                 LDlaw="uniform",  # or: "linear","quadratic","nonlinear",
                 LDCs=None,
                 positions=None,
                 areas=None,
                 blockedflux=None,
                 LD=None,
                 **kwargs):
        # check for required kwargs
        if not (("imfile" in kwargs) or ("opacitymat" in kwargs)):
            raise Exception("""
    Must initialize TransitingImage object with either imfile or
    opacitymat
                            """)

        if "imfile" in kwargs:
            opacitymat = pixelate_image(
                imfile=kwargs["imfile"],
                nside=lowres,
                method=lowrestype,
                rounding=lowresround
            )
        elif (("opacitymat" in kwargs) and (lowres is not None)):
            opacitymat = lowres_grid(
                opacitymat=kwargs["opacitymat"],
                positions=positions,
                nside=lowres,
                method=lowrestype,
                rounding=lowresround
            )
        else:
            opacitymat = kwargs["opacitymat"]
        # init such that the image should transit head to tail in 1 day
        if "v" not in kwargs:
            width = opacitymat.shape[1]  # in pixels
            height = opacitymat.shape[0]  # in pixels
            rstar_pix = height/2  # R_star in pixels
            width = width/rstar_pix  # in units of R_star
            v = 2+width  # full transit of star from front to tail

        if "t_ref" not in kwargs:
            t_ref = 0.5

        if "t_arr" not in kwargs:
            t_arr = np.linspace(0.001, 0.999, 1000)

        super().__init__(
            opacitymat=opacitymat,
            LDlaw=LDlaw,
            LDCs=LDCs,
            positions=positions,
            areas=areas,
            blockedflux=blockedflux,
            LD=LD,
            v=v,
            t_ref=t_ref,
            t_arr=t_arr
        )

    def gen_ref_lc(self, gpu=False):
        """Generates a reference light curve with ~1000 datapoints

        The model is scaled so that the min flux is 0 and the max is 1.

        Note: If the default calculated variables failed to capture the whole
        transit, the maxm may not be 1.
        """
        ref_LC, _ = self.gen_LC(t_arr=self.t_arr, gpu=gpu)
        model_invert = 1-ref_LC  # invert to scale depth of transit
        model_invert = model_invert*(1/max(model_invert))  # scale 0 to 1
        self.ref_LC = 1-model_invert
        if (min(self.ref_LC) != 0) or (max(self.ref_LC) != 1):
            warn(f"""
    Warning: The reference LC does not have the expected flux range.
    Min flux: {min(self.ref.LC):.3}, Max flux: {max(self.ref.LC):.3}
            """)
        return self.ref_LC

    def signal_fit(self, times, width, depth, tref):
        return signal_fit(self.ref_LC, times, width, depth, tref)

    def show_model(self):
        if "plt" not in dir():
            import matplotlib.pyplot as plt
        try:
            fig, ax = plt.subplots(2,
                                   figsize=(6, 8),
                                   gridspec_kw={"height_ratios": [1, 1]},
                                   tight_layout=True)
            ax[0].imshow(self.opacitymat, cmap="Greys")
            ax[0].set_title("Reference shape");
            ax[0].set_xlabel("Pixel j");
            ax[0].set_ylabel("Pixel i");

            ax[1].plot(self.ref_LC)
            ax[1].set_title("Reference signal");
            ax[1].set_xlabel("Time step (unitless)");
            ax[1].set_ylabel("Normalized Flux (unitless)");

            plt.show()
        except AttributeError:
            print("""
    Reference light curve not yet generated.
    Call `model.gen_ref_lc()`.
            """)
        return


class CircleTransit(TransitModel):
    def __init__(self, gpu=False):
        opacitymat = CIRCLEMAT
        super().__init__(opacitymat=opacitymat)
        _ = self.gen_ref_lc(gpu=gpu)