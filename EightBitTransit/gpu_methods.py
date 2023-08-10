from .pixeloverlap import overlap, pixel_overlap
from numba.cuda.cudadrv.nvvm import NvvmSupportError
from numba import vectorize
from warnings import warn

try:
    @vectorize(['float32(float32, float32, float32, boolean)'], target='cuda')
    def pixel_overlap_gpu(x0, y0, w, verbose=False):
        return overlap(x0, y0, w, verbose)
except NvvmSupportError:
    warn("""
    NvvmSupportError: libNVVM cannot be found. Do `conda install cudatoolkit`:
    libnvvm.so: cannot open shared object file: No such file or directory

    Initializing EightBitTransit *without* gpu multiprocessing.
    """)
    pixel_overlap_gpu = pixel_overlap
    pass