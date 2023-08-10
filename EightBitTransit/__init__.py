# __all__ = ['TransitingImage','GridFunctions','misc','inversion']

__author__ = """Daniel Giles"""
__email__ = 'daniel.k.giles@gmail.com'
__version__ = '2.1.0'

from .cGridFunctions import pixelate_image, lowres_grid, LDfluxsmall
from .misc import (
    lowres_grid_ti,
    change_res_grid,
    sigmoid_opacities,
    continuous_opacities,
    RMS,
    RMS_penalty,
    b_penalty,
    toBinaryGrid,
    fromBinaryGrid,
    calculateLCdecrements,
    ternary,
    perimeter,
    compactness
)
from .inversion import (
    bruteForceSearch,
    nCr,
    makeArcBasisParsimony,
    makeArcBasisAverage,
    makeArcBasisCombinatoric,
    renormBasis, 
    whoAreMyArcNeighbors,
    arcRearrange,
    Gaussian2D_PDF,
    simultaneous_ART,
    wedgeRearrange,
    wedgeNegativeEdge,
    wedgeOptimize_sym,
    foldOpacities,
    invertLC,
    symmetrize_design_matrix,
    symmetrize_opacity_map,
    get_overlap_time_mask,
    unfold_opacity_map
)
from .TransitingImage import TransitingImage
from .transitmodel import scale_transit, signal_fit, TransitModel, CircleTransit
from .mats import CIRCLEMAT
from .pixeloverlap import positions, chord_area, numpy_sign, overlap, pixel_overlap, initialize
initialize(gpu=False)