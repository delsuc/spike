from spike.util.debug_tools import*  
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse, Rectangle
import matplotlib.lines as mlines
from spike.Visu.Pyside_PyQt4 import*

@dec_class_pr
@decclassdebugging
class ZOOM_TOOLS(object):# class regrouping methods about zoom
    def __init__(self, canv, interf, data, paramz, gtools):
        self.canv = canv
        self.interface = interf
        self.data = data
        self.paramz = paramz
        self.gtools = gtools
    
    
