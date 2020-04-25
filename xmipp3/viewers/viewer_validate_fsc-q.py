# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from pyworkflow.utils import getExt, removeExt
from os.path import abspath


# from pyworkflow.em.viewers import LocalResolutionViewer, EmPlotter 
from pyworkflow.em.viewers.viewer_chimera import (Chimera,
                                                  sessionFile)
from pyworkflow.viewer import DESKTOP_TKINTER, Viewer


from pyworkflow.em.constants import (COLOR_JET, COLOR_TERRAIN,
 COLOR_GIST_EARTH, COLOR_GIST_NCAR, COLOR_GNU_PLOT, COLOR_GNU_PLOT2,
 COLOR_OTHER, COLOR_CHOICES, AX_X, AX_Y, AX_Z)
from pyworkflow.protocol.params import (LabelParam, StringParam, EnumParam,
                                        IntParam, EnumParam, LEVEL_ADVANCED)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pyworkflow.em.viewers import ChimeraView
from pyworkflow.em.metadata import MetaData, MDL_X, MDL_COUNT
from pyworkflow.em import ImageHandler

from .plotter import XmippPlotter
from xmipp3.protocols.protocol_validate_fitting import \
        XmippProtValFit, RESTA_FILE_MRC, OUTPUT_PDBMRC_FILE, \
        PDB_VALUE_FILE


class XmippProtValFitViewer(ProtocolViewer):
    """
    Visualization tools for validation fitting.
    
    DeepRes is a Xmipp package for computing the local resolution of 3D
    density maps studied in structural biology, primarily by cryo-electron
    microscopy (cryo-EM).
    """
    _label = 'viewer validation_fitting'
    _targets = [XmippProtValFit]      
    _environments = [DESKTOP_TKINTER]
    
    RESIDUE = 0
    ATOM = 1
    
    def __init__(self, *args, **kwargs):
        ProtocolViewer.__init__(self, *args, **kwargs)
    
    def _defineParams(self, form):
        self._env = os.environ.copy()
        form.addSection(label='Visualization')
        
        group = form.addGroup('Visualization')
        
        group.addParam('displayVolume', LabelParam,
                      important=True,
                      label='Display Volume Output')


        group.addParam('displayPDB', EnumParam,
                      choices=['by residue', 'by atom'],
                      default=0, important=True,
                      display=EnumParam.DISPLAY_COMBO,
                      label='Display PDB Output')  
            
        
    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        visualizeDict = {'displayVolume': self._visualize_vol,
                         'displayPDB': self._visualize_pdb}
        # If the is some error during the load, just show that instead
        # of any viewer
#         if self._errors:
#             for k in visualizeDict.keys():
#                 visualizeDict[k] = self._showErrors
        return visualizeDict    

    def _visualize_vol(self, obj, **args):
         
        # show coordinate axis
        fnRoot = os.path.abspath(self.protocol._getExtraPath())
        
        _inputVol = self.protocol.inputVolume.get()
#         dim = _inputVol.getDim()[0]
#         bildFileName = os.path.abspath(self.protocol._getTmpPath(
#             "axis_output.bild"))
#         Chimera.createCoordinateAxisFile(dim,
#                                  bildFileName=bildFileName,
#                                  sampling=_inputVol.getSamplingRate())
        fnCmd = self.protocol._getTmpPath("chimera_VOLoutput.cmd")
        f = open(fnCmd, 'w')
#         f.write("open %s\n" % bildFileName)
        # show volume
#         showVolFileName = os.path.abspath(
#                         ImageHandler.removeFileType(_inputVol.getFileName()))
        f.write("open %s\n" % (fnRoot+'/'+OUTPUT_PDBMRC_FILE))
        f.write("open %s\n" % (fnRoot+'/'+RESTA_FILE_MRC))
#         if _inputVol.hasOrigin():
#             x, y, z = _inputVol.getOrigin().getShifts()
#         else:
#             x, y, z = _inputVol.getOrigin(force=True).getShifts()
            
            
        f.write("volume #0 voxelSize %f step 1\n" % (_inputVol.getSamplingRate()))
        f.write("volume #1 voxelSize %f\n" % (_inputVol.getSamplingRate()))
        f.write("vol #1 hide\n")
        f.write("scolor #0 volume #1 perPixel false cmap -3,#ff0000:"
                "0,#ffff00:1,#00ff00:2,#00ffff:3,#0000ff\n")
        f.write("colorkey 0.01,0.05 0.02,0.95 -3 #ff0000 -2 #ff4500 -1 #ff7f00 "
                 "0 #ffff00 1  #00ff00 2 #00ffff 3 #0000ff\n")    

#         f.write("volume #1 style surface voxelSize %f\n"
#                 "volume #1 origin %0.2f,%0.2f,%0.2f\n"
#                 % (_inputVol.getSamplingRate(), x, y, z))
#         f.write("cofr #0\n")  # set center of coordinates       

        f.close()

        # run in the background
        Chimera.runProgram(Chimera.getProgram(), fnCmd+"&")
        return []
    
    
    def _visualize_pdb(self, obj, **args):
        
        # show coordinate axis
        fnRoot = os.path.abspath(self.protocol._getExtraPath())
        fnCmd = self.protocol._getTmpPath("chimera_PDBoutput.cmd")
        f = open(fnCmd, 'w')
        #open PDB

        if self.displayPDB == self.RESIDUE:
            f.write("open %s\n" % (fnRoot+'/'+PDB_VALUE_FILE))
            f.write("rangecol occupancy,r -3 red 0 white 1 green 2 cyan 3 blue\n")
        else:
            f.write("open %s\n" % (fnRoot+'/'+PDB_VALUE_FILE))
            f.write("display\n")
            f.write("~ribbon\n")
            f.write("rangecol occupancy,a -3 red 0 white 1 green 2 cyan 3 blue\n")  
        f.write("colorkey 0.01,0.05 0.02,0.95 -3 #ff0000 -2 #ff4500 -1 #ff7f00 "  
                "0 white 1  #00ff00 2 #00ffff 3 #0000ff\n")    
        f.close()  
                     
        Chimera.runProgram(Chimera.getProgram(), fnCmd+"&")
        return []   
     
   



