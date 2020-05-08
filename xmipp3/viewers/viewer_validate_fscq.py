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
from pwem.viewers.viewer_chimera import (Chimera)
from pyworkflow.protocol.params import (LabelParam, EnumParam)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from xmipp3.protocols.protocol_validate_fscq import (XmippProtValFit, 
                                                     RESTA_FILE_MRC, 
                                                     OUTPUT_PDBMRC_FILE, 
                                                     PDB_VALUE_FILE)


class XmippProtValFitViewer(ProtocolViewer):
    """
    Visualization tools for validation fsc-q.
    
    FSC-Q is a Xmipp package for evaluate the map-to-model fit
    """
    _label = 'viewer validation_fsc-q'
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
        return visualizeDict    

    def _visualize_vol(self, obj, **args):
         

        fnRoot = os.path.abspath(self.protocol._getExtraPath())
        
        _inputVol = self.protocol.inputVolume.get()
        fnCmd = self.protocol._getTmpPath("chimera_VOLoutput.cmd")
        
        f = open(fnCmd, 'w')

        f.write("open %s\n" % (fnRoot+'/'+OUTPUT_PDBMRC_FILE))
        f.write("open %s\n" % (fnRoot+'/'+RESTA_FILE_MRC))           
        f.write("volume #0 voxelSize %f step 1\n" % (_inputVol.getSamplingRate()))
        f.write("volume #1 voxelSize %f\n" % (_inputVol.getSamplingRate()))
        f.write("vol #1 hide\n")
        f.write("scolor #0 volume #1 perPixel false cmap -3,#ff0000:"
                "0,#ffff00:1,#00ff00:2,#00ffff:3,#0000ff\n")
        f.write("colorkey 0.01,0.05 0.02,0.95 -3 #ff0000 -2 #ff4500 -1 #ff7f00 "
                 "0 #ffff00 1  #00ff00 2 #00ffff 3 #0000ff\n")        

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
     
   



