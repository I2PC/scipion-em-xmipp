# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
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

from pyworkflow.object import String
from pyworkflow.protocol.params import StringParam

from pwem.protocols import ProtProcessParticles
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import writeSetOfParticles


class XmippProtAngMakeSymmetry(ProtProcessParticles):
    """
    Given an input set of particles with angular assignment which may not be in
    the asymmetric unit, find the location in the asymmetric unit equivalent to the input
    angles.

    Be aware that input symmetry values follows Xmipp conventions as described in:
    http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry
    """
    _label = 'make symmetry'

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help="See http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry"
                           " for a description of the symmetry groups format in Xmipp.\n"
                           "If no symmetry is present, use _c1_.")
    
    def _getDefaultParallel(self):
        """This protocol doesn't have mpi version"""
        return (0, 0)
     
    #--------------------------- INSERT steps functions --------------------------------------------            
    def _insertAllSteps(self):
        imgsFn = self._getPath('input_particles.xmd')
        self._insertFunctionStep('convertInputStep', imgsFn)
        self._insertFunctionStep('makeSymmetryStep', imgsFn)
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, outputFn):
        """ Create a metadata with the images and geometrical information. """
        writeSetOfParticles(self.inputParticles.get(), outputFn)

    #--------------------------- STEPS functions --------------------------------------------
    def makeSymmetryStep(self, imgsFn):
        outImagesMd = self._getPath('images.xmd')
        args = "-i Particles@%s --bringToAsymmetricUnit %s -o %s" % (imgsFn,
                                                 self.symmetryGroup.get(),
                                                 outImagesMd)
        self.runJob("xmipp_metadata_angles", args)
        self.outputMd = String(outImagesMd)

    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(imgSet)
        partSet.copyItems(imgSet,
                          updateItemCallback=self._createItemMatrix,
                          itemDataIterator=md.iterRows(self.outputMd.get(), sortByLabel=md.MDL_ITEM_ID))
        
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(imgSet, partSet)

    #--------------------------- INFO functions --------------------------------------------                
    def _summary(self):
        import os
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            summary.append("Symmetry: %s"% self.symmetryGroup.get())
        return summary
    
    def _validate(self):
        pass
        
    #--------------------------- Utils functions --------------------------------------------
    def _createItemMatrix(self, item, row):
        from xmipp3.convert import createItemMatrix

        createItemMatrix(item, row, align=ALIGN_PROJ)

