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

import os

from pyworkflow.protocol.params import PointerParam, StringParam
from pwem.protocols import ProtAnalysis3D

import xmipp3

        
class XmippProtVolumeStrain(ProtAnalysis3D):
    """Compares two volume states to analyze local strains and rotations. This protocol helps study structural changes by quantifying deformation and dynamic behavior between different conformations."""
    _label = 'calculate strain'
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        
        form.addParam('inputVolume0', PointerParam, label="Initial state", important=True,
                      pointerClass='Volume',
                      help='Initial state of the structure, it will be deformed to fit into the final state')
        form.addParam('inputVolumeF', PointerParam, label="Final state", important=True,
                      pointerClass='Volume',
                      help='Initial state of the structure, it will be deformed to fit into the final state')
        form.addParam('inputMask', PointerParam, label="Mask for the final state", important=True,
                      pointerClass='VolumeMask',
                      help='Binary mask that defines where the strains and rotations will be calculated')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group', 
                      help='See https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry for a description of the symmetry groups format'
                        'If no symmetry is present, give c1')
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _getFileName(self, fnRoot, key, **kwargs):
        return "%s_%s.mrc"%(fnRoot,key)

    def _insertAllSteps(self):
        fnVol0 = self.inputVolume0.get().getFileName()
        fnVolF = self.inputVolumeF.get().getFileName()
        fnMask = self.inputMask.get().getFileName()
        self._insertFunctionStep(self.calculateStrain,fnVol0,fnVolF,fnMask)
        self._insertFunctionStep(self.prepareOutput)
        self._insertFunctionStep(self.createChimeraScript)
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def calculateStrain(self, fnVol0, fnVolF, fnMask):
        fnRoot=self._getExtraPath('result')
        mirtDir = xmipp3.base.getXmippPath('external', 'mirt')
        # -wait -nodesktop
        args=('''-r "diary('%s'); xmipp_calculate_strain('%s','%s','%s','%s'); exit"'''
              % (fnRoot+"_matlab.log",fnVolF,fnVol0,fnMask,fnRoot))
        self.runJob("matlab", args, env=xmipp3.Plugin.getMatlabEnviron(mirtDir))
    
    def prepareOutput(self):
        volDim = self.inputVolume0.get().getDim()[0]
        Ts=self.inputVolume0.get().getSamplingRate()
        fnRoot=self._getExtraPath('result')

        def symmetrize(key):
            self.runJob("xmipp_transform_symmetrize", "-i %s --sym %s --dont_wrap" % \
                        (self._getFileName(fnRoot, key), self.symmetryGroup.get()))

        def changeSamplingRate(key):
            self.runJob("xmipp_image_header", "-i %s --sampling_rate %f" % (self._getFileName(fnRoot, key), Ts))

        def convert(key):
            self.runJob("xmipp_image_convert", "-i %s_%s.raw#%d,%d,%d,0,float -o %s" %
                        (fnRoot, key, volDim, volDim, volDim, self._getFileName(fnRoot, key)))
            self.runJob("xmipp_transform_mirror", "-i %s --flipX" % self._getFileName(fnRoot, key))
            changeSamplingRate(key)

        convert("initial")
        convert("final")
        convert("initialDeformedToFinal")
        convert("strain")
        convert("localrot")

        self.runJob("rm","-f "+self._getExtraPath('result_*.raw'))
        if self.symmetryGroup!="c1":
            symmetrize("strain")
            symmetrize("localrot")
            changeSamplingRate("strain")
            changeSamplingRate("localrot")

    def createChimeraScript(self):
        fnRoot = "extra/result"
        scriptFile = self._getPath('result') + '_strain_chimera.cmd'

        openStr = "open %s\n"

        fhCmd = open(scriptFile, 'w')
        fhCmd.write(openStr % self._getFileName(fnRoot,"final"))
        fhCmd.write(openStr % self._getFileName(fnRoot,"strain"))
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("scolor #0 volume #1 cmap rainbow reverseColors True\n")
        fhCmd.close()

        scriptFile = self._getPath('result') + '_localrot_chimera.cmd'
        fhCmd = open(scriptFile, 'w')
        fhCmd.write(openStr % self._getFileName(fnRoot,"final"))
        fhCmd.write(openStr % self._getFileName(fnRoot,"localrot"))
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("scolor #0 volume #1 cmap rainbow reverseColors True\n")
        fhCmd.close()

        scriptFile = self._getPath('result') + '_morph_chimera.cmd'
        fhCmd = open(scriptFile, 'w')
        fhCmd.write(openStr % self._getFileName(fnRoot,"initial"))
        fhCmd.write(openStr % self._getFileName(fnRoot,"final"))
        fhCmd.write("vol #0 hide\n")
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("vop morph #0,1 frames 50\n")
        fhCmd.close()

    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        xdim0 = self.inputVolume0.get().getDim()[0]
        xdimF = self.inputVolumeF.get().getDim()[0]
        if xdim0 != xdimF:
            errors.append("Make sure that the two volumes have the same size")
        return errors    
