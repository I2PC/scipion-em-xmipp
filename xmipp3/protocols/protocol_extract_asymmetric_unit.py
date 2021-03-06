# **************************************************************************
# * Authors:     Marta Martinez (mmmtnez@cnb.csic.es)
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

from pyworkflow import VERSION_3_0
from pwem.objects import Volume
from pwem.constants import (SYM_DIHEDRAL_X,SCIPION_SYM_NAME)
from pwem.objects import Transform
from pwem.convert import Ccp4Header
from pwem.protocols import EMProtocol
from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        EnumParam, IntParam)

from xmipp3.constants import (XMIPP_SYM_NAME, XMIPP_TO_SCIPION, XMIPP_CYCLIC,
                              XMIPP_DIHEDRAL_X, XMIPP_TETRAHEDRAL, XMIPP_OCTAHEDRAL,
                              XMIPP_I222, XMIPP_I222r, XMIPP_In25, XMIPP_In25r)

DEBUG = True


class XmippProtExtractUnit(EMProtocol):
    """ generates files for volumes and FSCs to submit structures to EMDB
    """
    _label = 'extract asymmetric unit'
    _program = ""
    _version = VERSION_3_0

    def __init__(self, **kwargs):
        EMProtocol.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolumes', PointerParam, label="Input Volume",
                      important=True, pointerClass='Volume',
                      help='This volume will be cropped')
        form.addParam('symmetryGroup', EnumParam,
                      choices=[XMIPP_SYM_NAME[XMIPP_CYCLIC] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_CYCLIC]] + ")",
                               XMIPP_SYM_NAME[XMIPP_DIHEDRAL_X] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_DIHEDRAL_X]] + ")",
                               XMIPP_SYM_NAME[XMIPP_TETRAHEDRAL] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_TETRAHEDRAL]] + ")",
                               XMIPP_SYM_NAME[XMIPP_OCTAHEDRAL] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_OCTAHEDRAL]] + ")",
                               XMIPP_SYM_NAME[XMIPP_I222] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_I222]] + ")",
                               XMIPP_SYM_NAME[XMIPP_I222r] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_I222r]] + ")",
                               XMIPP_SYM_NAME[XMIPP_In25] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_In25]] + ")",
                               XMIPP_SYM_NAME[XMIPP_In25r] +
                               " (" + SCIPION_SYM_NAME[XMIPP_TO_SCIPION[XMIPP_In25r]] + ")"],
                      default=XMIPP_I222,
                      label="Symmetry",
                      help="See http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/"
                           "Symmetry for a description of the symmetry groups "
                           "format in Xmipp.\n"
                           "If no symmetry is present, use _c1_."
                      )
        form.addParam('symmetryOrder', IntParam, default=1,
                      condition='symmetryGroup<=%d' % SYM_DIHEDRAL_X,
                      label='Symmetry Order',
                      help='Order of cyclic symmetry.')
        form.addParam('offset', FloatParam, default=0.,
                      condition='symmetryGroup<=%d' % SYM_DIHEDRAL_X,
                      label="offset",
                      help="rotate unit cell around z-axis by offset degrees")
        form.addParam('innerRadius', FloatParam, default=-1,
                      label="Inner Radius (px)",
                      help="inner Mask radius, if -1, the radius will be 0")
        form.addParam('outerRadius', FloatParam, default=-1,
                      label="Outer Radius (px)",
                      help="outer Mask radius, if -1, the radius will be "
                           "volume_size/2")
        form.addParam('expandFactor', FloatParam, default=0.,
                      label="Expand Factor",
                      help="Increment cropped region by this factor")

    # --------------------------- INSERT steps functions ----------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('extractUnit')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions -----------------------------

    def extractUnit(self):
        sym = self.symmetryGroup.get()
        if sym == XMIPP_CYCLIC:
            sym = "%s%d" % (XMIPP_SYM_NAME[XMIPP_CYCLIC][:1], self.symmetryOrder)
        elif sym == XMIPP_DIHEDRAL_X:
            sym = "%s%d" % \
                  (XMIPP_SYM_NAME[XMIPP_DIHEDRAL_X][:1], self.symmetryOrder)
        elif sym == XMIPP_TETRAHEDRAL:
            sym = "%s" % XMIPP_SYM_NAME[XMIPP_TETRAHEDRAL]
        elif sym == XMIPP_OCTAHEDRAL:
            sym = "%s" % XMIPP_SYM_NAME[XMIPP_OCTAHEDRAL]
        elif sym >= XMIPP_I222 and sym <= XMIPP_In25r :
            sym = XMIPP_SYM_NAME[sym]
        
        inFileName = self.inputVolumes.get().getFileName()
        if inFileName.endswith('.mrc'):
            inFileName = inFileName + ":mrc"
        args = "-i %s -o %s" % \
               (inFileName, self._getOutputVol())
        args += " --unitcell %s " % sym
        args += " %f " % self.innerRadius.get()
        args += " %f " % self.outerRadius.get()
        args += " %f " % self.expandFactor.get()
        args += " %f " % self.offset.get()
        sampling = self.inputVolumes.get().getSamplingRate()
        args += " %f " % sampling
        origin = self.inputVolumes.get().getShiftsFromOrigin()
        # x origin coordinate (from Angstroms to pixels)
        args += " %f " % (origin[0] / (-1. * sampling))
        # y origin coordinate (from Angstroms to pixels)
        args += " %f " % (origin[1] / (-1. * sampling))
        # z origin coordinate (from Angstroms to pixels)
        args += " %f " % (origin[2] / (-1. * sampling))

        self.runJob("xmipp_transform_window", args)

    def createOutputStep(self):
        vol = Volume()
        vol.setLocation(self._getOutputVol())
        sampling = self.inputVolumes.get().getSamplingRate()
        vol.setSamplingRate(sampling)
        #

        ccp4header = Ccp4Header(self._getOutputVol(), readHeader=True)
        t = Transform()

        x, y, z = ccp4header.getOrigin()  # origin output vol
        # coordinates

        t.setShifts(x, y, z)
        vol.setOrigin(t)
        #
        self._defineOutputs(outputVolume=vol)
        self._defineSourceRelation(self.inputVolumes, self.outputVolume)

    # --------------------------- INFO functions ------------------------------
    def _validate(self):
        message = []
        return message

    def _summary(self):
        # message = "Data Available at : *%s*"% self.filesPath.get()
        message = ""
        return [message]

    def _methods(self):
        return []

    # --------------------------- UTILS functions -----------------------------

    def _getOutputVol(self):
        prefix = os.path.basename(self.inputVolumes.get().getFileName()).split(".")[0]

        return self._getExtraPath(prefix + "_output_volume.mrc")

    def replace_at_index(self, tup, ix, val):
        return tup[:ix] + (val,) + tup[ix + 1:]