
# **************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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



from pyworkflow.em.protocol import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pyworkflow.em.convert import ImageHandler
from pyworkflow.em.data import Volume
from pyworkflow import VERSION_1_2


class XmippProtVolumeDeformSPH(ProtAnalysis3D):
    """ Protocol for volume deformation based on spherical harmonics. """
    _label = 'sph volume deform'
    _lastUpdateVersion = VERSION_1_2

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolume', params.PointerParam, label="Input volume",
                      pointerClass='Volume')
        form.addParam('refVolume', params.PointerParam, label="Reference volume",
                      pointerClass='Volume')
        form.addParam('targetResolution', params.FloatParam, label="Target resolution",
                      default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('depth', params.IntParam, default=1,
                      label='Harmonical depth',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Harmonical depth of the deformation=1,2,3,...')


    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'fnRefVol': self._getExtraPath('ref_volume.vol'),
            'fnInputVol': self._getExtraPath('input_volume.vol'),
            'fnOutVol': self._getExtraPath('vol1DeformedTo2.vol')
                 }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def convertStep(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnRefVol = self._getFileName('fnRefVol')

        Xdim = self.inputVolume.get().getDim()[0]
        Ts = self.inputVolume.get().getSamplingRate()
        self.newTs = self.targetResolution.get() * 1.0 / 3.0
        self.newTs = max(Ts, self.newTs)
        self.newXdim = long(Xdim * Ts / self.newTs)

        ih = ImageHandler()
        ih.convert(self.inputVolume.get(), fnInputVol)
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (fnInputVol, self.newXdim))

        ih.convert(self.refVolume.get(), fnRefVol)
        Xdim = self.refVolume.get().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (fnRefVol, self.newXdim))

    def deformStep(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnRefVol = self._getFileName('fnRefVol')
        fnOutVol = self._getFileName('fnOutVol')

        params = ' --i1 %s --i2 %s --apply %s --least_squares --local ' % \
                 (fnRefVol, fnInputVol, fnOutVol)
        self.runJob("xmipp_volume_align", params)


        params = ' -i %s -r %s -o %s --analyzeStrain --depth %d' % \
                 (fnOutVol, fnRefVol, fnOutVol, self.depth.get())
        self.runJob("xmipp_volume_deform_sph", params)


    def createOutputStep(self):
        vol = Volume()
        vol.setLocation(self._getFileName('fnOutVol'))
        vol.setSamplingRate(self.newTs)
        self._defineOutputs(outputVolume=vol)
        self._defineSourceRelation(self.inputVolume, vol)

