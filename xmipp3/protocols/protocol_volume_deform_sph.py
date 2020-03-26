
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

from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pwem.emlib.image import ImageHandler
from pwem.objects import Volume
from pyworkflow import VERSION_2_0


class XmippProtVolumeDeformSPH(ProtAnalysis3D):
    """ Protocol for volume deformation based on spherical harmonics. """
    _label = 'sph volume deform'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolume', params.PointerParam, label="Input volume",
                      pointerClass='Volume')
        form.addParam('refVolume', params.PointerParam, label="Reference volume",
                      pointerClass='Volume')
        form.addParam('sigma', params.NumericListParam, label="Multiresolution", default="1 2",
                      help="Perform the analysys comparing different filtered versions of the volumes")
        form.addParam('targetResolution', params.FloatParam, label="Target resolution",
                      default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('Rmax', params.IntParam, default=0,
                      label='Sphere radius',
                      experLevel=params.LEVEL_ADVANCED,
                      help='Radius of the sphere where the spherical harmonics will be computed.')
        form.addParam('depth', params.IntParam, default=3,
                      label='Harmonical depth',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Harmonical depth of the deformation=1,2,3,...')


    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'fnRefVol': self._getExtraPath('ref_volume.vol'),
            'fnInputVol': self._getExtraPath('input_volume.vol'),
            'fnInputFilt': self._getExtraPath('input_volume_filt.vol'),
            'fnRefFilt': self._getExtraPath('ref_volume_filt.vol'),
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

        XdimI = self.inputVolume.get().getDim()[0]
        TsI = self.inputVolume.get().getSamplingRate()
        XdimR = self.refVolume.get().getDim()[0]
        TsR = self.refVolume.get().getSamplingRate()
        self.newTs = self.targetResolution.get() * 1.0 / 3.0
        self.newTs = max(TsI, TsR, self.newTs)
        newXdimI = XdimI * TsI / self.newTs
        newXdimR = XdimR * TsR / self.newTs
        newRmax = self.Rmax.get() * TsI / self.newTs
        self.newXdim = min(newXdimI, newXdimR)
        self.newRmax = min(newRmax, self.Rmax.get())

        ih = ImageHandler()
        ih.convert(self.inputVolume.get(), fnInputVol)
        if XdimI != newXdimI:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (fnInputVol, newXdimI))
        if newXdimI>self.newXdim:
            self.runJob("xmipp_transform_window", " -i %s --crop %d" %
                        (fnInputVol, (newXdimI-self.newXdim)))


        ih.convert(self.refVolume.get(), fnRefVol)

        if XdimR != newXdimR:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (fnRefVol, newXdimR))
        if newXdimR>self.newXdim:
            self.runJob("xmipp_transform_window", " -i %s --crop %d" %
                        (fnRefVol, (newXdimR-self.newXdim)))


    def deformStep(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnRefVol = self._getFileName('fnRefVol')
        fnOutVol = self._getFileName('fnOutVol')

        self.alignSimulated()

        # Normalize the volumes to be passed to SPH program
        # params = " -i %s --method Robust --mask circular 50 --clip -o %s" % (fnRefVol, fnRefVol)
        # self.runJob("xmipp_transform_normalize", params)
        #
        # params = " -i %s --method Robust --mask circular 50 --clip -o %s" % (fnOutVol, fnOutVol)
        # self.runJob("xmipp_transform_normalize", params)

        params = ' -i %s -r %s -o %s --analyzeStrain --depth %d --sigma "%s"' % \
                 (fnOutVol, fnRefVol, fnOutVol, self.depth.get(), self.sigma.get())
        if self.newRmax != 0:
            params = params + ' --Rmax %d' % self.newRmax
        self.runJob("xmipp_volume_deform_sph", params)


    def createOutputStep(self):
        vol = Volume()
        vol.setLocation(self._getFileName('fnOutVol'))
        vol.setSamplingRate(self.newTs)
        self._defineOutputs(outputVolume=vol)
        self._defineSourceRelation(self.inputVolume, vol)

    def alignSimulated(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnInputFilt = self._getFileName('fnInputFilt')
        fnRefVol = self._getFileName('fnRefVol')
        fnRefFilt = self._getFileName('fnRefFilt')
        fnOutVol = self._getFileName('fnOutVol')

        # Filter the volumes to improve alignment quality
        params = " -i %s -o %s --fourier real_gaussian 2" %  (fnInputVol, fnInputFilt)
        self.runJob("xmipp_transform_filter", params)
        params = " -i %s -o %s --fourier real_gaussian 2" % (fnRefVol, fnRefFilt)
        self.runJob("xmipp_transform_filter", params)

        # Find transformation needed to align the volumes
        params = ' --i1 %s --i2 %s --apply residual.vol --local --dontScale ' \
                 '--copyGeo %s' % \
                 (fnRefFilt, fnInputFilt, self._getExtraPath("geo.txt"))
        self.runJob("xmipp_volume_align", params)

        # Apply transformation of filtered volume to original volume
        with open(self._getExtraPath("geo.txt"), 'r') as file:
            geo_str = file.read().replace('\n', ',')
        params = " -i %s -o %s --matrix %s --dont_wrap" % (fnInputVol, fnOutVol, geo_str)
        self.runJob("xmipp_transform_geometry", params)