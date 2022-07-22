
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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


import numpy as np
import os

from pyworkflow.object import String, Integer
from pyworkflow import VERSION_2_0
from pyworkflow.utils import copyFile, getExt
import pyworkflow.protocol.params as params

from pwem.protocols import ProtAnalysis3D
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfVolumes, Volume


class XmippProtComputeHeterogeneityPriorsZernike3D(ProtAnalysis3D):
    """ Compute Zernike3D priors and assign them to a SetOfVolumes """
    _label = 'compute heterogeneity priors - Zernike3D'
    _lastUpdateVersion = VERSION_2_0
    OUTPUT_SUFFIX = '_%d_crop.mrc'
    ALIGNED_VOL = 'vol%dAligned.mrc'
    OUTPUT_PREFIX = "zernikeVolumes"

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        # form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
        #                label="Use GPU for execution",
        #                help="This protocol has both CPU and GPU implementation.\
        #                    Select the one you want to use.")
        # form.addHidden(params.GPU_LIST, params.StringParam, default='0',
        #                expertLevel=params.LEVEL_ADVANCED,
        #                label="Choose GPU IDs",
        #                help="Add a list of GPU devices that can be used")
        form.addParam('inputVolumes', params.MultiPointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Volumes used to compute the Zernike3D priors')
        form.addParam('reference', params.PointerParam,
                      pointerClass='Volume',
                      label="Zernike3D reference map", important=True,
                      help='Priors computed will be refered to this reference map')
        form.addParam('mask', params.PointerParam,
                      pointerClass='VolumeMask',
                      label="Reference map mask", allowsNull=True,
                      help='Mask determining where to compute the deformation field in the '
                           'reference volume. The tightest the mask, the higher the '
                           'performance boost')
        form.addParam('boxSize', params.IntParam, default=128,
                      label='Downsample particles to this box size', expertLevel=params.LEVEL_ADVANCED,
                      help='In general, downsampling the volumes will increase performance without compromising '
                           'the estimation the deformation field for each particle. Note that output particles will '
                           'have the original box size, and Zernike3D coefficients will be modified to work with the '
                           'original size volumes')
        form.addParam('Rmax', params.IntParam, default=0,
                      label='Sphere radius',
                      experLevel=params.LEVEL_ADVANCED,
                      help='Radius of the sphere where the spherical harmonics will be computed (in voxels).')
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('penalization', params.FloatParam, default=0.00025, label='Regularization',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Penalization to deformations (higher values penalize more the deformation).')
        form.addParallelSection(threads=1, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        reference = self.reference.get()
        volList = [reference.getFileName()]
        dimList = [reference.getDim()[0]]
        srList = [reference.getSamplingRate()]
        volList, dimList, srList, _ = self._iterInputVolumes(self.inputVolumes, volList, dimList, srList, [])

        # Resize volumes according to new sampling rate
        nVoli = 1
        depsConvert = []
        for _ in volList:
            convert = self._insertFunctionStep(self.convertStep, volList[nVoli - 1],
                                     dimList[nVoli - 1], srList[nVoli - 1],
                                     min(dimList), max(srList), nVoli, prerequisites=[])
            depsConvert.append(convert)
            nVoli += 1

        stepID = self._insertFunctionStep(self.preprocessMask, max(srList), prerequisites=depsConvert)
        depsMask = [stepID]

        # Align all volumes to the first one (reference)
        nVoli = 1
        nVolj = 2
        depsAlign = []
        for _ in volList[1:]:
            stepID = self._insertFunctionStep(self.alignStep, volList[nVolj - 1],
                                              volList[nVoli - 1], nVoli - 1,
                                              nVolj - 1, prerequisites=depsMask)
            depsAlign.append(stepID)
            nVolj += 1

        # Zernikes3D
        nVoli = 1
        nVolj = 2
        depsZernike = []
        for _ in volList[1:]:
            stepID = self._insertFunctionStep(self.computePriorsStep, volList[nVolj - 1],
                                              volList[nVoli - 1], nVoli - 1,
                                              nVolj - 1, nVolj,  prerequisites=depsAlign)
            depsZernike.append(stepID)
            nVolj += 1

        self._insertFunctionStep(self.createOutputStep, volList, prerequisites=depsZernike)

    # --------------------------- STEPS functions ---------------------------------------------------
    def preprocessMask(self, maxSr):
        mask = self.mask.get()

        if mask:
            maskFn = mask.getFileName()
            fnOut = self._getExtraPath("mask_reference.mrc")
            Xdim = mask.getDim()[0]
            # Ts = mask.getSamplingRate()
            # newTs = self.targetResolution.get() * 1.0 / 3.0
            # newTs = max(maxSr, newTs)
            # newXdim = Xdim * Ts / newTs
            newXdim = self.boxSize.get()

            ih = ImageHandler()
            volFn = maskFn if getExt(maskFn) == '.vol' else maskFn + ':mrc'
            if Xdim != newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s -o %s --dim %d" % (volFn, fnOut, newXdim))
                self.runJob("xmipp_transform_threshold",
                            "-i %s -o %s --select below 0.001 "
                            "--substitute binarize" % (fnOut, fnOut))
            else:
                ih.convert(volFn, fnOut)

    def convertStep(self, volFn, volDim, volSr, minDim, maxSr, nVoli):
        Xdim = volDim
        Ts = volSr
        # newTs = self.targetResolution.get() * 1.0 / 3.0
        # newTs = max(maxSr, newTs)
        # newXdim = Xdim * Ts / newTs
        newXdim = self.boxSize.get()
        correctionFactor = newXdim / Xdim
        self.newRmax = self.Rmax.get() * correctionFactor
        fnOut = os.path.splitext(volFn)[0]
        fnOut = self._getExtraPath(os.path.basename(fnOut + self.OUTPUT_SUFFIX % nVoli))

        ih = ImageHandler()
        volFn = volFn if getExt(volFn) == '.vol' else volFn + ':mrc'
        if Xdim != newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --dim %d" % (volFn, fnOut, newXdim))

        else:
            ih.convert(volFn, fnOut)

        if newXdim>minDim:
            self.runJob("xmipp_transform_window", " -i %s -o %s --crop %d" %
                        (fnOut, fnOut, (newXdim - minDim)))

    def alignStep(self, inputVolFn, refVolFn, i, j):
        inputVolFn = self._getExtraPath(os.path.basename(os.path.splitext(inputVolFn)[0]
                                                         + self.OUTPUT_SUFFIX % (j+1)))
        refVolFn = self._getExtraPath(os.path.basename(os.path.splitext(refVolFn)[0]
                                                       + self.OUTPUT_SUFFIX % (i+1)))
        fnOut = self._getTmpPath(self.ALIGNED_VOL % (j + 1))
        params = ' --i1 %s --i2 %s --apply %s --local --dontScale' % \
                 (refVolFn, inputVolFn, fnOut)

        self.runJob("xmipp_volume_align", params)

    def computePriorsStep(self, inputVolFn, refVolFn, i, j, step_id):
        if i == 0:
            fnOut_aux = self._getExtraPath(os.path.basename(os.path.splitext(inputVolFn)[0]
                                                            + self.OUTPUT_SUFFIX % (j + 1)))
        else:
            fnOut_aux = self._getTmpPath(self.ALIGNED_VOL % (j + 1))
        refVolFn = self._getExtraPath(os.path.basename(os.path.splitext(refVolFn)[0]
                                                       + self.OUTPUT_SUFFIX % (i + 1)))
        fnOut = self._getTmpPath("input_%d.mrc" % step_id)
        copyFile(fnOut_aux, fnOut)
        fnOut2 = self._getExtraPath('vol%dDeformedTo%d.mrc' % (i + 1, j + 1))

        params = ' -i %s -r %s -o %s --l1 %d --l2 %d --step 1 --blobr 1 --oroot %s --regularization %f' %\
                 (refVolFn, fnOut, fnOut2, self.l1.get(), self.l2.get(),
                  self._getExtraPath('Pair_%d_%d' % (i, j)), self.penalization.get())

        if self.mask.get():
            params += " --maski %s" % self._getExtraPath("mask_reference.mrc")

        if self.newRmax != 0:
            params = params + ' --Rmax %d' % self.newRmax

        # if self.useGpu.get():
        #     self.runJob("xmipp_cuda_volume_deform_sph", params)
        # else:
        #     params = params + ' --thr 1'
        #     self.runJob("xmipp_volume_deform_sph", params)

        self.runJob("xmipp_forward_zernike_volume", params)

    def createOutputStep(self, volList):
        reference = self.reference.get()
        mask = self.mask.get()
        dim = reference.getDim()[0]
        sr = reference.getSamplingRate()
        L1 = Integer(self.l1.get())
        L2 = Integer(self.l2.get())
        Rmax = Integer(0.5 * reference.getDim()[0])
        reference_filename = String(reference.getFileName())
        mask_filename = String(mask.getFileName()) if mask else String("")

        zernikeVols = self._createSetOfVolumes()
        zernikeVols.setSamplingRate(sr)
        zernikeVols.L1 = L1
        zernikeVols.L2 = L2
        zernikeVols.Rmax = Rmax
        zernikeVols.refMap = reference_filename
        zernikeVols.refMask = mask_filename

        nVolj = 2
        for _ in volList[1:]:
            zernikeVol = Volume()
            zernikeVol.setFileName(reference.getFileName())
            zernikeVol.setSamplingRate(sr)

            # Read Zernike coefficient
            z_clnm_file = self._getExtraPath('Pair_%d_%d_clnm.txt' % (0, nVolj - 1))
            z_clnm = self.readZernike3DFile(z_clnm_file)
            factor = dim / (2 * z_clnm[0][2])
            z_clnm = factor * z_clnm[1]
            zernikeVol._xmipp_sphCoefficients = String(','.join(['%f' % c for c in z_clnm]))
            zernikeVol.L1 = L1
            zernikeVol.L2 = L2
            zernikeVol.Rmax = Rmax
            zernikeVol.refMap = reference_filename
            zernikeVol.refMask = mask_filename

            zernikeVols.append(zernikeVol)

            nVolj += 1

        # Save new output
        name = self.OUTPUT_PREFIX
        args = {}
        args[name] = zernikeVols
        self._defineOutputs(**args)
        self._defineSourceRelation(reference, zernikeVols)
        self._updateOutputSet(name, zernikeVols, state=zernikeVols.STREAM_CLOSED)

    # --------------------------- UTILS functions --------------------------------------------
    def _iterInputVolumes(self, volumes, volList, dimList, srList, idList):
        """ Iterate over all the input volumes. """
        count = 1
        for pointer in volumes:
            item = pointer.get()
            if item is None:
                break
            if isinstance(item, Volume):
                volList.append(item.getFileName())
                dimList.append(item.getDim()[0])
                srList.append(item.getSamplingRate())
                idList.append(count)
            elif isinstance(item, SetOfVolumes):
                for vol in item:
                    volList.append(vol.getFileName())
                    dimList.append(vol.getDim()[0])
                    srList.append(vol.getSamplingRate())
                    idList.append(count)
                    count += 1
            count += 1
        return volList, dimList, srList, idList

    def readZernike3DFile(self, file):
        z_clnm = []
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                z_clnm.append(np.fromstring(line, dtype=float, sep=' '))
        return z_clnm

    # ------------------------- VALIDATE functions -----------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        l1 = self.l1.get()
        l2 = self.l2.get()
        if (l1 - l2) < 0:
            errors.append('Zernike degree must be higher than '
                          'SPH degree.')
        return errors




