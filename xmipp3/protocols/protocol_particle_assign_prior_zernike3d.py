
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es)
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

import pyworkflow.protocol.params as params
from pyworkflow.object import Integer
from pyworkflow.utils.path import moveFile
from pyworkflow import VERSION_2_0

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.emlib.image import ImageHandler
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import (writeSetOfParticles, createItemMatrix,
                            setXmippAttributes)
from xmipp3.base import writeInfoField, readInfoField


class XmippProtAssignPriorsZernike3D(ProtAnalysis3D):
    """ Assign Zernike3D to a SetOfParticles """
    _label = 'assign priors - Zernike3D'
    _lastUpdateVersion = VERSION_2_0

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
        form.addParam('particles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles',
                      help="Zernike3D priors will be assigned to these particles")
        form.addParam('volumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      label="Input volume(s)", important=True,
                      help='Volumes with priors to be assigned to the particles')
        form.addParam('targetResolution', params.FloatParam, label="Target resolution (A)", default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('maxShift', params.FloatParam, default=-1,
                      label='Maximum shift (px)', expertLevel=params.LEVEL_ADVANCED,
                      help='Maximum shift allowed in pixels')
        form.addParam('maxAngular', params.FloatParam, default=5,
                      label='Maximum angular change (degrees)', expertLevel=params.LEVEL_ADVANCED,
                      help='Maximum angular change allowed (in degrees)')
        form.addParam('maxResolution', params.FloatParam, default=4.0,
                      label='Maximum resolution (A)', expertLevel=params.LEVEL_ADVANCED,
                      help='Maximum resolution (A)')
        form.addParam('regularization', params.FloatParam, default=0.005, label='Regularization',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Penalization to deformations (higher values penalize more the deformation).')
        form.addParam('ignoreCTF', params.BooleanParam, default=False, label='Ignore CTF?',
                      expertLevel=params.LEVEL_ADVANCED,
                      help="If true, volume projection won't be subjected to CTF corrections")
        form.addParam('optimizeAlignment', params.BooleanParam, default=True, label='Optimize alignment?',
                     expertLevel=params.LEVEL_ADVANCED)
        form.addParallelSection(threads=1, mpi=8)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('input_volume.vol'),
            'fnVolMask': self._getExtraPath('input_volume_mask.vol'),
            'fnOut': self._getExtraPath('output_particles.xmd'),
            'fnOutDir': self._getExtraPath()
                 }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertStep)
        self._insertFunctionStep(self.assignPriorsStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnVolMask = self._getFileName('fnVolMask')

        inputParticles = self.particles.get()
        writeSetOfParticles(inputParticles, imgsFn)
        Xdim = inputParticles.getXDim()
        self.Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0 / 3.0
        self.newTs = max(self.Ts, newTs)
        self.newXdim = int(Xdim * self.Ts / newTs)
        writeInfoField(self._getExtraPath(), "sampling", md.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", md.MDL_XSIZE, self.newXdim)
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (imgsFn,
                         self._getExtraPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         self.newXdim), numberOfMpi=1)
            moveFile(self._getExtraPath('scaled_particles.xmd'), imgsFn)

        volumes = self.volumes.get()
        reference = volumes.refMap.get()
        mask = volumes.refMask.get()
        ih = ImageHandler()
        ih.convert(reference, fnVol)
        Xdim = reference.getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d " % (fnVol, self.newXdim), numberOfMpi=self.numberOfMpi.get())
        if mask:
            ih.convert(mask, fnVolMask)
            if Xdim != self.newXdim:
                self.runJob("xmipp_image_resize",
                            "-i %s -o %s --dim %d" % (fnVolMask, fnVolMask, self.newXdim))
                self.runJob("xmipp_transform_threshold",
                            "-i %s -o %s --select below 0.001 "
                            "--substitute binarize" % (fnVolMask, fnVolMask))

    def assignPriorsStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnOut = self._getFileName('fnOut')
        fnVolMask = self._getFileName('fnVolMask')
        fnOutDir = self._getFileName('fnOutDir')

        # Zernike parameters
        volumes = self.volumes.get()
        L1 = volumes.L1
        L2 = volumes.L2

        # Zernike prior file
        z_clnm_file = self._getExtraPath("z_clnm_priors.txt")
        self.writeZernikeFile(z_clnm_file)

        Ts = readInfoField(self._getExtraPath(), "sampling", md.MDL_SAMPLINGRATE)
        params = ' -i %s --ref %s -o %s --l1 %d --l2 %d --max_shift %f --priors %s ' \
                 '--max_angular_change %f --sampling %f --max_resolution %f ' \
                 '--odir %s --resume --regularization %f --image_mode 1 ' \
                 '--step 1 --blobr 1' %\
                 (imgsFn, fnVol, fnOut, L1, L2, self.maxShift, z_clnm_file,
                  self.maxAngular, Ts, self.maxResolution, fnOutDir, self.regularization.get())
        if self.optimizeAlignment.get():
            params += ' --optimizeAlignment'
        if not self.ignoreCTF.get():
            params += ' --useCTF'
        if self.inputParticles.get().isPhaseFlipped():
            params += ' --phaseFlipped'
        if self.mask.get():
            params += ' --mask %s' % fnVolMask

        # if self.useGpu.get():
        #     params += ' --device %d' % self.getGpuList()[0]
        #     program = 'xmipp_cuda_angular_sph_alignment'
        #     self.runJob(program, params)
        # else:
        #     program = 'xmipp_angular_sph_alignment'
        #     self.runJob(program, params, numberOfMpi=self.numberOfMpi.get())
        program = 'xmipp_forward_zernike_images_priors'
        self.runJob(program, params, numberOfMpi=self.numberOfMpi.get())


    def createOutputStep(self):
        Xdim = self.particles.get().getXDim()
        self.Ts = self.particles.get().getSamplingRate()
        newTs = self.particles.get() * 1.0 / 3.0
        self.newTs = max(self.Ts, newTs)
        self.newXdim = int(Xdim * self.Ts / newTs)
        fnOut = self._getFileName('fnOut')
        mdOut = md.MetaData(fnOut)

        # Zernike parameters
        volumes = self.volumes.get()
        L1 = volumes.L1
        L2 = volumes.L2
        Rmax = volumes.Rmax
        refMap = volumes.refMap
        refMask = volumes.refMask

        newMdOut = md.MetaData()
        i = 0
        for row in md.iterRows(mdOut):
            newRow = row
            if self.newTs != self.Ts:
                coeffs = mdOut.getValue(md.MDL_SPH_COEFFICIENTS, row.getObjId())
                correctionFactor = volumes[1].getDim()[0] / self.newXdim
                coeffs = [correctionFactor * coeff for coeff in coeffs]
                newRow.setValue(md.MDL_SPH_COEFFICIENTS, coeffs)
            newRow.addToMd(newMdOut)
            i += 1
        newMdOut.write(fnOut)

        inputSet = self.particles.get()
        partSet = self._createSetOfParticles()

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(fnOut, sortByLabel=md.MDL_ITEM_ID))
        partSet.L1 = L1
        partSet.L2 = L2
        partSet.Rmax = Rmax
        partSet.refMap = refMap
        partSet.mask = refMask

        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(self.inputParticles, partSet)

# --------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_FLIP, md.MDL_SPH_DEFORMATION,
                           md.MDL_SPH_COEFFICIENTS, md.MDL_SPH_TSNE_COEFF1D,
                           md.MDL_SPH_TSNE_COEFF2D)
        createItemMatrix(item, row, align=ALIGN_PROJ)

    def getInputParticles(self):
        return self.inputParticles.get()

    def writeZernikeFile(self, file):
        volumes = self.volumes.get()
        L1 = volumes.L1
        L2 = volumes.L2
        R = int(0.5 * self.newXdim)
        factor = self.newXdim / volumes[1].getDim()[0]
        z_clnm_v = factor * volumes.aggregate(["MAX"], "_xmipp_sphCoefficients", ["_xmipp_sphCoefficients"])

        with open(file, 'w') as f:
            f.write(' '.join(map(str, [L1, L2, R])) + "\n")
            for z_clnm in z_clnm_v:
                f.write(' '.join(map(str, z_clnm.reshape(-1))) + "\n")

    # ----------------------- VALIDATE functions ----------------------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        l1 = self.l1.get()
        l2 = self.l2.get()
        if (l1 - l2) < 0:
            errors.append('Zernike degree must be higher than '
                          'SPH degree.')
        return errors
