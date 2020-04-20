
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

import pyworkflow.protocol.params as params
from pyworkflow.utils.path import moveFile

from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pwem.emlib.image import ImageHandler
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import (writeSetOfParticles, createItemMatrix,
                            setXmippAttributes)
from xmipp3.base import writeInfoField, readInfoField
import numpy as np

from pyworkflow import VERSION_2_0


class XmippProtAngularAlignmentSPH(ProtAnalysis3D):
    """ Protocol for flexible angular alignment based on spherical harmonics. """
    _label = 'sph angular align'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam, label="Input particles", pointerClass='SetOfParticles')
        form.addParam('inputVolume', params.PointerParam, label="Input volume", pointerClass='Volume')
        form.addParam('targetResolution', params.FloatParam, label="Target resolution (A)", default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('depth', params.IntParam, default=3,
                      label='Harmonical depth', expertLevel=params.LEVEL_ADVANCED,
                      help='Harmonical depth of the deformation=1,2,3,...')
        form.addParam('maxShift', params.FloatParam, default=-1,
                      label='Maximum shift (px)', expertLevel=params.LEVEL_ADVANCED,
                      help='Maximum shift allowed in pixels')
        form.addParam('maxAngular', params.FloatParam, default=5,
                      label='Maximum angular change (degrees)', expertLevel=params.LEVEL_ADVANCED,
                      help='Maximum angular change allowed (in degrees)')
        form.addParam('maxResolution', params.FloatParam, default=4.0,
                      label='Maximum resolution (A)', expertLevel=params.LEVEL_ADVANCED,
                      help='Maximum resolution (A)')
        form.addParallelSection(threads=1, mpi=8)

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'imgsFn': self._getExtraPath('input_particles.xmd'),
            'fnVol': self._getExtraPath('input_volume.vol'),
            'fnOut': self._getExtraPath('output_particles.xmd'),
            'fnOutDir': self._getExtraPath()
                 }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("alignmentStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')

        inputParticles = self.inputParticles.get()
        writeSetOfParticles(inputParticles, imgsFn)
        Xdim = inputParticles.getXDim()
        Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0 /3.0
        newTs = max(Ts, newTs)
        self.newXdim = int(Xdim * Ts / newTs)
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

        ih = ImageHandler()
        ih.convert(self.inputVolume.get(), fnVol)
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize"
                        ,"-i %s --dim %d " %(fnVol, self.newXdim), numberOfMpi=1)


    def alignmentStep(self):
        imgsFn = self._getFileName('imgsFn')
        fnVol = self._getFileName('fnVol')
        fnOut =  self._getFileName('fnOut')
        fnOutDir = self._getFileName('fnOutDir')
        Ts = readInfoField(self._getExtraPath(), "sampling", md.MDL_SAMPLINGRATE)
        params = ' -i %s --ref %s -o %s --optimizeAlignment --optimizeDeformation ' \
                 '--depth %d --max_shift %f --max_angular_change %f --sampling %f ' \
                 ' --max_resolution %f --odir %s --resume' %\
                 (imgsFn, fnVol, fnOut, self.depth, self.maxShift, self.maxAngular,
                  Ts, self.maxResolution, fnOutDir)
        if self.inputParticles.get().isPhaseFlipped(): #preguntar
            params += ' --phaseFlipped'
        self.runJob("xmipp_angular_sph_alignment", params, numberOfMpi=self.numberOfMpi.get())


    def createOutputStep(self):
        from sklearn.manifold import TSNE
        fnOut = self._getFileName('fnOut')
        mdOut = md.MetaData(fnOut)
        i = 0
        for row in md.iterRows(mdOut):
            coeffs = mdOut.getValue(md.MDL_SPH_COEFFICIENTS, row.getObjId())
            if i==0:
                coeffMatrix = coeffs
            else:
                coeffMatrix = np.vstack((coeffMatrix, coeffs))
            i+=1
        X_tsne_1d = TSNE(n_components=1).fit_transform(coeffMatrix)
        X_tsne_2d = TSNE(n_components=2).fit_transform(coeffMatrix)

        newMdOut = md.MetaData()
        i=0
        for row in md.iterRows(mdOut):
            newRow = row
            newRow.setValue(md.MDL_SPH_TSNE_COEFF1D, float(X_tsne_1d[i,0]))
            newRow.setValue(md.MDL_SPH_TSNE_COEFF2D, [float(X_tsne_2d[i, 0]),  float(X_tsne_2d[i, 1])])
            newRow.addToMd(newMdOut)
            i+=1
            newMdOut.write(fnOut)

        inputSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()

        partSet.copyInfo(inputSet)
        partSet.setAlignmentProj()
        partSet.copyItems(inputSet,
                          updateItemCallback=self._updateParticle,
                          itemDataIterator=md.iterRows(fnOut, sortByLabel=md.MDL_ITEM_ID))

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





