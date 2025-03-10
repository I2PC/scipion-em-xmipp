# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
# *              James Krieger        (jamesmkrieger@gmail.com)
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
from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, StringParam, USE_GPU, GPU_LIST,
                                        FloatParam, BooleanParam, IntParam)
from pyworkflow.protocol import STEPS_PARALLEL

from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume, SetOfParticles
import pwem.emlib.metadata as md
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem import emlib
from xmipp3.convert import setXmippAttributes
import xmippLib

import os

class XmippProtComputeLikelihood(ProtAnalysis3D):
    """This protocol computes the likelihood of a set of particles with assigned angles when compared to a
       set of maps or atomic models"""

    _label = 'compute likelihood'
    _lastUpdateVersion = VERSION_1_1
    _possibleOutputs = {"reprojections": SetOfParticles}
    
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self._classesInfo = dict()

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addParam('binThreads', IntParam,
                      label='threads',
                      default=2,
                      help='Number of threads used by Xmipp each time it is called in the protocol execution. For '
                           'example, if 3 Scipion threads and 3 Xmipp threads are set, the tomograms will be '
                           'processed in groups of 2 at the same time with a call of Xmipp with 3 threads each, so '
                           '6 threads will be used at the same time. Beware the memory of your machine has '
                           'memory enough to load together the number of tomograms specified by Scipion threads.')
        
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label="Input images", important=True,
                      pointerClass='SetOfParticles', pointerCondition='hasAlignmentProj')
        form.addParam('inputRefs', PointerParam, label="References", important=True,
                      pointerClass='Volume,SetOfVolumes',
                      help='Volume or set of volumes to which the set of particles will be compared')
        form.addParam('particleRadius', IntParam, label="Particle radius (px): ", default=-1,
                      help='This radius should include the particle but be small enough to leave room to create a ring for estimating noise\n'
                            'If left at -1, this will take half the image width. In this case, the whole circle will be used to estimate noise')
        form.addParam('noiseRadius', IntParam, label="Noise radius (px): ", default=-1,
                      help='This radius should be larger than the particle radius to create a ring for estimating noise\n'
                            'If left at -1, this will take half the image width.')

        form.addParam('newProg', BooleanParam, label="Use new program: ", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Whether to use new program xmipp_continuous_create_residuals. This removes the low-pass filter and '
                      'applies transformations to the projection, not the original image.')

        form.addParam('resol', FloatParam, label="Filter at resolution: ", default=0, expertLevel=LEVEL_ADVANCED,
                      help='Resolution (A) at which subtraction will be performed, filtering the volume projections.'
                           'Value 0 implies no filtering.', condition='newProg==False')
        form.addParam('optimizeGray', BooleanParam, label="Optimize gray: ", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Optimize the gray value between the map reprojection and the experimental image')
        form.addParam('maxGrayChange', FloatParam, label="Max. gray change: ", default=0.99, expertLevel=LEVEL_ADVANCED,
                      condition='optimizeGray',
                      help='The actual gray value can be at most as small as 1-change or as large as 1+change')

        form.addParam('printTerms', BooleanParam, label="Print terms of LL: ", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Whether to print terms 1 and 2, LL and noise variance')

        form.addParam('useBothTerms', BooleanParam, label="Use both terms: ", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Whether to use both terms. Otherwise, just use term 1')

        form.addParam('useNegSos', BooleanParam, label="Use negative sum of squares: ", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Whether to use negative sum of squares instead of full variance-adjusted term 1')

        form.addParallelSection(threads=2, mpi=2)

        form.addHidden(USE_GPU, BooleanParam, default=False,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use. Be aware that the GPU program is new and may have problems")

        form.addHidden(GPU_LIST, StringParam, default='',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """ Convert input images then run continuous_assign2 then create output """
        
        convId = self._insertFunctionStep(self.convertStep, prerequisites=[], needsGPU=False)

        stepIds = []
        inputRefs = self.inputRefs.get()
        i=1
        if isinstance(inputRefs, Volume):
            prodId = self._insertFunctionStep(self.produceResidualsStep, inputRefs.getFileName(), i,
                                              prerequisites=convId, needsGPU=self.useGpu)
            i += 1
            stepIds.append(prodId)
        else:
            for volume in inputRefs:
                prodId = self._insertFunctionStep(self.produceResidualsStep, volume.getFileName(), i,
                                                  prerequisites=convId, needsGPU=self.useGpu)
                i += 1
                stepIds.append(prodId)

        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=stepIds,
                                 needsGPU=False)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        from ..convert import writeSetOfParticles
        imgSet = self.inputParticles.get()
        writeSetOfParticles(imgSet, self._getExtraPath("images.xmd"))

    def getMasks(self):
        if not (hasattr(self, 'mask') and hasattr(self, 'noiseMask')):
            Xdim = self.inputParticles.get().getDimensions()[0]
            Y, X = np.ogrid[:Xdim, :Xdim]
            dist_from_center = np.sqrt((X - Xdim/2) ** 2 + (Y - Xdim/2) ** 2)

            particleRadius = self.particleRadius.get()
            if particleRadius<0:
                particleRadius=Xdim/2
            self.mask = dist_from_center <= particleRadius

            noiseRadius = self.noiseRadius.get()
            if noiseRadius == -1:
                noiseRadius = Xdim/2
            if noiseRadius > particleRadius:
                self.noiseMask = (dist_from_center > particleRadius) & (dist_from_center <= noiseRadius)
            else:
                self.noiseMask = self.mask

        return self.mask, self.noiseMask

    def produceResidualsStep(self, fnVol, i):
        fnAngles = self._getExtraPath("images.xmd")

        if self.newProg:
            anglesOutFn = self._getExtraPath("anglesCont%03d.xmd"%i)
        else:
            anglesOutFn = self._getExtraPath("anglesCont%03d.stk"%i)

        fnResiduals = self._getExtraPath("residuals%03d.stk"%i)
        fnProjections = self._getExtraPath("projections%03d.stk"%i)

        Ts = self.inputParticles.get().getSamplingRate()
        args = "-i %s -o %s --ref %s --sampling %f --oresiduals %s --oprojections %s" % (fnAngles, anglesOutFn, fnVol, Ts, fnResiduals, fnProjections)

        if self.resol.get()>0:
            args+=" --max_resolution %f"%self.resol
        if self.optimizeGray:
            args+=" --optimizeGray --max_gray_scale %f"%self.maxGrayChange

        if self.useGpu:
            args+=" --nThreads %d"%self.binThreads.get()
            gpuId = self._stepsExecutor.getGpuList()
            if isinstance(gpuId, int):
                gpuStr = str(gpuId)
            else:
                gpuStr = ','.join([str(g) for g in gpuId])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuStr
            self.runJob('xmipp_cuda_angular_continuous_assign2', args, numberOfMpi=1)
        elif self.newProg:
            self.runJob("xmipp_continuous_create_residuals", args, numberOfMpi=self.numberOfMpi.get())
        else:
            self.runJob("xmipp_angular_continuous_assign2", args, numberOfMpi=self.numberOfMpi.get())

        mdResults = md.MetaData(self._getExtraPath("anglesCont%03d.xmd"%i))
        mdOut = md.MetaData()

        if self.printTerms.get():
                print('{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\n'.format('-sos', 'term1', 'term2',
                                                                                 'LL', 'var', '1/(2*var)', 'std'))
        for objId in mdResults:
            itemId = mdResults.getValue(emlib.MDL_ITEM_ID,objId)
            
            fnResidual = mdResults.getValue(emlib.MDL_IMAGE_RESIDUAL,objId)
            I = xmippLib.Image(fnResidual)

            elements_within_circle = I.getData()[self.getMasks()[0]]
            sum_of_squares = np.sum(elements_within_circle**2)
            Npix = elements_within_circle.size

            elements_between_circles = I.getData()[self.getMasks()[1]]
            var = np.var(elements_between_circles)

            if self.useNegSos.get():
                term1 = -sum_of_squares
            else:
                term1 = -sum_of_squares/(2*var)

            term2 = Npix/2 * np.log(2*np.pi*var)

            if self.useBothTerms.get():
                LL = term1 - term2
            else:
                LL = term1

            if self.printTerms.get():
                print('{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\n'.format(-sum_of_squares, term1, term2,
                                                                                      LL, var, 1/(2*var), var**0.5))

            newRow = md.Row()
            newRow.setValue(emlib.MDL_ITEM_ID, itemId)
            newRow.setValue(emlib.MDL_LL, float(LL))
            newRow.setValue(emlib.MDL_IMAGE_REF, fnVol)
            # newRow.setValue(emlib.MDL_RESIDUAL_VARIANCE, var)
            newRow.setValue(emlib.MDL_IMAGE_RESIDUAL, fnResidual)
            newRow.addToMd(mdOut)
        mdOut.write(self._getExtraPath("logLikelihood%03d.xmd"%i))

    def appendRows(self, outputSet, fnXmd):
        self.iterMd = md.iterRows(fnXmd, md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd)
        outputSet.copyItems(self.inputParticles.get(), updateItemCallback=self._processRow)

    def createOutputStep(self):
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(self.inputParticles.get())

        i=1
        if isinstance(self.inputRefs.get(), Volume):
            self.appendRows(outputSet, self._getExtraPath("logLikelihood%03d.xmd" % i))
            i += 1
        else:
            for _ in self.inputRefs.get():
                self.appendRows(outputSet, self._getExtraPath("logLikelihood%03d.xmd" % i))
                i += 1

        self._defineOutputs(reprojections=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

        matrix = np.array([particle._xmipp_logLikelihood.get() for particle in outputSet])
        matrix = matrix.reshape((i-1,-1))
        np.save(self._getExtraPath('matrix.npy'), matrix)

    def _getMdRow(self, mdFile, id):
        """ To get a row. Maybe there is way to request a specific row."""
        for row in md.iterRows(mdFile):
            if row.getValue(md.MDL_ITEM_ID) == id:
                return row

        raise Exception("Missing row %s at %s" % (id, mdFile))

    def _processRow(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_ITEM_ID):
            count += 1
            if count:
                particle.setObjId(None)
                setXmippAttributes(particle, self.lastRow,
                                   emlib.MDL_LL, emlib.MDL_IMAGE_REF)
                setXmippAttributes(particle, self.lastRow, emlib.MDL_IMAGE_RESIDUAL)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
        particle._appendItem = count > 0

    def _validate(self):
        errors = []

        if self.useGpu.get() and self.newProg.get():
            errors.append("You need to use the new program without GPU")

        return errors