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

try:
    from itertools import izip
except ImportError:
    izip = zip

import numpy as np
import os

from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume, SetOfParticles, SetOfClasses3D
from pwem import emlib
import pwem.emlib.metadata as md

from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, EnumParam, FloatParam,
                                        BooleanParam, IntParam, 
                                        USE_GPU, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL


from xmipp3.convert import setXmippAttributes, writeSetOfParticles
import xmippLib
from pyworkflow import BETA, UPDATED, NEW, PROD



class XmippProtComputeLikelihood(ProtAnalysis3D):
    """This protocol computes the log likelihood or correlation of a set of particles with assigned angles when compared to a set of maps or atomic models"""

    _label = 'log likelihood'
    _lastUpdateVersion = VERSION_1_1
    _possibleOutputs = {"reprojections": SetOfParticles}
    _devStatus = PROD
    stepsExecutionMode = STEPS_PARALLEL

    # Normalization enum constants
    NORM_OLD = 0
    NORM_NEW = 1
    NORM_RAMP =2

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self._classesInfo = dict()

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addParam('binThreads', IntParam,
                      label='threads',
                      default=2,
                      help='Number of threads used by Xmipp each time it is called in the protocol execution. For '
                           'example, if 3 Scipion threads and 3 Xmipp threads are set, the particles will be '
                           'processed in groups of 2 at the same time with a call of Xmipp with 3 threads each, so '
                           '6 threads will be used at the same time. Beware the memory of your machine has '
                           'memory enough to load together the number of particles specified by Scipion threads.')
        
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

        form.addParam('optimizeGray', BooleanParam, label="Optimize gray: ", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Optimize the gray value between the map reprojection and the experimental image')
        form.addParam('maxGrayChange', FloatParam, label="Max. gray change: ", default=0.99, expertLevel=LEVEL_ADVANCED,
                      condition='optimizeGray',
                      help='The actual gray value can be at most as small as 1-change or as large as 1+change')

        form.addParam('doNorm', BooleanParam, default=False,
                      label='Normalize', expertLevel=LEVEL_ADVANCED,
                      help='Whether to subtract background gray values and normalize'
                           'so that in the background there is 0 mean and'
                           'standard deviation 1. This is applied to particles and volumes')
        form.addParam('normType', EnumParam, condition='doNorm',
                      label='Particle normalization type', expertLevel=LEVEL_ADVANCED,
                      choices=['OldXmipp','NewXmipp','Ramp'],
                      default=self.NORM_RAMP, display=EnumParam.DISPLAY_COMBO,
                      help='OldXmipp: mean(Image)=0, stddev(Image)=1\n'
                           'NewXmipp: mean(background)=0, stddev(background)=1\n'
                           'Ramp: subtract background + NewXmipp\n'
                           'Only New and Old Xmipp are supported for volumes.'
                           'If Ramp is selected then New is used for volumes')

        form.addParam('ignoreCTF', BooleanParam, label="Do not apply CTF: ", default=False, expertLevel=LEVEL_ADVANCED,
                      help='This should be used when images are treated with a Weiner filter instead')

        form.addParam('printTerms', BooleanParam, label="Print terms of LL: ", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Whether to print terms 1 and 2, LL and noise variance')

        form.addParallelSection(threads=2, mpi=2)
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """ Convert input images then run continuous_assign2 then create output """
        
        self.imagesXmd = self._getExtraPath("images.xmd")
        self.imagesStk = self._getExtraPath("images.stk")

        convId = self._insertFunctionStep(self.convertStep, prerequisites=[], needsGPU=False)
        stepIds = [convId]
        if self.doNorm:
            normPsId = self._insertFunctionStep(self.normalizeParticlesStep, prerequisites=convId,
                                                needsGPU=False)
            stepIds.append(normPsId)

        inputRefs = self.inputRefs.get()
        i=1
        if isinstance(inputRefs, Volume):
            prodId = self._insertFunctionStep(self.produceResidualsStep, inputRefs.getFileName(), i,
                                              prerequisites=stepIds, needsGPU=False)
            i += 1
            stepIds.append(prodId)
        else:
            for volume in inputRefs:
                prodId = self._insertFunctionStep(self.produceResidualsStep, volume.getFileName(), i,
                                                  prerequisites=stepIds, needsGPU=False)
                i += 1
                stepIds.append(prodId)

        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=stepIds,
                                 needsGPU=False)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        imgSet = self.inputParticles.get()
        writeSetOfParticles(imgSet, self.imagesXmd)

    def normalizeParticlesStep(self):
        argsNorm = self._argsNormalize(particles=True)
        self.runJob("xmipp_transform_normalize", argsNorm)

    def getMasks(self):
        if not (hasattr(self, 'mask') and hasattr(self, 'noiseMask')):
            Xdim = self._getSize()
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

        if self.doNorm:
            fnVolOut = self._getExtraPath('%03d_' % (i) + os.path.split(fnVol)[1])
            argsNorm = "-i %s -o %s" % (fnVol, fnVolOut) + self._argsNormalize()
            self.runJob("xmipp_transform_normalize", argsNorm)
            fnVol = fnVolOut

        if self.newProg:
            anglesOutFn = self._getExtraPath("anglesCont%03d.xmd"%i)
            prog = "xmipp_continuous_create_residuals"
        else:
            anglesOutFn = self._getExtraPath("anglesCont%03d.stk"%i)
            prog = "xmipp_angular_continuous_assign2"

        fnResiduals = self._getExtraPath("residuals%03d.stk"%i)
        fnProjections = self._getExtraPath("projections%03d.stk"%i)

        Ts = self.inputParticles.get().getSamplingRate()
        args = "-i %s -o %s --ref %s --sampling %f --oresiduals %s --oprojections %s" % (self.imagesXmd, anglesOutFn,
                                                                                         fnVol, Ts,
                                                                                         fnResiduals, fnProjections)

        if self.optimizeGray:
            args+=" --optimizeGray --max_gray_scale %f"%self.maxGrayChange

        self.runJob(prog, args, numberOfMpi=self.numberOfMpi.get())

        mdResults = md.MetaData(self._getExtraPath("anglesCont%03d.xmd"%i))
        mdOut = md.MetaData()

        if self.printTerms.get():
                print('{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\t{:9s}\n'.format('-sos', 'term1', 'term2',
                                                                                 'LL', 'var', '1/(2*var)', 'std'))
        for objId in mdResults:
            itemId = mdResults.getValue(emlib.MDL_ITEM_ID, objId)

            if self.optimizeGray:
                fnResidual = mdResults.getValue(emlib.MDL_IMAGE_RESIDUAL, objId)
                I = xmippLib.Image(fnResidual)

                elements_within_circle = I.getData()[self.getMasks()[0]]
                sum_of_squares = np.sum(elements_within_circle**2)
                Npix = elements_within_circle.size

                elements_between_circles = I.getData()[self.getMasks()[1]]
                var = np.var(elements_between_circles)

                term1 = -sum_of_squares/(2*var)
                term2 = Npix/2 * np.log(2*np.pi*var)
                LL = term1 - term2

                if self.printTerms.get():
                    print('{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\t{:9.2e}\n'.format(-sum_of_squares, term1, term2,
                                                                                                    LL, var, 1/(2*var), var**0.5))

            else:
                LL = mdResults.getValue(emlib.MDL_COST, objId)

            newRow = md.Row()
            newRow.setValue(emlib.MDL_ITEM_ID, itemId)
            newRow.setValue(emlib.MDL_LL, float(LL))
            newRow.setValue(emlib.MDL_IMAGE_REF, fnVol)

            if self.optimizeGray:
                newRow.setValue(emlib.MDL_IMAGE_RESIDUAL, fnResidual)

            newRow.addToMd(mdOut)
        mdOut.write(self._getLLfilename(i))

    def appendRows(self, outputSet, fnXmd):
        self.iterMd = md.iterRows(fnXmd, md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd)
        outputSet.copyItems(self.inputParticles.get(), updateItemCallback=self._processRow)

    def createOutputStep(self):
        inputPartSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputPartSet)

        refsDict = {}
        i=1
        if isinstance(self.inputRefs.get(), Volume):
            self.appendRows(outputSet, self._getLLfilename(i))
            refsDict[i] = self.inputRefs.get()
            i += 1
        else:
            for volume in self.inputRefs.get():
                self.appendRows(outputSet, self._getLLfilename(i))
                refsDict[i] = volume.clone()
                i += 1

        self._defineOutputs(reprojections=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

        matrix = np.array([particle._xmipp_logLikelihood.get() for particle in outputSet])
        matrix = matrix.reshape((i-1,-1))
        np.save(self._getExtraPath('matrix.npy'), matrix)

        classIds = np.argmax(matrix, axis=0)+1

        clsSet = SetOfClasses3D.create(self._getExtraPath())
        clsSet.setImages(inputPartSet)

        clsDict = {}  # Dictionary to store the (classId, classSet) pairs

        for ref, rep in refsDict.items():
            # add empty classes
            classItem = clsSet.ITEM_TYPE.create(self._getExtraPath(), suffix=ref+1)
            classItem.setRepresentative(rep)
            clsDict[ref] = classItem
            clsSet.append(classItem)

        cls_prev = 1
        for img, ref in izip(inputPartSet, classIds):
            if ref != cls_prev:
                cls_prev = ref

            classItem = clsDict[ref]
            classItem.append(img)

        for classItem in clsDict.values():
            clsSet.update(classItem)

        clsSet.write()

        self._defineOutputs(outputClasses=clsSet)
        self._defineSourceRelation(self.inputParticles, clsSet)
        self._defineSourceRelation(self.inputRefs, clsSet)

    def _getMdRow(self, mdFile, id):
        """ To get a row. Maybe there is way to request a specific row."""
        for row in md.iterRows(mdFile):
            if row.getValue(md.MDL_ITEM_ID) == id:
                return row

        raise ValueError("Missing row %s at %s" % (id, mdFile))

    def _processRow(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_ITEM_ID):
            count += 1
            if count:
                particle.setObjId(None)
                setXmippAttributes(particle, self.lastRow,
                                   emlib.MDL_LL, emlib.MDL_IMAGE_REF,
                                   emlib.MDL_IMAGE_RESIDUAL)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
        particle._appendItem = count > 0

    def _argsNormalize(self, particles=False):
        args = ""
        if particles:
            args += "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                   % (self.imagesXmd, self.imagesStk, self.imagesXmd)

        normType = self.normType.get()
        bgRadius = self.particleRadius.get()
        radii = self._getSize()
        if bgRadius <= 0:
            bgRadius = int(radii)

        if normType == self.NORM_OLD:
            args += " --method OldXmipp"
        elif normType == self.NORM_NEW or not particles:
            args += " --method NewXmipp --background circle %d" % bgRadius
        else:
            args += " --method Ramp --background circle %d" % bgRadius
        return args

    def _getSize(self):
        return self.inputParticles.get().getDimensions()[0]

    def _getLLfilename(self, i):
        return self._getExtraPath("logLikelihood%03d.xmd" % i)
