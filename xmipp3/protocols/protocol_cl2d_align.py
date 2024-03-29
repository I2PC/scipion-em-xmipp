# ******************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
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
# ******************************************************************************w
import pyworkflow.protocol.params as params

import pwem.emlib.metadata as md
from pwem.objects import Particle, SetOfAverages, SetOfClasses2D
from pwem.protocols import ProtAlign2D
from pwem import ALIGN_NONE, ALIGN_2D

from xmipp3.convert import createItemMatrix, writeSetOfParticles, getImageLocation


class XmippProtCL2DAlign(ProtAlign2D):
    """ Aligns a set of particles using the CL2D algorithm. """
    _label = 'align with cl2d'
    
    # --------------------------- DEFINE param functions -----------------------
    def _defineAlignParams(self, form):
        form.addParam('useReferenceImage', params.BooleanParam, default=False,
                      label='Use a Reference Image ?',
                      help='If you set to *Yes*, you should provide a '
                           'reference image.\n'
                           'If *No*, the default generation is done by '
                           'averaging subsets of the input images.')
        form.addParam('referenceImage', params.PointerParam,
                      condition='useReferenceImage',
                      pointerClass='Particle, SetOfAverages, SetOfClasses2D', allowsNull=True,
                      label="Reference image",
                      help='Image that will serve as class reference. If the input is a set, then the first image '
                           'will be used as reference.')
        form.addParam('maximumShift', params.IntParam, default=10,
                      label='Maximum shift (px):')
        form.addParam('numberOfIterations', params.IntParam, default=10,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Number of iterations:',
                      help='Maximum number of iterations')
        form.addParallelSection(threads=0, mpi=4)
    
    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """ Mainly prepare the command line for call cl2d align program"""
        
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('images.xmd')
        self._insertFunctionStep('convertInputStep')
        
        # Prepare arguments to call program: xmipp_classify_CL2D
        self._params = {'imgsFn': self.imgsFn,
                        'extraDir': self._getExtraPath(),
                        'maxshift': self.maximumShift.get(),
                        'iter': self.numberOfIterations.get(),
                        }
        args = ('-i %(imgsFn)s --odir %(extraDir)s --nref 1 --iter %(iter)d '
                '--maxShift %(maxshift)d')
        
        if self.useReferenceImage:
            if isinstance(self.referenceImage.get(),Particle):
                args += " --ref0 " + getImageLocation(self.referenceImage.get())
            elif isinstance(self.referenceImage.get(), SetOfClasses2D):
                args += " --ref0 " + getImageLocation(self.referenceImage.get().getFirstItem().getRepresentative())
            else:
                args += " --ref0 " + getImageLocation(self.referenceImage.get().getFirstItem())
        else:
            args += " --nref0 1"
        self._insertRunJobStep("xmipp_classify_CL2D", args % self._params)
        
        self._insertFunctionStep('createOutputStep')
    
    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(), self.imgsFn,
                            alignType=ALIGN_NONE)
    
    def createOutputStep(self):
        """ Store the setOfParticles object
        as result of the protocol.
        """
        particles = self.inputParticles.get()
        # Define the output average
        avgFile = self._getExtraPath('level_00', 'class_classes.stk')
        avg = Particle()
        avg.setLocation(1, avgFile)
        avg.copyInfo(particles)
        self._defineOutputs(outputAverage=avg)
        self._defineSourceRelation(self.inputParticles, avg)
        
        # Generate the Set of Particles with alignment
        alignedSet = self._createSetOfParticles()
        alignedSet.copyInfo(particles)
        alignedSet.setRepresentative(avg)
        alignedSet.copyItems(particles,
                             updateItemCallback=self._createItemMatrix,
                             itemDataIterator=md.iterRows(self.imgsFn,
                                                          sortByLabel=md.MDL_ITEM_ID))
        alignedSet.setAlignment(ALIGN_2D)
        self._defineOutputs(outputParticles=alignedSet)
        self._defineSourceRelation(self.inputParticles, alignedSet)
    
    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        if self.numberOfMpi <= 1:
            errors.append('Mpi needs to be greater than 1.')
        if self.useReferenceImage:
            if self.referenceImage.hasValue():
                refImage = self.referenceImage.get()
                if isinstance(refImage,Particle):
                    [x1, y1, z1] = refImage.getDim()
                else:
                    [x1, y1, z1] = refImage.getFirstItem().getDim()
                [x2, y2, z2] = self.inputParticles.get().getDim()
                if x1 != x2 or y1 != y2 or z1 != z2:
                    errors.append('The input images and the reference image '
                                  'have different sizes')
            else:
                errors.append("Please, enter a reference image")
        return errors
    
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output alignment not ready yet.")
        else:
            summary.append("Input Particles: %s"
                           % self.inputParticles.get().getSize())
            if self.useReferenceImage:
                summary.append("Aligned with reference image: %s"
                               % self.referenceImage.get().getNameId())
            else:
                summary.append("Aligned with no reference image.")
        return summary
    
    def _citations(self):
        return ['Sorzano2010a']
    
    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output alignment not ready yet.")
        else:
            if self.useReferenceImage:
                methods.append(
                    "We aligned images %s with respect to the reference image "
                    "%s using CL2D [Sorzano2010a]"
                    % (self.getObjectTag('inputParticles'),
                       self.getObjectTag('referenceImage')))
            else:
                methods.append(
                    "We aligned images %s with no reference using CL2D "
                    "[Sorzano2010a]" % self.getObjectTag('inputParticles'))
            methods.append(" and produced %s images."
                           % self.getObjectTag('outputParticles'))
        return methods
    
    def _createItemMatrix(self, item, row):
        createItemMatrix(item, row, align=ALIGN_2D)
