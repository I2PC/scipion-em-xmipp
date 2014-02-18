# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (jgomez@cnb.csic.es)
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
# *  e-mail address 'jgomez@cnb.csic.es'
# *
# **************************************************************************
"""
This sub-package contains wrapper around kendersom Xmipp program
"""

from os.path import join, dirname, exists
from pyworkflow.em import *  
import xmipp

import xmipp3
from convert import createXmippInputImages, readSetOfClasses2D
from glob import glob


class KendersomBaseClassify(ProtClassify):
    """ Class to create a base template for Kendersom and rotational spectra protocols that share
    a common structure. 
    """
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputImages', PointerParam, label="Input images", important=True, 
                      pointerClass='SetOfParticles', pointerCondition='hasAlignment',
                      help='Select the input images from the project.'
                           'It should be a SetOfParticles class')
        form.addParam('useMask', BooleanParam, default=False,
                      label='Use a Mask ?', 
                      help='If you set to *Yes*, you should provide a mask')
        form.addParam('Mask', StringParam , condition='useMask',
                      label="Mask",
                      help='Mask image will serve to enhance the classification')
        form.addParam('SomXdim', IntParam, default=7,
                      label='X-dimension of the map:')
        form.addParam('SomYdim', IntParam, default=7,
                      label='Y-dimension of the map:')
        form.addParam('SomReg0', IntParam, default=1000, expertLevel=LEVEL_ADVANCED,
                      label='Initial regularization factor', 
                      help='The kerdenSOM algorithm anneals from an initial high regularization factor'
                      'to a final lower one, in a user-defined number of steps.'
                      'If the output map is too smooth, lower the regularization factors'
                      'If the output map is not organized, higher the regularization factors'
                      'See [[http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/KerDenSOM][KerDenSOM]]')
        form.addParam('SomReg1', IntParam, default=200, expertLevel=LEVEL_ADVANCED,
                      label='Final regularization factor:')
        form.addParam('SomSteps', IntParam, default=5, expertLevel=LEVEL_ADVANCED,
                      label='Regularization steps:',
                      help='Number of steps to lower the regularization factor')
        form.addParam('extraParams', StringParam, default='', expertLevel=LEVEL_ADVANCED,
                      label="Additional parameters:", 
                      help='Additional parameters for kerdensom program. \n For a complete description'
                      'See [[http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/KerDenSOM][KerDenSOM]]')
        self._addParams(form)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _prepareParams(self):
        # Convert input images if necessary
        imgsFn = createXmippInputImages(self, self.inputImages.get())
        
        mask = self.Mask.get()
        self._params = {'oroot': self._getExtraPath("kerdensom"),
                        'imgsFn': imgsFn,
                        'mask': mask,
                        'SomXdim': self.SomXdim.get(),
                        'SomYdim': self.SomYdim.get(),
                        'SomReg0': self.SomReg0.get(),
                        'SomReg1': self.SomReg1.get(),
                        'SomSteps': self.SomSteps.get(),
                        'extraParams': self.extraParams.get(),
                        'vectors': self._getExtraPath("vectors.xmd"),
                        'classes': self._getExtraPath("classes.stk"),
                        'kvectors': self._getExtraPath("kerdensom_vectors.xmd"),
                        'kclasses': self._getExtraPath("kerdensom_classes.xmd")
                       }
    
    def _insertAllSteps(self):
        self._prepareParams()
        self._insertProccesStep()
        self._insertFunctionStep('createOutputStep')
    
    def _insertKerdensomStep(self):
        args = '-i %(vectors)s --oroot %(oroot)s --xdim %(SomXdim)d --ydim %(SomYdim)d' + \
               ' --deterministic_annealing %(SomSteps)f %(SomReg0)f %(SomReg1)f %(extraParams)s'
        self._insertRunJobStep("xmipp_classify_kerdensom", args % self._params)
#        deleteFiles([self._getExtraPath("vectors.xmd"),self._getExtraPath("vectors.vec")], True)
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def createOutputStep(self):
        """ Store the kenserdom object 
        as result of the protocol.
        """
        imgSet = self.inputImages.get()
        classes2DSet = self._createSetOfClasses2D(self.inputImages.get())
        readSetOfClasses2D(classes2DSet, self._params['kclasses'])
        self._defineOutputs(outputClasses=classes2DSet)
        self._defineSourceRelation(imgSet, classes2DSet)
    
    #--------------------------- INFO functions ----------------------------------------------------
    def _validate(self):
        errors = []
        mask = self.Mask.get()
        if self.SomReg0 < self.SomReg1:
            errors.append("Regularization must decrease over iterations:")
            errors.append("    Initial regularization must be larger than final")
        if self.useMask:
            if len(mask) > 0:
                if not exists(mask):
                    errors.append("Cannot find the file " + mask)
            else:
                errors.append("Please, enter a mask file")
        return errors
    
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputClasses'):
            summary.append("Output classification not ready yet.")
        else:
            summary.append("Input Images: %s" % self.inputImages.get().getNameId())
            summary.append("Output Classified Images: %s" % self.outputClasses)
        return summary
    
    def _methods(self):
        messages = []      
        messages.append("*Kendersom classification*")
        messages.append('The particles %s were classified to obtain the classes %s.' % (self.inputImages.get().getNameId(), self.outputClasses.getNameId()))
        return messages


class XmippProtKerdensom(KendersomBaseClassify):
    """
    Perform Kernel Probability Density Estimator Self-Organizing Map.
    
    The kerdenSOM algorithm anneals from an initial high regularization factor
    to a final lower one, in a user-defined number of steps.
    
    KerdenSOM is an excellent tool for classification, especially when
    using a large number of data and classes and when the transition between
    the classes is almost continuous, with no clear separation between them.
    
    The input images must be previously aligned.
    """
    _label = 'kerdensom'
    
    def __init__(self, **args):
        KendersomBaseClassify.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _addParams(self, form):
        pass
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertProccesStep(self):
        self._insertImgToVectorStep()
        self._insertKerdensomStep()
        self._insertVectorToImgStep()
        self._insertFunctionStep('rewriteClassBlockStep')
    
    def _insertImgToVectorStep(self):
        """ Insert runJob for convert into a vector Md """
        args = ' -i %(imgsFn)s -o %(vectors)s '
        if self.useMask:
            args += ' --mask binary_file %(mask)s'
        
        self._insertRunJobStep("xmipp_image_vectorize", args % self._params)
   
    def _insertVectorToImgStep(self):
        args = ' -i %(kvectors)s -o %(classes)s' 
        if self.useMask:
            args += ' --mask binary_file %(mask)s'
        self._insertRunJobStep("xmipp_image_vectorize", args % self._params)
#        deleteFiles([self._getExtraPath("kerdensom_vectors.xmd"),self._getExtraPath("kerdensom_vectors.vec")], True)
    
    #--------------------------- INFO functions ----------------------------------------------------
    def rewriteClassBlockStep(self):
        fnClass = "classes@%(kclasses)s" % self._params
        fnClassStack = self._params['classes']
        md = xmipp.MetaData(fnClass)
        # TODO: Check if following is necessary
        counter = 1
        for objId in md:
            md.setValue(xmipp.MDL_IMAGE,"%06d@%s"%(counter,fnClassStack), objId)
            counter += 1
        md.write(fnClass, xmipp.MD_APPEND)
    
    #--------------------------- INFO functions ----------------------------------------------------
    def _citations(self):
        return ['PascualMontano2001', 'PascualMontano2002']
