# **************************************************************************
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
# **************************************************************************
from pyworkflow.protocol import (PointerParam, BooleanParam, IntParam,
                                 LEVEL_ADVANCED, StringParam)

from pwem.emlib.image import ImageHandler
from pwem.objects import Particle
from pwem.protocols import ProtClassify2D


from pwem import emlib


from xmipp3.convert import (writeSetOfParticles, readSetOfClasses2D,
                            xmippToLocation)


class KendersomBaseClassify(ProtClassify2D):
    """ Class to create a base template for Kendersom and rotational spectra
    protocols that share a common structure.

    AI Generated

    ## Overview

    The Kerdensom protocol performs 2D classification of a set of aligned particle
    images using a self-organizing map approach.

    The method is based on Kohonen self-organizing maps combined with fuzzy
    clustering ideas. Instead of producing only a small number of independent and
    clearly separated classes, Kerdensom organizes the classes on a two-dimensional
    map. Neighboring classes in this map are expected to represent similar image
    patterns.

    This makes the protocol especially useful when the dataset contains gradual or
    continuous variability. For example, particles may differ by small changes in
    orientation, conformation, flexibility, occupancy, or image quality. In such
    cases, the transition between classes may be smooth rather than sharply
    separated.

    The input particles must already be aligned. Kerdensom is not intended to solve
    the initial alignment problem. Its purpose is to organize already aligned images
    into a structured set of representative 2D classes.

    ## Inputs and General Workflow

    The main input is a set of aligned particles. The protocol converts these
    images into a vector representation, optionally applying a mask to focus the
    classification on the relevant region of the particle.

    The Kerdensom algorithm then classifies these vectors into a two-dimensional
    self-organizing map. The size of this map is defined by the user through the X
    and Y dimensions. Each position in the map corresponds to one class.

    After the classification, the protocol converts the resulting class
    representatives back into images and computes an average image for each class.
    The final output is a Scipion set of 2D classes.

    ## Input Images

    The **Input images** parameter should point to a SetOfParticles with alignment
    information.

    This requirement is important. Since the method compares image intensities
    across particles, the particles should already be in a common orientation or
    reference frame. If the particles are not aligned, the classification may
    separate images mainly by rotations or shifts rather than by meaningful
    biological or structural differences.

    Typical inputs include particles after a 2D alignment step, particles extracted
    from a homogeneous subset, or images that have already been centered and
    oriented by a previous protocol.

    ## Use of a Mask

    The protocol can optionally use a **mask** during vectorization and
    classification.

    A mask restricts the comparison to the region of the image that is most
    relevant for classification. This is useful when the particle occupies only
    part of the box or when the background contains noise, carbon edges,
    neighboring particles, contaminants, or other features that should not drive
    the classification.

    Using a mask can help the self-organizing map focus on the biological signal.
    However, the mask should be chosen carefully. If it is too tight, it may remove
    important parts of the particle. If it is too broad, background noise may still
    influence the classification.

    For most particle datasets, a soft mask around the particle region is usually
    preferable to a very sharp or overly restrictive mask.

    ## Dimension of the Map

    The **Dimension of the map** defines the size of the self-organizing map. The
    two parameters, X and Y, determine the number of class positions.

    For example, a 7 by 7 map produces 49 class positions.

    This does not only define the number of classes. It also defines their
    organization. Classes that are close to each other on the map should correspond
    to similar particle appearances, while classes that are far apart should
    represent more different image patterns.

    A small map gives a compact summary of the dataset, but may merge distinct
    states or views. A large map gives a more detailed description of variability,
    but may produce classes with fewer particles and noisier averages.

    The best map size depends on the number of particles, the heterogeneity of the
    dataset, and the goal of the analysis. For exploratory analysis, a moderate map
    size is usually a good starting point.

    ## Regularization Factors

    Kerdensom uses a deterministic annealing strategy controlled by two
    regularization factors:

    - the **Initial regularization factor**;
    - the **Final regularization factor**.

    The algorithm starts with a high regularization value and gradually decreases
    it. At the beginning, stronger regularization encourages a smoother and more
    organized map. As the regularization decreases, the classes are allowed to
    adapt more closely to the data.

    The initial regularization factor must be larger than the final regularization
    factor. This is checked by the protocol.

    If the resulting map is too smooth, different classes may look too similar. In
    that case, lower regularization values may help the classes adapt more strongly
    to the data.

    If the resulting map is poorly organized, with neighboring classes not showing
    a clear relationship, higher regularization values may help preserve the
    self-organizing structure.

    These parameters are advanced options. In routine use, the default values are a
    reasonable starting point.

    ## Regularization Steps

    The **Regularization steps** parameter controls how many steps are used to
    decrease the regularization factor from its initial value to its final value.

    More steps provide a more gradual annealing process, which may help the map
    organize more smoothly. Fewer steps make the transition faster and may reduce
    computation time.

    In most cases, the default value should be sufficient. Advanced users may
    increase the number of steps when working with complex datasets or when the map
    organization appears unstable.

    ## Additional Parameters

    The **Additional parameters** field allows advanced users to pass extra options
    directly to the underlying Xmipp Kerdensom program.

    This option is intended for users who already know the command-line behavior of
    the underlying method. Most users should leave this field empty.

    Changing additional parameters without understanding their effect may make the
    classification harder to interpret or less reproducible.

    ## Class Representatives and Class Averages

    After classification, the protocol produces class representatives and class
    averages.

    The representative images describe the positions of the self-organizing map.
    The class averages are computed from the experimental particles assigned to
    each class.

    If a class has particles assigned to it, the protocol computes the average of
    those particles. If a class is empty, the protocol creates an empty average
    image with the correct dimensions.

    The class average is often the most useful image for biological
    interpretation, because it shows the average experimental signal of particles
    assigned to that class.

    ## Output Classes

    The main output is a **SetOfClasses2D**.

    Each class contains the particles assigned to that position in the
    self-organizing map, together with an average image. The number of possible
    classes is determined by the X and Y dimensions of the map, although some
    classes may be empty if no particles are assigned to them.

    The output classes can be inspected in Scipion to evaluate the organization of
    the map, the quality of the averages, and the distribution of particles across
    classes.

    A well-organized Kerdensom result often shows gradual transitions between
    neighboring classes. This can be very useful for identifying continuous
    structural variability or for selecting subsets of particles corresponding to
    particular appearances.

    ## Interpreting the Self-Organizing Map

    The two-dimensional layout of the output should be interpreted as part of the
    result.

    Neighboring classes are expected to be similar. For example, one region of the
    map may contain particles with one appearance, while another region may contain
    a different appearance, with intermediate classes forming a gradual transition.

    This makes Kerdensom different from ordinary classification methods where class
    numbers are often arbitrary and unordered. Here, the spatial organization of
    the classes is meaningful.

    However, the map should not be overinterpreted. The position of a class in the
    map is a data-driven organization, not a direct physical coordinate. It should
    be used as a guide to explore variability, not as definitive proof of a
    continuous biological pathway.

    ## Practical Recommendations

    Use Kerdensom with particles that have already been aligned. If the particles
    are not aligned, run an alignment or 2D classification protocol first.

    Start with a moderate map size, such as the default 7 by 7 map. Increase the
    map size if the dataset is large and contains rich variability. Decrease it if
    the dataset is small or if many classes become empty or too noisy.

    Use a mask when the particle occupies a well-defined region and the background
    may interfere with classification. Make sure the mask includes the relevant
    particle density.

    Keep the default regularization values at first. If the map is too smooth,
    reduce the regularization factors. If the map is poorly organized, increase
    them.

    Inspect both the class averages and the map layout. The most informative
    result is not only whether individual classes look good, but whether
    neighboring classes show a coherent organization.

    Be cautious with very small classes. They may represent rare states, but they
    may also reflect noise, contaminants, or unstable classification.

    ## Final Perspective

    Kerdensom is a 2D classification protocol designed to organize aligned
    particles into a structured self-organizing map.

    For biological users, its main value is exploratory. It can reveal gradual
    changes in particle appearance, help identify heterogeneous subsets, and
    provide a visual organization of the dataset that is richer than a simple list
    of independent classes.

    The protocol is most useful when the particles are already reasonably aligned
    and when the user wants to study continuous or subtle variability rather than
    only obtain a few sharply separated 2D classes.
    """
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label="Input images", important=True, 
                      pointerClass='SetOfParticles', pointerCondition='hasAlignment',
                      help='Select the input images from the project.'
                           'It should be a SetOfParticles class')
        form.addParam('useMask', BooleanParam, default=False,
                      label='Use a Mask ?', 
                      help='If you set to *Yes*, you should provide a mask')
        form.addParam('Mask', PointerParam , condition='useMask',
                      label="Mask", pointerClass='Mask',
                      help='Mask image will serve to enhance the classification')
        
        line = form.addLine('Dimension of the map', 
                            help='Josue tiene que meter el help')
        line.addParam('SomXdim', IntParam, default=7,
                      label='X')
        line.addParam('SomYdim', IntParam, default=7,
                      label='Y')
               
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
        self.imgsFn = self._getExtraPath('images.xmd') 
        self._insertFunctionStep('convertInputStep')  
        
        if self.useMask:
            mask = self.Mask.get().getFileName()
        else:
            mask = None
            
        self._params = {'oroot': self._getExtraPath("kerdensom"),
                        'imgsFn': self.imgsFn,
                        'mask': mask,
                        'SomXdim': self.SomXdim.get(),
                        'SomYdim': self.SomYdim.get(),
                        'SomReg0': self.SomReg0.get(),
                        'SomReg1': self.SomReg1.get(),
                        'SomSteps': self.SomSteps.get(),
                        'extraParams': self.extraParams.get(),
                        'vectors': self._getExtraPath("vectors.xmd"),
                        'classes': self._getExtraPath("classes.stk"),
                        'averages': self._getExtraPath("averages.stk"),
                        'kvectors': self._getExtraPath("kerdensom_vectors.xmd"),
                        'kclasses': self._getExtraPath("kerdensom_classes.xmd")
                       }
    
    def _insertAllSteps(self):
        self._prepareParams()
        self._insertProccesStep()
        self._insertFunctionStep('rewriteClassBlockStep')
        self._insertFunctionStep('createOutputStep')
    
    def _insertKerdensomStep(self):
        args = '-i %(vectors)s --oroot %(oroot)s --xdim %(SomXdim)d --ydim %(SomYdim)d' + \
               ' --deterministic_annealing %(SomSteps)f %(SomReg0)f %(SomReg1)f %(extraParams)s'
        self._insertRunJobStep("xmipp_classify_kerdensom", args % self._params)
#        deleteFiles([self._getExtraPath("vectors.xmd"),self._getExtraPath("vectors.vec")], True)
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(),self.imgsFn) 

    #--------------------------- INFO functions ----------------------------------------------------
    def rewriteClassBlockStep(self):
        firstImage = self.inputParticles.get().getFirstItem()
        fnClasses = self._params['kclasses']
        mdClasses = "classes@%s" % fnClasses
        fnClassStack = self._params['classes']
        fnAverageStack = self._params['averages']      
        
        md = emlib.MetaData(mdClasses)
        image = ImageHandler().createImage()
        
        counter = 1
        
        for objId in md:
            imageName =  "%06d@%s" % (counter, fnClassStack)
            averageName = "%06d@%s" % (counter, fnAverageStack)
            
            if md.getValue(emlib.MDL_CLASS_COUNT, objId) > 0:
                # compute the average of images assigned to this class
                classPrefix = 'class%06d' % counter
                classMd = '%s_images@%s' % (classPrefix, fnClasses)
                classRoot = self._getTmpPath(classPrefix)
                self.runJob('xmipp_image_statistics', 
                            '-i %s --save_image_stats %s -v 0' % (classMd, classRoot))
                image.read(classRoot + 'average.xmp')
            else:
                # Create emtpy image as average
                image.read(firstImage.getLocation()) # just to take the right dimensions and datatype
                image.initConstant(0.)
                
            image.write(averageName)
            md.setValue(emlib.MDL_IMAGE, imageName, objId)
            md.setValue(emlib.MDL_IMAGE2, averageName, objId)
            
            counter += 1
            
        md.write(mdClasses, emlib.MD_APPEND)
        
    def _preprocessClass(self, classItem, classRow):
        classItem.average = Particle()
        classItem.average.setLocation(xmippToLocation(classRow.getValue(emlib.MDL_IMAGE2)))
        
    def createOutputStep(self):
        """ Store the kenserdom object 
        as result of the protocol.
        """
        imgSet = self.inputParticles.get()
        classes2DSet = self._createSetOfClasses2D(imgSet)
        readSetOfClasses2D(classes2DSet, self._params['kclasses'], 
                           preprocessClass=self._preprocessClass)
        self._defineOutputs(outputClasses=classes2DSet)
        self._defineSourceRelation(self.inputParticles, classes2DSet)
    
    #--------------------------- INFO functions ----------------------------------------------------
    def _validate(self):
        errors = []
        if self.SomReg0 < self.SomReg1:
            errors.append("Regularization must decrease over iterations:")
            errors.append("    Initial regularization must be larger than final")
        if self.useMask:
            mask = self.Mask.get()
            if mask is None:
                errors.append("You have selected to use a mask. Select one.")
        return errors
    
    def _summary(self):
        return self._methods()

    def _methods(self):
        messages = []  
        if not hasattr(self, 'outputClasses'):
            messages.append("Output classification not ready yet.")
        elif self.inputParticles.get() is None:
            messages.append('Input not selected yet.')
        else:    
            messages.append("*Kendersom classification*")
            messages.append('%s particles from %s were classified to obtain %s classes %s.'
                            % (self.inputParticles.get().getSize(), self.getObjectTag('inputParticles'), self.outputClasses.getSize(), self.getObjectTag('outputClasses')))
            if self.useMask:
                messages.append('Mask %s was used in classification.' % self.getObjectTag('Mask'))
        return messages


class XmippProtKerdensom(KendersomBaseClassify):
    """
    Classifies a set of images using  Kohonen's Self-Organizing Feature Maps (SOM) 
    and Fuzzy c-means clustering technique (FCM) .
    
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
    def _validate(self):
        return KendersomBaseClassify._validate(self)
    
    def _summary(self):
        return KendersomBaseClassify._summary(self)
    
    def _methods(self):
        return KendersomBaseClassify._methods(self)
    
    def _citations(self):
        return ['PascualMontano2001', 'PascualMontano2002']
