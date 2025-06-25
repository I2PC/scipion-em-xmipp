# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin [1]
# *              Adrian Quintana (aquintana@cnb.csic.es)
# *            
# * [1] SciLifeLab, Stockholm University
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

import os
import subprocess

from pyworkflow.protocol import Protocol
from pyworkflow.utils.path import cleanPath
from pwem import emlib
import pwem
from .constants import XMIPP_DLTK_NAME

try:  # If binding is not already done, this will fail.
    from xmipp_base import *  # xmipp_base and xmippViz come from the binding and
    from xmippViz import *    #  it is not available before installing the binaries
except Exception as exc:  # TODO: catch exception by type and ensure it is caused by CondaEnvManager
    error = exc
    class CondaEnvManager:
        """ Fake class to avoid very early fails (during installation). """
        @classmethod
        def yieldInstallAllCmds(*args, **kwargs):
            yield ('echo "Some error occurred when importing xmipp_base '
                   '(from Xmipp binding): %s"' % error, 'void.target')

import xmipp3


LABEL_TYPES = { 
               emlib.LABEL_SIZET: int,
               emlib.LABEL_DOUBLE: float,
               emlib.LABEL_INT: int,
               emlib.LABEL_BOOL: bool
               }


def getXmippPath(*paths):
    '''Return the path the the Xmipp installation folder
    if a subfolder is provided, will be concatenated to the path'''
    return os.path.join(pwem.Config.XMIPP_HOME, *paths)

def isXmippCudaPresent(program=""):
    if program=="":
        return os.path.isfile(getXmippPath("bin","xmipp_cuda_reconstruct_fourier"))
    else:
        return os.path.isfile(getXmippPath("bin",program))

def getLabelPythonType(label):
    """ From xmipp label to python variable type """
    labelType = emlib.labelType(label)
    return LABEL_TYPES.get(labelType, str)


class XmippProtocol:
    """ This class groups some common functionalities that
    share some Xmipp protocols, like converting steps.
    """
           
    def _insertConvertStep(self, inputName, xmippClass, resultFn):
        """ Insert the convertInputToXmipp if the inputName attribute
        is not an instance of xmippClass.
        It will return the result filename, if the
        conversion is needed, this will be input resultFn.
        If not, it will be inputAttr.getFileName()
        """
        inputAttr = getattr(self, inputName)
        if not isinstance(inputAttr, xmippClass):
            self._insertFunctionStep('convertInputToXmipp', inputName, xmippClass,
                                     resultFn)
            return resultFn
        return inputAttr.getFileName()
         
    def convertInputToXmipp(self, inputName, xmippClass, resultFn):
        """ This step can be used whenever a conversion is needed.
        It will receive the inputName and get this attribute from self,
        invoke the convert function and check the result files if
        conversion was done (otherwise the input was already in Xmipp format).
        """
        inputAttr = getattr(self, inputName)
        inputXmipp = xmippClass.convert(inputAttr, resultFn)
         
        if inputXmipp is not inputAttr:
            self._insertChild(inputName + 'Xmipp', inputXmipp)
            return [resultFn]  # validate resultFn was produced if converted
         
    def getConvertedInput(self, inputName):
        """ Retrieve the converted input, it can be the case that
        it is the same as input, when not conversion was done.
        """
        return getattr(self, inputName + 'Xmipp', getattr(self, inputName))

    @classmethod
    def getModel(cls, *modelPath, **kwargs):
        """ Returns the path to the models folder followed by
            the given relative path.
        .../xmipp/models/myModel/myFile.h5 <= getModel('myModel', 'myFile.h5')

            NOTE: it raise and exception when model not found, set doRaise=False
                  in the arguments to skip that raise, especially in validation
                  asserions!
        """
        os.environ['XMIPP_HOME'] = getXmippPath()
        return getModel(*modelPath, **kwargs)

    def validateDLtoolkit(self, errors=None, **kwargs):
        """ Validates if the deepLearningToolkit is installed.
            Additionally, it assert if a certain models is present when
            kwargs are present, following:
              - _conda_env: The conda env to be load
                            (default: the protocol._conda_env or CONDA_DEFAULT_ENVIRON)
              - assertModel: if models should be evaluated or not (default: True).
              - errorMsg: a custom message error (default: '').
              - model: a certain model name/route/list (default: no model assert)
                + model='myModel': asserts if myModel exists
                + model=('myModel', 'myFile.h5'): asserts is myModel/myFile.h5 exists
                + model=['myModel1', 'myModel2', ('myModel3', 'myFile3.h5')]: a combination

            usage (3 examples):
              errors = validateDLtoolkit(errors, doAssert=self.useModel.get(),
                                         model="myModel")

              errors = validateDLtoolkit(model=("myModel2", "myFile.h5"))

              errors = validateDLtoolkit(doAssert=self.mode.get()==PREDICT,
                                         model=("myModel3", "subFolder", "model.h5"),
                                         errorMsg="myModel3 is required for the "
                                                  "prediction mode")
        """
        # initialize errors if needed
        errors = errors if errors is not None else []

        # Looking for the installation target for that conda environment.
        kerasError = False
        envName = CondaEnvManager.getCondaName(self, **kwargs)
        targetName = CondaEnvManager.getCondaEnvTargetFilename(self, envName)
        if not os.path.isfile(os.path.join(pwem.Config.EM_ROOT,
                                           XMIPP_DLTK_NAME,
                                           targetName)):
            errors.append("*%s environment not found*. "
                          "Required to run this protocol." % envName)
            kerasError = True

        # Asserting if the model exists only if the software is well installed
        models = kwargs.get('model', '')
        failedModels = []
        if not kerasError and kwargs.get('assertModel', True) and models != '':
            models = models if isinstance(models, list) else [models]
            for model in models:
                if isinstance(model, str):
                    modelFn = self.getModel(model, doRaise=False)
                    modelName = model.split('/')[0]  # This differs from dirname
                elif isinstance(model, tuple):
                    modelFn = self.getModel(*model, doRaise=False)
                    modelName = model[0]
                else:
                    print("Deep Learning model type unknown (%s validation):\n"
                          " > %s %s" % (self, model, type(model)))
                    continue
                if not os.path.exists(modelFn):
                    failedModels.append(modelName)
            if failedModels:
                errors.append("*Pre-trained model(s) not found*: %s"
                              % ', '.join(failedModels))

        # Hint to install the deepLearningToolkit
        if kerasError or failedModels:
            errors.append("Please, *run* 'scipion installb deepLearningToolkit' "
                          "or install the scipion-em-xmipp > deepLearningToolkit "
                          "package using the *plugin manager*.")
        return errors

    @classmethod
    def getCondaEnv(cls, **kwargs):
        """ Returns the environ corresponding to the condaEnv of the protocol.
            kwargs can contain:
               - 'env': Any environ (the xmipp one is the default)
               - '_conda_env': An installed condaEnv name
            > _conda_env preference: kwargs > protocol default > general default
        """
        envName = CondaEnvManager.getCondaName(cls, **kwargs)
        env = kwargs.get('env', xmipp3.Plugin.getEnviron())
        return CondaEnvManager.getCondaEnv(env, envName)

# findRow() cannot go to xmipp_base (binding) because depends on emlib.metadata.Row()
def findRow(md, label, value):
    """ Query the metadata for a row with label=value.
    Params:
        md: metadata to query.
        label: label to check value
        value: value for equal condition
    Returns:
        Row object of the row found.
        None if no row is found with label=value
    """
    mdQuery = emlib.MetaData()  # store result
    mdQuery.importObjects(md, emlib.MDValueEQ(label, value))
    n = mdQuery.size()

    if n == 0:
        row = None
    elif n == 1:
        row = emlib.metadata.Row()
        row.readFromMd(mdQuery, mdQuery.firstObject())
    else:
        raise Exception("findRow: more than one row found matching the query "
                        "%s = %s" % (emlib.label2Str(label), value))

    return row


def findRowById(md, value):
    """ Same as findRow, but using MDL_ITEM_ID for label. """
    return findRow(md, emlib.MDL_ITEM_ID, int(value))


# getMdFirstRow() cannot go to xmipp_base (binding) because depends on emlib.metadata.Row()
def getMdFirstRow(filename):
    """ Create a MetaData but only read the first row.
    This method should be used for validations of labels
    or metadata size, but the full metadata is not needed.
    """
    md = emlib.MetaData()
    md.read(filename, 1)
    if md.getParsedLines():
        row = emlib.metadata.Row()
        row.readFromMd(md, md.firstObject())
    else:
        row = None

    return row

# iterMdRows() cannot go to xmipp_base (binding) because depends on emlib.metadata.Row()
def iterMdRows(md):
    """ Iterate over the rows of the given metadata. """
    # If md is string, take as filename and create the metadata
    if isinstance(md, str):
        md = emlib.MetaData(md)

    row = emlib.metadata.Row()

    for objId in md:
        row.readFromMd(md, objId)
        yield row
  
  
class XmippSet:
    # FIXME: It seems unused...
    """ Support class to store sets in Xmipp base on a MetaData. """
    def __init__(self, itemClass):
        """ Create new set, base on a Metadata.
        itemClass: Class that represent the items.
        A method .getFileName should be available to store the md.
        Items contained in XmippSet are supposed to inherit from Row.
        """
        self._itemClass = itemClass 
        # self._fileName = fileName
        self._md = emlib.MetaData()
        
        
    def __iter__(self):
        """Iterate over the set of images in the MetaData"""
        #self._md.read(self._fileName)
        
        for objId in self._md:  
            item = self._itemClass()
            item.readFromMd(self._md, objId)  
            #m = emib.Image(md.getValue(emlib.MDL_IMAGE, objId))
            #if self.hasCTF():
            #    m.ctfModel = XmippCTFModel(md.getValue(emlib.MDL_CTF_MODEL, objId))
            yield item
        
#        
#    def setFileName(self, filename):
#        self._fileName = filename
        
    def setMd(self, md):
        self._md = md
        
    def write(self, filename, mode):
        self._md.write(filename, mode)
        
    def append(self, item):
        """Add a new item to the set, the item can be of a base class (EM)
        and will be try to convert it to the respective _itemClass."""
        objId = self._md.addObject()
        # Convert to xmipp micrograph if necessary
        if isinstance(item, self._itemClass):
            itemXmipp = item
        else:
            itemXmipp = self._itemClass.convert(item)
        itemXmipp.writeToMd(self._md, objId)
        
    def sort(self, label):
        self._md.sort(label)
        
    def read(self, filename):
        self._md.read(filename)
        
    def isEmpty(self):
        return self._md.isEmpty()
    
    def getItemClass(self):
        """ Return the Class of the items in the Set. """
        return self._itemClass
    
    def getSize(self):
        return self._md.size()
        
    def convert(self, xmippSetClass, filename):
        """ Convert from a generic set to a xmippSetClass.
        In particular a filename is requiered to store the result MetaData.
        It is also asummed that this class have a .copyInfo method.
        """
        if isinstance(self, xmippSetClass):
            return self
        
        setOut = xmippSetClass(filename)
        setOut.copyInfo(self)
        
        for item in self:
            setOut.append(item)
        setOut.write()
        
        return setOut   
    
    
class ProjMatcher:
    """ Base class for protocols that use a projection """
    
    def projMatchStep(self, volume, angularSampling, symmetryGroup, images, fnAngles, Xdim):
        # Generate gallery of projections        
        fnGallery = self._getExtraPath('gallery.stk')
        if volume.endswith('.mrc'):
            volume+=":mrc"
        
        self.runJob("xmipp_angular_project_library",
                    "-i %s -o %s --sampling_rate %f --sym %s --method fourier 1 0.25 bspline "
                    "--compute_neighbors --angular_distance -1 --experimental_images %s"
                    % (volume, fnGallery, angularSampling, symmetryGroup, images))
    
        # Assign angles
        self.runJob("xmipp_angular_projection_matching",
                    "-i %s -o %s --ref %s --Ri 0 --Ro %s --max_shift 1000 "
                    "--search5d_shift %s --search5d_step  %s --append"
                    % (images, fnAngles, fnGallery, str(Xdim/2),
                       str(int(Xdim/10)), str(int(Xdim/25))))
        
        cleanPath(self._getExtraPath('gallery_sampling.xmd'))
        cleanPath(self._getExtraPath('gallery_angles.doc'))
        cleanPath(self._getExtraPath('gallery.doc'))
    
        # Write angles in the original file and sort
        MD=emlib.MetaData(fnAngles)
        for id in MD:
            galleryReference = MD.getValue(emlib.MDL_REF,id)
            MD.setValue(emlib.MDL_IMAGE_REF, "%05d@%s" % (galleryReference+1, fnGallery), id)
        MD.write(fnAngles)
        
    def produceAlignedImagesStep(self, volumeIsCTFCorrected, fn, images):
        
        from numpy import array, dot
        fnOut = 'classes_aligned@' + fn
        MDin = emlib.MetaData(images)
        MDout = emlib.MetaData()
        n = 1
        hasCTF = MDin.containsLabel(emlib.MDL_CTF_MODEL)
        for i in MDin:
            fnImg = MDin.getValue(emlib.MDL_IMAGE,i)
            fnImgRef = MDin.getValue(emlib.MDL_IMAGE_REF,i)
            maxCC = MDin.getValue(emlib.MDL_MAXCC,i)
            rot =  MDin.getValue(emlib.MDL_ANGLE_ROT,i)
            tilt = MDin.getValue(emlib.MDL_ANGLE_TILT,i)
            psi =-1.*MDin.getValue(emlib.MDL_ANGLE_PSI,i)
            flip = MDin.getValue(emlib.MDL_FLIP,i)
            if flip:
                psi = -psi
            eulerMatrix = emlib.Euler_angles2matrix(0., 0., psi)
            x = MDin.getValue(emlib.MDL_SHIFT_X,i)
            y = MDin.getValue(emlib.MDL_SHIFT_Y,i)
            shift = array([x, y, 0])
            shiftOut = dot(eulerMatrix, shift)
            [x,y,z]= shiftOut
            if flip:
                x = -x
            id = MDout.addObject()
            MDout.setValue(emlib.MDL_IMAGE, fnImg, id)
            MDout.setValue(emlib.MDL_IMAGE_REF, fnImgRef, id)
            MDout.setValue(emlib.MDL_IMAGE1, "%05d@%s"%(n, self._getExtraPath("diff.stk")), id)
            if hasCTF:
                fnCTF = MDin.getValue(emlib.MDL_CTF_MODEL,i)
                MDout.setValue(emlib.MDL_CTF_MODEL,fnCTF,id)
            MDout.setValue(emlib.MDL_MAXCC, maxCC, id)
            MDout.setValue(emlib.MDL_ANGLE_ROT, rot, id)
            MDout.setValue(emlib.MDL_ANGLE_TILT, tilt, id)
            MDout.setValue(emlib.MDL_ANGLE_PSI, psi, id)
            MDout.setValue(emlib.MDL_SHIFT_X, x,id)
            MDout.setValue(emlib.MDL_SHIFT_Y, y,id)
            MDout.setValue(emlib.MDL_FLIP,flip,id)
            MDout.setValue(emlib.MDL_ENABLED,1,id)
            n+=1
        MDout.write(fnOut,emlib.MD_APPEND)
        
        # Actually create the differences
        img = emlib.Image()
        imgRef = emlib.Image()
        if hasCTF and volumeIsCTFCorrected:
            Ts = MDin.getValue(emlib.MDL_SAMPLINGRATE, MDin.firstObject())
    
        for i in MDout:
            img.readApplyGeo(MDout,i)
            imgRef.read(MDout.getValue(emlib.MDL_IMAGE_REF,i))
            if hasCTF and volumeIsCTFCorrected:
                fnCTF = MDout.getValue(emlib.MDL_CTF_MODEL,i)
                emlib.applyCTF(imgRef, fnCTF, Ts)
                img.convert2DataType(emlib.DT_DOUBLE)
            imgDiff = img-imgRef
            imgDiff.write(MDout.getValue(emlib.MDL_IMAGE1,i))


class HelicalFinder:
    """ Base class for protocols that find helical symmetry """

    def getSymmetry(self,dihedral):
        if dihedral:
            return "helicalDihedral"
        else:
            return "helical"

    def runCoarseSearch(self, fnVol, dihedral, heightFraction, z0, zF, zStep,
                        rot0, rotF, rotStep, Nthr, fnOut, cylinderInnerRadius,
                        cylinderOuterRadius, height, Ts, Cn="c1"):
        args = ("-i %s --sym %s --heightFraction %f -z %f %f %f "
                "--rotHelical %f %f %f --thr %d -o %s --sampling %f"
                %(fnVol, self.getSymmetry(dihedral), heightFraction, z0, zF, zStep,
                  rot0, rotF, rotStep, Nthr, fnOut, Ts))
        if Cn!="c1":
            args += " --sym2 %s"%Cn
        if cylinderOuterRadius > 0 and cylinderInnerRadius < 0:
            args += " --mask cylinder %d %d"%(-cylinderOuterRadius, -height)
        elif cylinderOuterRadius > 0 and cylinderInnerRadius > 0:
            args += " --mask tube %d %d %d" % (-cylinderInnerRadius, -cylinderOuterRadius, -height)
        self.runJob('xmipp_volume_find_symmetry', args, numberOfMpi=1)

    def runFineSearch(self, fnVol, dihedral, fnCoarse, fnFine, heightFraction,
                      z0, zF, rot0, rotF, cylinderInnerRadius, cylinderOuterRadius,
                      height, Ts, Cn="c1"):
        md=emlib.MetaData(fnCoarse)
        objId=md.firstObject()
        rotInit=md.getValue(emlib.MDL_ANGLE_ROT,objId)
        zInit=md.getValue(emlib.MDL_SHIFT_Z,objId)
        args=("-i %s --sym %s --heightFraction %f --localHelical %f %f -o %s "
              "-z %f %f 1 --rotHelical %f %f 1 --sampling %f"
              %(fnVol, self.getSymmetry(dihedral), heightFraction, zInit, rotInit,
                fnFine, z0, zF, rot0, rotF, Ts))
        if Cn!="c1":
            args += " --sym2 %s"%Cn
        if cylinderOuterRadius>0 and cylinderInnerRadius<0:
            args+=" --mask cylinder %d %d"%(-cylinderOuterRadius,-height)
        elif cylinderOuterRadius>0 and cylinderInnerRadius>0:
            args+=" --mask tube %d %d %d"%(-cylinderInnerRadius,-cylinderOuterRadius,-height)
        self.runJob('xmipp_volume_find_symmetry',args, numberOfMpi=1)

    def runSymmetrize(self, fnVol, dihedral, fnParams, fnOut, heightFraction,
                      cylinderInnerRadius, cylinderOuterRadius, height, Ts, Cn="c1"):
        md=emlib.MetaData(fnParams)
        objId=md.firstObject()
        rot0=md.getValue(emlib.MDL_ANGLE_ROT,objId)
        z0=md.getValue(emlib.MDL_SHIFT_Z,objId)
        args=("-i %s --sym %s --helixParams %f %f --heightFraction %f -o %s "
              "--sampling %f --dont_wrap"
              % (fnVol,self.getSymmetry(dihedral),z0,rot0,heightFraction,fnOut,Ts))
        if Cn!="c1":
            args += " --sym2 %s"%Cn
        self.runJob('xmipp_transform_symmetrize',args,numberOfMpi=1)
        doMask=False
        if cylinderOuterRadius>0 and cylinderInnerRadius<0:
            args="-i %s --mask cylinder %d %d"%(fnVol,-cylinderOuterRadius,-height)
            doMask=True
        elif cylinderOuterRadius>0 and cylinderInnerRadius>0:
            args="-i %s --mask tube %d %d %d"%(fnVol,-cylinderInnerRadius,
                                               -cylinderOuterRadius,-height)
            doMask=True
        if doMask:
            self.runJob('xmipp_transform_mask',args,numberOfMpi=1)

