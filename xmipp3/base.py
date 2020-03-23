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

import sys
from collections import OrderedDict

from pyworkflow.object import ObjectWrap
from pwem import emlib
import pwem

try:  # TODO: Avoid these imports by importing them in the protocols/viewers
    from xmipp_base import *  # xmipp_base and xmippViz come from the binding and
    from xmippViz import *    #  it is not available before installing the binaries
except:
    pass

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
            self._insertFunctionStep('convertInputToXmipp', inputName, xmippClass, resultFn)
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
            return [resultFn] # validate resultFn was produced if converted
         
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
        model = getXmippPath('models', *modelPath)

        # Raising an error to prevent posterior errors and to print a hint
        if kwargs.get('doRaise', True) and not os.path.exists(model):
            raise Exception("'%s' model not found. Please, run: \n"
                            " > scipion installb deepLearningToolkit" % modelPath[0])
        return model

    def validateDLtoolkit(self, errors=None, **kwargs):
        """ Validates if the deepLearningToolkit is installed.
            Additionally, it assert if a certain models is present when
            kwargs are present, following:
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
        if (hasattr(self, "_conda_env") ):
            condaEnvName= self._conda_env
        else:
            condaEnvName = None

        # initialize errors if needed
        errors = errors if errors is not None else []

        # Trying to import keras to assert if DeepLearningToolkit works fine.
        kerasError = False
        if condaEnvName is None:
            condaEnvName = CondaEnvManager.CONDA_DEFAULT_ENVIRON
        env = CondaEnvManager.modifyEnvToUseConda(xmipp3.Plugin.getEnviron(), condaEnvName)
        import subprocess
        try:
#            subprocess.call('which python', shell=True, env=env)
            subprocess.check_output('python -c "import keras"', shell=True, env=env)
        except subprocess.CalledProcessError as e:
            errors.append("*Keras/Tensorflow not found*. Required to run this protocol.")
            kerasError = True

        # Asserting if the model exists only if the software is well installed
        modelError = False
        models = kwargs.get('model', '')
        if not kerasError and kwargs.get('assertModel', True) and models != '':
            models = models if isinstance(models, list) else [models]
            for model in models:
                if isinstance(model, str):
                    if not os.path.exists(self.getModel(model, doRaise=False)):
                        modelError = True
                elif isinstance(model, tuple):
                    if not os.path.exists(self.getModel(*model, doRaise=False)):
                        modelError = True
            if modelError:
                errors.append("*Pre-trained model not found*. %s"
                              % kwargs.get('errorMsg', ''))

        # Hint to install the deepLearningToolkit
        if kerasError or modelError:
            errors.append("Please, *run* 'scipion installb deepLearningToolkit' "
                          "or install the scipion-em-xmipp/deepLearningToolkit "
                          "package using the *plugin manager*.")

        return errors

    def runCondaJob(self, program, arguments, **kwargs):
        '''
        Performs the same operation as self.runJob but preparing the environment to use conda instead.
        It will use the CONDA_DEFAULT_ENVIRON except when the class have defined the _conda_env argument
        :param program: string
        :param arguments: string
        :param kwargs: options
        :return:
        '''
        if "_conda_env" in kwargs:
            condaEnvName = kwargs.pop("_conda_env")
        elif (hasattr(self, "_conda_env") ):
            condaEnvName = self._conda_env
        else:
            condaEnvName = CondaEnvManager.CONDA_DEFAULT_ENVIRON
            print("Warning: using default conda environment '%s'. "
                  "CondaJobs should be run under a specific environment to "
                  "avoid problems. Please, fix it or contact to the developer."
                  % CondaEnvManager.CONDA_DEFAULT_ENVIRON)
        if "env" not in kwargs:
            kwargs['env'] = xmipp3.Plugin.getEnviron()
        program, arguments, kwargs = prepareRunConda(program, arguments, condaEnvName, **kwargs)

        super(type(self), self).runJob(program, arguments, **kwargs)


class XmippMdRow:
    """ Support Xmipp class to store label and value pairs 
    corresponding to a Metadata row. 
    """
    def __init__(self):
        self._labelDict = OrderedDict() # Dictionary containing labels and values
        self._objId = None # Set this id when reading from a metadata
        
    def getObjId(self):
        return self._objId
    
    def hasLabel(self, label):
        return self.containsLabel(label)
    
    def containsLabel(self, label):
        # Allow getValue using the label string
        if isinstance(label, str):
            label = emlib.str2Label(label)
        return label in self._labelDict
    
    def removeLabel(self, label):
        if self.hasLabel(label):
            del self._labelDict[label]
    
    def setValue(self, label, value):
        """args: this list should contains tuples with 
        MetaData Label and the desired value"""
        # Allow setValue using the label string
        if isinstance(label, str):
            label = emlib.str2Label(label)
        self._labelDict[label] = value
            
    def getValue(self, label, default=None):
        """ Return the value of the row for a given label. """
        # Allow getValue using the label string
        if isinstance(label, str):
            label = emlib.str2Label(label)
        return self._labelDict.get(label, default)
    
    def getValueAsObject(self, label, default=None):
        """ Same as getValue, but making an Object wrapping. """
        return ObjectWrap(self.getValue(label, default))
    
    def readFromMd(self, md, objId):
        """ Get all row values from a given id of a metadata. """
        self._labelDict.clear()
        self._objId = objId
        
        for label in md.getActiveLabels():
            self._labelDict[label] = md.getValue(label, objId)
            
    def writeToMd(self, md, objId):
        """ Set back row values to a metadata row. """
        for label, value in self._labelDict.items():
            # TODO: Check how to handle correctly unicode type
            # in Xmipp and Scipion
            t = type(value)
            
            if t is str:
                value = str(value)
                
            if t is int and emlib.labelType(label) == emlib.LABEL_SIZET:
                value = int(value)
                
            try:
                md.setValue(label, value, objId)
            except Exception as ex:
                print("XmippMdRow.writeToMd: Error writing value to metadata.",
                      file=sys.stderr)
                print("                     label: %s, value: %s, type(value): %s"
                      % (emlib.label2Str(label), value, type(value)), file=sys.stderr)
                raise ex
            
    def readFromFile(self, fn):
        md = emlib.MetaData(fn)
        self.readFromMd(md, md.firstObject())
        
    def copyFromRow(self, other):
        for label, value in other._labelDict.items():
            self.setValue(label, value)
            
    def __str__(self):
        s = '{'
        for k, v in self._labelDict.items():
            s += '  %s = %s\n' % (emlib.label2Str(k), v)
        return s + '}'
    
    def __iter__(self):
        return self._labelDict.items()
            
    def printDict(self):
        """ Fancy printing of the row, mainly for debugging. """
        print(str(self))
    
    
class RowMetaData:
    """ This class is a wrapper for MetaData in row mode.
    Where only one object is used.
    """
    def __init__(self, filename=None):
        self._md = emlib.MetaData()
        self._md.setColumnFormat(False)
        self._id = self._md.addObject()
        
        if filename:
            self.read(filename)
        
    def setValue(self, label, value):
        self._md.setValue(label, value, self._id)
        
    def getValue(self, label):
        return self._md.getValue(label, self._id)
        
    def write(self, filename, mode=emlib.MD_APPEND):
        self._md.write(filename, mode)
        
    def read(self, filename):
        self._md.read(filename)
        self._md.setColumnFormat(False)
        self._id = self._md.firstObject()
        
    def __str__(self):
        return str(self._md)
    
        
def findRow(md, label, value):
    """ Query the metadata for a row with label=value.
    Params:
        md: metadata to query.
        label: label to check value
        value: value for equal condition
    Returns:
        XmippMdRow object of the row found.
        None if no row is found with label=value
    """
    mdQuery = emlib.MetaData() # store result
    mdQuery.importObjects(md, emlib.MDValueEQ(label, value))
    n = mdQuery.size()
    
    if n == 0:
        row = None
    elif n == 1:
        row = XmippMdRow()
        row.readFromMd(mdQuery, mdQuery.firstObject())
    else:
        raise Exception("findRow: more than one row found matching the query %s = %s" % (emlib.label2Str(label), value))
    
    return row

def findRowById(md, value):
    """ Same as findRow, but using MDL_ITEM_ID for label. """
    return findRow(md, emlib.MDL_ITEM_ID, int(value))
  
  
class XmippSet:
    """ Support class to store sets in Xmipp base on a MetaData. """
    def __init__(self, itemClass):
        """ Create new set, base on a Metadata.
        itemClass: Class that represent the items.
        A method .getFileName should be available to store the md.
        Items contained in XmippSet are supposed to inherit from XmippMdRow.
        """
        self._itemClass = itemClass 
        #self._fileName = fileName      
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
        from pyworkflow.utils.path import cleanPath
        # Generate gallery of projections        
        fnGallery = self._getExtraPath('gallery.stk')
        if volume.endswith('.mrc'):
            volume+=":mrc"
        
        self.runJob("xmipp_angular_project_library", "-i %s -o %s --sampling_rate %f --sym %s --method fourier 1 0.25 bspline --compute_neighbors --angular_distance -1 --experimental_images %s"\
                   % (volume, fnGallery, angularSampling, symmetryGroup, images))
    
        # Assign angles
        self.runJob("xmipp_angular_projection_matching", "-i %s -o %s --ref %s --Ri 0 --Ro %s --max_shift 1000 --search5d_shift %s --search5d_step  %s --append"\
                   % (images, fnAngles, fnGallery, str(Xdim/2), str(int(Xdim/10)), str(int(Xdim/25))))
        
        cleanPath(self._getExtraPath('gallery_sampling.xmd'))
        cleanPath(self._getExtraPath('gallery_angles.doc'))
        cleanPath(self._getExtraPath('gallery.doc'))
    
        # Write angles in the original file and sort
        MD=emlib.MetaData(fnAngles)
        for id in MD:
            galleryReference = MD.getValue(emlib.MDL_REF,id)
            MD.setValue(emlib.MDL_IMAGE_REF, "%05d@%s" % (galleryReference+1,fnGallery), id)
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
            eulerMatrix = emlib.Euler_angles2matrix(0.,0.,psi)
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
                        cylinderOuterRadius, height, Ts):
        args="-i %s --sym %s --heightFraction %f -z %f %f %f --rotHelical %f %f %f --thr %d -o %s --sampling %f"%(fnVol,self.getSymmetry(dihedral),heightFraction,
                                                                                z0,zF,zStep,rot0,rotF,rotStep,Nthr,fnOut,Ts)
        if cylinderOuterRadius>0 and cylinderInnerRadius<0:
            args+=" --mask cylinder %d %d"%(-cylinderOuterRadius,-height)
        elif cylinderOuterRadius>0 and cylinderInnerRadius>0:
            args+=" --mask tube %d %d %d"%(-cylinderInnerRadius,-cylinderOuterRadius,-height)
        self.runJob('xmipp_volume_find_symmetry',args, numberOfMpi=1)

    def runFineSearch(self, fnVol, dihedral, fnCoarse, fnFine, heightFraction, z0, zF, rot0, rotF, cylinderInnerRadius, cylinderOuterRadius, height, Ts):
        md=emlib.MetaData(fnCoarse)
        objId=md.firstObject()
        rotInit=md.getValue(emlib.MDL_ANGLE_ROT,objId)
        zInit=md.getValue(emlib.MDL_SHIFT_Z,objId)
        args="-i %s --sym %s --heightFraction %f --localHelical %f %f -o %s -z %f %f 1 --rotHelical %f %f 1 --sampling %f"%(fnVol,self.getSymmetry(dihedral),heightFraction,
                                                                                           zInit,rotInit,fnFine,z0,zF,rot0,rotF,Ts)
        if cylinderOuterRadius>0 and cylinderInnerRadius<0:
            args+=" --mask cylinder %d %d"%(-cylinderOuterRadius,-height)
        elif cylinderOuterRadius>0 and cylinderInnerRadius>0:
            args+=" --mask tube %d %d %d"%(-cylinderInnerRadius,-cylinderOuterRadius,-height)
        self.runJob('xmipp_volume_find_symmetry',args, numberOfMpi=1)

    def runSymmetrize(self, fnVol, dihedral, fnParams, fnOut, heightFraction, cylinderInnerRadius, cylinderOuterRadius, height, Ts):
        md=emlib.MetaData(fnParams)
        objId=md.firstObject()
        rot0=md.getValue(emlib.MDL_ANGLE_ROT,objId)
        z0=md.getValue(emlib.MDL_SHIFT_Z,objId)
        args="-i %s --sym %s --helixParams %f %f --heightFraction %f -o %s --sampling %f --dont_wrap"%(fnVol,self.getSymmetry(dihedral),z0,rot0,heightFraction,fnOut,Ts)
        self.runJob('xmipp_transform_symmetrize',args,numberOfMpi=1)
        doMask=False
        if cylinderOuterRadius>0 and cylinderInnerRadius<0:
            args="-i %s --mask cylinder %d %d"%(fnVol,-cylinderOuterRadius,-height)
            doMask=True
        elif cylinderOuterRadius>0 and cylinderInnerRadius>0:
            args="-i %s --mask tube %d %d %d"%(fnVol,-cylinderInnerRadius,-cylinderOuterRadius,-height)
            doMask=True
        if doMask:
            self.runJob('xmipp_transform_mask',args,numberOfMpi=1)

