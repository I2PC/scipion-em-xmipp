# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Laura del Cano (ldelcano@cnb.csic.es)
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
"""
This module contains utils functions for Xmipp protocols
"""

from os.path import exists, join
import subprocess
import numpy as np

import xmipp3
from pyworkflow import Config
from pwem import emlib
from .base import XmippMdRow

def validateXmippGpuBins():
    pass

def getMdFirstRow(filename):
    """ Create a MetaData but only read the first row.
    This method should be used for validations of labels
    or metadata size, but the full metadata is not needed.
    """
    md = emlib.MetaData()
    md.read(filename, 1)
    if md.getParsedLines():
        row = XmippMdRow()
        row.readFromMd(md, md.firstObject())
    else:
        row = None
    
    return row


def getMdSize(filename):
    """ Return the metadata size without parsing entirely. """
    md = emlib.MetaData()
    md.read(filename, 1)
    return md.getParsedLines()


def isMdEmpty(filename):
    """ Use getMdSize to check if metadata is empty. """
    return getMdSize(filename) == 0


def iterMdRows(md):
    """ Iterate over the rows of the given metadata. """
    # If md is string, take as filename and create the metadata
    if isinstance(md, str):
        md = emlib.MetaData(md)
        
    row = XmippMdRow()
    
    for objId in md:
        row.readFromMd(md, objId)
        yield row

def readInfoField(fnDir,block,label):
    mdInfo = emlib.MetaData("%s@%s"%(block,join(fnDir,"iterInfo.xmd")))
    return mdInfo.getValue(label, mdInfo.firstObject())

def writeInfoField(fnDir,block,label, value):
    mdInfo = emlib.MetaData()
    objId=mdInfo.addObject()
    mdInfo.setValue(label,value,objId)
    mdInfo.write("%s@%s"%(block,join(fnDir,"iterInfo.xmd")),emlib.MD_APPEND)
    
def validateDLtoolkit(errors=None, **kwargs):
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
    # initialize errors if needed
    errors = errors if errors is not None else []

    # Trying to import keras to assert if DeepLearningToolkit works fine.
    kerasError = False
    try:
        subprocess.check_output('python -c "import keras"', shell=True)
    except subprocess.CalledProcessError:
        errors.append("*Keras/Tensorflow not found*. Required to run this protocol.")
        kerasError=True

    # Asserting if the model exists only if the software is well installed
    modelError = False
    models = kwargs.get('model', '')
    if not kerasError and kwargs.get('assertModel', True) and models != '':
        models = models if isinstance(models, list) else [models]
        for model in models:
            if isinstance(model, str):
                if not exists(xmipp3.Plugin.getModel(model, doRaise=False)):
                    modelError = True
            elif isinstance(model, tuple):
                if not exists(xmipp3.Plugin.getModel(*model, doRaise=False)):
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

BAD_IMPORT_TENSORFLOW_KERAS_MSG='''
Error, tensorflow/keras is probably not installed. Install it with:\n  ./scipion installb deepLearningToolkit
If gpu version of tensorflow desired, install cuda 8.0 or cuda 9.0
We will try to automatically install cudnn, if unsucesfully, install cudnn and add to LD_LIBRARY_PATH
add to {0}
CUDA = True
CUDA_VERSION = 8.0 or 9.0
CUDA_HOME = /path/to/cuda-%(CUDA_VERSION)s
CUDA_BIN = %(CUDA_HOME)s/bin
CUDA_LIB = %(CUDA_HOME)s/lib64
CUDNN_VERSION = 6 or 7
'''.format(Config.SCIPION_CONFIG)

def copy_image(imag):
    ''' Return a copy of a xmipp_image
    '''
    new_imag = emlib.Image()
    new_imag.setData(imag.getData())
    return new_imag

def matmul_serie(mat_list, size=4):
    '''Return the matmul of several numpy arrays'''
    #Return the identity matrix if te list is empty
    if len(mat_list) > 0:
        res = np.identity(len(mat_list[0]))
        for i in range(len(mat_list)):
            res = np.matmul(res, mat_list[i])
    else:
        res = np.identity(size)
    return res

def normalize_array(ar):
    '''Normalize values in an array with mean 0 and std deviation 1
    '''
    ar -= np.mean(ar)
    ar /= np.std(ar)
    return ar

def surrounding_values(a,ii,jj,depth=1):
    '''Return a list with the surrounding elements, given the indexs of the center, from an 2D numpy array
    '''
    values=[]
    for i in range(ii-depth,ii+depth+1):
        for j in range(jj-depth,jj+depth+1):
            if i>=0 and j>=0 and i<a.shape[0] and j<a.shape[1]:
                if i!=ii or j!=jj:
                    values+=[a[i][j]]
    return values