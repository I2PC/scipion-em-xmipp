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
This module contains utils functions to operate over xmipp metadata files.
"""

from os.path import exists

import xmipp3
import xmippLib
from .base import XmippMdRow

def validateXmippGpuBins():
    pass

def getMdFirstRow(filename):
    """ Create a MetaData but only read the first row.
    This method should be used for validations of labels
    or metadata size, but the full metadata is not needed.
    """
    md = xmippLib.MetaData()
    md.read(filename, 1)
    if md.getParsedLines():
        row = XmippMdRow()
        row.readFromMd(md, md.firstObject())
    else:
        row = None
    
    return row


def getMdSize(filename):
    """ Return the metadata size without parsing entirely. """
    md = xmippLib.MetaData()
    md.read(filename, 1)
    return md.getParsedLines()


def isMdEmpty(filename):
    """ Use getMdSize to check if metadata is empty. """
    return getMdSize(filename) == 0


def iterMdRows(md):
    """ Iterate over the rows of the given metadata. """
    # If md is string, take as filename and create the metadata
    if isinstance(md, basestring):
        md = xmippLib.MetaData(md)
        
    row = XmippMdRow()
    
    for objId in md:
        row.readFromMd(md, objId)
        yield row


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
        import keras
    except:
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