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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
This module contains converter functions that will serve to:
1. Write from base classes to Xmipp specific files
2. Read from Xmipp files to base classes
"""

import xmipp
from data import *
from xmipp3 import XmippMdRow
from pyworkflow.em.constants import NO_INDEX

LABEL_TYPES = { 
               xmipp.LABEL_SIZET: long,
               xmipp.LABEL_DOUBLE: float,
               xmipp.LABEL_INT: int,
               xmipp.LABEL_BOOL: bool              
               }

def objectToRow(obj, row, attrDict):
    """ This function will convert an EMObject into a XmippMdRow.
    Params:
        obj: the EMObject instance (input)
        row: the XmippMdRow instance (output)
        attrDict: dictionary with the map between obj attributes(keys) and 
            row MDLabels in Xmipp (values).
    """
    for attr, label in attrDict.iteritems():
        if hasattr(obj, attr):
            labelType = xmipp.labelType(label)
            valueType = LABEL_TYPES.get(labelType, str)
            row.setValue(label, valueType(getattr(obj, attr).get()))

def _rowToObject(row, obj, attrDict):
    """ This function will convert from a XmippMdRow to an EMObject.
    Params:
        row: the XmippMdRow instance (input)
        obj: the EMObject instance (output)
        attrDict: dictionary with the map between obj attributes(keys) and 
            row MDLabels in Xmipp (values).
    """
    for attr, label in attrDict.iteritems():
        if not hasattr(obj, attr):
            setattr(obj, attr, String()) #TODO: change string for the type of label
        getattr(obj, attr).set(row.getValue(label))
    
def rowToObject(md, objId, obj, attrDict):
    """ Same as rowToObject, but creating the row from md and objId. """
    row = XmippMdRow()
    row.readFromMd(md, objId)
    _rowToObject(row, obj, attrDict)
    
def readCTFModel(filename):
    """ Read from Xmipp .ctfparam and create a CTFModel object. """
    md = xmipp.MetaData(filename)
    ctfObj = CTFModel()    
    ctfDict = { 
               "defocusU": xmipp.MDL_CTF_DEFOCUSU,
               "defocusV": xmipp.MDL_CTF_DEFOCUSV,
               "defocusAngle": xmipp.MDL_CTF_DEFOCUS_ANGLE,
               "sphericalAberration": xmipp.MDL_CTF_CS
               }    
    rowToObject(md, md.firstObject(), ctfObj, ctfDict)  
    ctfObj._xmippMd = String(filename) 
    
    return ctfObj


def writeCTFModel(ctfObj, filename):
    """ Write a CTFModel object as Xmipp .ctfparam"""
    pass


def readMicrograph(md, objId):
    """ Create a Micrograph object from a row of Xmipp metadata. """
    pass

def locationToXmipp(index, filename):
    """ Convert an index and filename location
    to a string with @ as expected in Xmipp.
    """
    #TODO: Maybe we need to add more logic dependent of the format
    if index != NO_INDEX:
        return "%d@%s" % (index, filename)
    
    return filename

def micrographToRow(mic, micRow):
    """ Set labels values from Micrograph mic to md row. """
    micDict = { 
               "_id": xmipp.MDL_ITEM_ID,
#               "defocusV": xmipp.MDL_CTF_DEFOCUSV,
#               "defocusAngle": xmipp.MDL_CTF_DEFOCUS_ANGLE,
#               "sphericalAberration": xmipp.MDL_CTF_CS
               }
    index, filename = mic.getLocation()
    fn = locationToXmipp(index, filename)
    micRow.setValue(xmipp.MDL_MICROGRAPH, fn)
      
    objectToRow(mic, micRow, micDict)

def rowToCoordinate(md, objId):
    """ Create a Coordinate from a row of a metadata. """
    coordDict = { 
               "_x": xmipp.MDL_XCOOR,
               "_y": xmipp.MDL_YCOOR,
#               "sphericalAberration": xmipp.MDL_CTF_CS
               }
    coord = Coordinate()
    rowToObject(md, objId, coord, coordDict)
    
    return coord

def readSetOfMicrographs(filename):
    pass


def writeSetOfMicrographs(micSet, filename, rowFunc=None):
    """ This function will write a SetOfMicrographs as Xmipp metadata.
    Params:
        micSet: the SetOfMicrograph instance.
        filename: the filename where to write the metadata.
        rowFunc: this function can be used to setup the row before 
            adding to metadata.
    """
    md = xmipp.MetaData()
    
    for mic in micSet:
        objId = md.addObject()
        micRow = XmippMdRow()
        micrographToRow(mic, micRow)
        if rowFunc:
            rowFunc(mic, micRow)
        micRow.writeToMd(md, objId)
        
    md.write(filename)
    micSet._xmippMd = String(filename)
    
    
def readPosCoordinates(posFile):
    """ Read the coordinates in .pos file. 
    and return corresponding metadata. 
    """
    md = xmipp.MetaData(posFile)
    blocks = xmipp.getBlocksInMetaDataFile(posFile)
    
    for b in ['particles', 'particles_auto']:
        if b in blocks:
            mdAux = xmipp.MetaData('%(b)s@%(posFile)s' % locals())
            md.unionAll(mdAux)
    
    return md
            
def readSetOfCoordinates(posDir, micSet, coordSet):
    """ Read from Xmipp .pos files.
    Params:
        posDir: the directory where the .pos files are.
            It is also expected a file named: config.xmd
            in this directory where the box size can be read.
        micSet: the SetOfMicrographs to associate the .pos, which 
            name should be the same of the micrographs.
        coordSet: the SetOfCoordinates that will be populated.
    """
    # Read the boxSize from the config.xmd metadata
    md = xmipp.MetaData('properties@' + join(posDir, 'config.xmd'))
    boxSize = md.getValue(xmipp.MDL_PICKING_PARTICLE_SIZE, md.firstObject())
    
    for mic in micSet:
        posFile = join(posDir, replaceBaseExt(mic.getFileName(), 'pos'))
        posMd = readPosCoordinates(posFile)
        
        for objId in posMd:
            coord = rowToCoordinate(posMd, objId)
            coord.setMicrograph(mic)
            coordSet.append(coord)

    coordSet.setBoxSize(boxSize)
    
