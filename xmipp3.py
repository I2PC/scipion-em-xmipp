# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
This sub-package will contains Xmipp3.0 specific protocols
"""

import os
import xmipp

from constants import *
import pyworkflow.dataset as ds

def getXmippPath(*paths):
    '''Return the path the the Xmipp installation folder
    if a subfolder is provided, will be concatenated to the path'''
    if os.environ.has_key('XMIPP_HOME'):
        return os.path.join(os.environ['XMIPP_HOME'], *paths)  
    else:
        raise Exception('XMIPP_HOME environment variable not set')
    
    
class XmippProtocol():
    """ This class groups some common functionalities that
    share some Xmipp protocols, like converting steps.
    """
    
    def _insertConvertStep(self, inputName, xmippClass, resultFn):
        """ Insert the convertInputToXmipp if the inputName attribute
        is not an instance of xmippClass.
        It will return the result filename, if the 
        convertion is needed, this will be input resultFn.
        If not, it will be inputAttr.getFileName()
        """
        inputAttr = getattr(self, inputName)
        print "inputAttr.getClassName()", inputAttr.getClassName()
        print "xmippClass", xmippClass
        if not isinstance(inputAttr, xmippClass):
            self._insertFunctionStep('convertInputToXmipp', inputName, xmippClass, resultFn)
            return resultFn
        return inputAttr.getFileName()
        
    def convertInputToXmipp(self, inputName, xmippClass, resultFn):
        """ This step can be used whenever a convertion is needed.
        It will receive the inputName and get this attribute from self,
        invoke the convert function and check the result files if
        convertion was done (otherwise the input was already in Xmipp format).
        """
        inputAttr = getattr(self, inputName)
        inputXmipp = xmippClass.convert(inputAttr, resultFn)
        
        if inputXmipp is not inputAttr:
            print "======== CONVERTIN........."
            self._insertChild(inputName + 'Xmipp', inputXmipp)
            return [resultFn] # validate resultFn was produced if converted
        
    def getConvertedInput(self, inputName):
        """ Retrieve the converted input, it can be the case that
        it is the same as input, when not convertion was done. 
        """
        return getattr(self, inputName + 'Xmipp', getattr(self, inputName))
        

class XmippMdRow():
    """ Support Xmipp class to store label and value pairs 
    corresponding to a Metadata row. It can be used as base
    for classes that maps to a MetaData row like XmippImage, XmippMicrograph..etc. 
    """
    def __init__(self):
        self._labelDict = {} # Dictionary containing labels and values
    
    def hasLabel(self, label):
        return label in self._labelDict
    
    def setValue(self, label, value):
        """args: this list should contains tuples with 
        MetaData Label and the desired value"""
        self._labelDict[label] = value
            
    def getValue(self, label):
        return self._labelDict[label]
    
    def getFromMd(self, md, objId):
        """ Get all row values from a given id of a metadata. """
        self._labelDict.clear()
        for label in md.getActiveLabels():
            self._labelDict[label] = md.getValue(label, objId)
            
    def setToMd(self, md, objId):
        """ Set back row values to a metadata row. """
        for label, value in self._labelDict.iteritems():
            # TODO: Check how to handle correctly unicode type
            # in Xmipp and Scipion
            if type(value) is unicode:
                value = str(value)
            md.setValue(label, value, objId)
            
    def __str__(self):
        s = '{'
        for k, v in self._labelDict.iteritems():
            s += '%s: %s, ' % (xmipp.label2Str(k), v)
        return s + '}'
    
    
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
    mdQuery = xmipp.MetaData() # store result
    mdQuery.importObjects(md, xmipp.MDValueEQ(label, value))
    n = mdQuery.size()
    
    if n == 0:
        row = None
    elif n == 1:
        row = XmippMdRow()
        row.getFromMd(mdQuery, mdQuery.firstObject())
    else:
        raise Exception("findRow: more than one row found matching the query %s = %s" % (xmipp.label2Str(label), value))
    
    return row

def findRowById(md, value):
    """ Same as findRow, but using MDL_ITEM_ID for label. """
    return findRow(md, xmipp.MDL_ITEM_ID, long(value))
  
  
class XmippSet():
    """ Support class to store sets in Xmipp base on a MetaData. """
    def __init__(self, itemClass):
        """ Create new set, base on a Metadata.
        itemClass: Class that represent the items.
        A method .getFileName should be available to store the md.
        Items contained in XmippSet are suposed to inherit from XmippMdRow.
        """
        self._itemClass = itemClass 
        #self._fileName = fileName      
        self._md = xmipp.MetaData()
        
        
    def __iter__(self):
        """Iterate over the set of images in the MetaData"""
        #self._md.read(self._fileName)
        
        for objId in self._md:  
            item = self._itemClass()
            item.getFromMd(self._md, objId)  
            #m = Image(md.getValue(xmipp.MDL_IMAGE, objId))
            #if self.hasCTF():
            #    m.ctfModel = XmippCTFModel(md.getValue(xmipp.MDL_CTF_MODEL, objId)) 
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
        itemXmipp.setToMd(self._md, objId)
        
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
    
    
class XmippDataSet(ds.DataSet):
    """ this class will implement a dataset base on
    a metadata file, which can contains several blocks.
    Each block is a table on the dataset and is read
    as a Xmipp metadata. 
    """
    def __init__(self, filename):
        self._filename = filename
        blocks = xmipp.getBlocksInMetaDataFile(filename)
        ds.DataSet.__init__(self, blocks)
        
    def _loadTable(self, tableName):
        md = xmipp.MetaData(tableName + "@" + self._filename)
        return self._convertMdToTable(md)
        
    def _convertMdToTable(self, md):
        """ Convert a metatada into a table. """
        labels = md.getActiveLabels()
        hasTransformation = self._hasTransformation(labels)  
        labelsStr = [xmipp.label2Str(l) for l in labels]   
        #NAPA de LUXE (xmipp deberia saber a que campo va asignado el transformation matrix)             
        if hasTransformation:
            labelsStr.append("image_transformationMatrix")   
        columns = [ds.Column(l) for l in labelsStr]
   
        table = ds.Table(*columns)
        
        for objId in md:
            values = [md.getValue(l, objId) for l in labels]
            if hasTransformation:
                values.append(self._getTransformation(md, objId))
            d = dict(zip(labelsStr, values))
            table.addRow(objId, **d)
            
        return table
    
    def _convertTableToMd(self, table):
        colLabels = [(col.getName(), xmipp.str2Label(col.getName())) 
                     for col in table.iterColumns()]
        md = xmipp.MetaData()
        
        for row in table.iterRows():
            objId = md.addObject()
            for col, label in colLabels:
                if col != 'id':
                    value = getattr(row, col)
                    md.setValue(label, value, objId)
                
        return md
        
    def writeTable(self, tableName, table):
        """ Write changes made to a table. """
        md = self._convertTableToMd(table)
        md.write("%s@%s" % (tableName, self._filename), xmipp.MD_APPEND)
        
    def _hasTransformation(self, labels):
        for l in [xmipp.MDL_SHIFT_X, xmipp.MDL_SHIFT_Y, xmipp.MDL_SHIFT_Z]:
            if l in labels:
                return True
        return False
        
    def _getTransformation(self, md, objId):
        rot  = md.getValue(xmipp.MDL_ANGLE_ROT ,objId)
        tilt = md.getValue(xmipp.MDL_ANGLE_TILT,objId)
        psi  = md.getValue(xmipp.MDL_ANGLE_PSI ,objId)
        if rot is  None:
            rot = 0
        if tilt is  None:
            tilt = 0
        if psi is  None:
            psi = 0

        tMatrix=xmipp.Euler_angles2matrix(rot,tilt,psi)
        x = md.getValue(xmipp.MDL_SHIFT_X, objId)
        y = md.getValue(xmipp.MDL_SHIFT_Y, objId)
        z = md.getValue(xmipp.MDL_SHIFT_Z, objId)

        return [tMatrix[0][0], tMatrix[0][1], tMatrix[0][2], x,
                tMatrix[1][0], tMatrix[1][1], tMatrix[1][2], y,
                tMatrix[2][0], tMatrix[2][1], tMatrix[2][2], z]


