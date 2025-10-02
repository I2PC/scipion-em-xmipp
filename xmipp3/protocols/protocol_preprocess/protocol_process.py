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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from pwem.constants import ALIGN_NONE
from pwem.protocols import ProtProcessParticles, ProtPreprocessVolumes
from pwem.objects import Volume
import pwem.emlib.metadata as md
from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.convert import headers

from xmipp3.convert import (writeSetOfParticles, xmippToLocation,
                            writeSetOfVolumes, getImageLocation)


class XmippProcessParticles(ProtProcessParticles):
    """ Class to create a base template for Xmipp protocols 
    that process SetOfParticles
    """
    _devStatus = UPDATED

    def __init__(self, **kwargs):
        ProtProcessParticles.__init__(self, **kwargs)
        self._args = "-i %(inputFn)s "

    #--------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._defineFilenames()
        self._insertFunctionStep("convertInputStep")
        self._insertProcessStep()
        self._insertFunctionStep('createOutputStep')
        
    def _preprocessOutput(self, output):
        """ This function should be implemented
        if some additional modifications needs to be done
        before population of the output elements.
        """
        pass  
      
    def _postprocessOutput(self, output):
        """ This function should be implemented
        if some additional modifications needs to be done
        after the population of output elements.
        """
        pass
    
    def _updateItem(self, item, row):
        """ Implement this function to do some
        update actions over each single item
        that will be stored in the output Set.
        """
        # By default update the item location (index, filename)
        # with the new binary data location (after preprocessing)
        newFn = row.getValue(md.MDL_IMAGE)
        newLoc = xmippToLocation(newFn)
        item.setLocation(newLoc)
            
    #--------------------------- STEPS functions ------------------------------
    def convertInputStep(self):
        """ convert if necessary"""
        # By default the prepocess protocol will ignore geometry
        # info and apply the operation on the binary data only.
        # then the new location (index, filename) is the most
        # common property to update in the single items.
        writeSetOfParticles(self.inputParticles.get(), self.inputFn,
                            alignType=ALIGN_NONE)
    
    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        # outputSet could be SetOfParticles, SetOfAverages or any future sub-class of SetOfParticles
        className = inputSet.getClassName()
        outputSet = inputSet.createCopy(self._getPath())
        outputSet.copyInfo(inputSet)

        self._preprocessOutput(outputSet)
        
        outputSet.copyItems(inputSet, 
                            updateItemCallback=self._updateItem,
                            itemDataIterator=md.iterRows(self.outputMd, sortByLabel=md.MDL_ITEM_ID))
        self._postprocessOutput(outputSet)
        
        outputKey = className.replace('SetOf', 'output')
        self._defineOutputs(**{outputKey: outputSet})
        self._defineTransformRelation(inputSet, outputSet)
    
    #--------------------------- UTILS functions ------------------------------
    def _defineFilenames(self):
        self.inputFn = self._getTmpPath('input_particles.xmd')
        self.outputMd = self._getExtraPath('output_images.xmd')
        self.outputStk = self._getExtraPath('output_images.mrcs')


class XmippProcessVolumes(ProtPreprocessVolumes):
    """ Class to create a base template for Xmipp protocols that process 
    both volume or a SetOfVolumes objects 
    """
    def __init__(self, **kwargs):
        ProtPreprocessVolumes.__init__(self, **kwargs)
        self._args = "-i %(inputFn)s "

    #--------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._defineFilenames()
        self._insertFunctionStep("convertInputStep")
        self._insertProcessStep()
        self._insertFunctionStep('createOutputStep')
        
    def _preprocessOutput(self, output):
        """ This function should be implemented
        if some additional modifications needs to be done
        before population of the output elements.
        """
        pass  
      
    def _postprocessOutput(self, output):
        """ This function should be implemented
        if some additional modifications needs to be done
        after the population of output elements.
        """
        pass
        
    #--------------------------- STEPS functions ------------------------------
    def convertInputStep(self):
        """ convert if necessary"""
        if not self._isSingleInput():
            writeSetOfVolumes(self.inputVolumes.get(), self.inputFn)
            
    def createOutputStep(self):
        volInput = self.inputVolumes.get()
        if hasattr(self, 'resizeOption') and self.doResize:
            samplingRate = self.resizeSamplingRate.get()
        else:
            samplingRate = volInput.getSamplingRate()
        if self._isSingleInput():
            # Create the output with the same class as
            # the input, that should be Volume or a subclass
            # of Volume like VolumeMask
            volClass = volInput.getClass()
            vol = volClass() # Create an instance with the same class of input 
            vol.copyInfo(volInput)
            vol.setLocation(self.outputStk)
            if self.outputStk.endswith(".mrc"):
                self.setHeader(self.outputStk, samplingRate)
                if volInput.hasHalfMaps():
                    vol.setHalfMaps([self._getExtraPath("half1.mrc"), self._getExtraPath("half2.mrc")])

            if volInput.hasOrigin():
                vol.setOrigin(volInput.getOrigin())
            self._postprocessOutput(vol)
            self._defineOutputs(outputVol=vol)
        else:
            # ToDo: createSetOfVolumes not work properly when the protocol is resumed.
            volumes = volInput.create(self._getPath())
            volumes.copyInfo(volInput)
            self._preprocessOutput(volumes)
            for i, obj in enumerate(volInput.iterItems()):
                vol = obj
                fnSingle = self._getExtraPath("output_volume%03d.mrc"%(i+1))
                self.runJob("xmipp_image_convert","-i %d@%s  -o %s -t vol"%(i+1,self.outputStk, fnSingle))
                self.setHeader(fnSingle, samplingRate)
                vol.setLocation(fnSingle)
                volumes.append(vol)
            self.runJob("rm",self.outputStk)
            self._postprocessOutput(volumes)
            self._defineOutputs(outputVol=volumes)
            
        self._defineTransformRelation(self.inputVolumes, self.outputVol)
    
    def setHeader(self, filename, samplingRate):
        ccp4header = headers.Ccp4Header(filename, readHeader=True)
        ccp4header.setSampling(samplingRate)
        ccp4header.writeHeader()
    
    #--------------------------- UTILS functions ------------------------------
    def _isSingleInput(self):
        return isinstance(self.inputVolumes.get(), Volume)
        
    def _defineFilenames(self):
        """ Prepare the files to process """
        if self._isSingleInput():
            self.inputFn = getImageLocation(self.inputVolumes.get())
            self.outputStk = self._getExtraPath("output_volume.mrc")
        else:
            self.inputFn = self._getTmpPath('input_volumes.xmd')
            self.outputStk = self._getExtraPath("output_volume.mrcs")
            self.outputMd = self._getExtraPath('output_volumes.xmd')

