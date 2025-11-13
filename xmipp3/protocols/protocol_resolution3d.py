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

from pyworkflow.protocol import PointerParam, BooleanParam
from pyworkflow.utils import *

import pwem.emlib.metadata as md
from pwem.objects import FSC
from pwem.protocols import ProtAnalysis3D

from xmipp3.convert import (getImageLocation)
from pyworkflow import BETA, UPDATED, NEW, PROD

def _fixMRC(fn):
    if fn.endswith(".mrc"):
        return fn+":mrc"
    else:
        return fn

class XmippProtResolution3D(ProtAnalysis3D):
    """ Computes the resolution of 3D volumes using the Fourier Shell Correlation (FSC) criteria. The protocol requires two volumes, which are assumed to be independently reconstructed. In addition, the protocol can also compute the B-factor for the volumes. """
    _label = 'resolution 3D'
    _devStatus = UPDATED
      
    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        form.addSection(label='Input')
        # Volumes to process
        form.addParam('inputVolume', PointerParam, label="Volume to compare", important=True, 
                      pointerClass='Volume',
                      help='This volume will be compared to the reference volume.')  
        form.addParam('doFSC', BooleanParam, default=True,
                      label="Calculate FSC and DPR?", 
                      help='If set True calculate FSC and DPR.')
        form.addParam('useHalves', BooleanParam, default=False,
                      label="Use half maps",
                      help='If available.')
        form.addParam('referenceVolume', PointerParam, label="Reference volume", condition='doFSC and not useHalves',
                      pointerClass='Volume', allowsNull=True,
                      help='Input volume will be compared to this volume.')  
        form.addParam('doComputeBfactor', BooleanParam, default=True,
                      label="Calculate B-factor?", 
                      help="If set True the so-called B-factor will be estimated.\n"
                           "The B-factor can be used to sharpen a volume.\n"
                           "The high-resolution features will enhanced, thereby\n"
                           "correcting the envelope functions of the microscope,\n"
                           "detector etc. This implementation follows the\n"
                           "automated mode based on methodology developed by Rosenthal2003\n\n"
                           "*Note*: after finished, you can apply the B-factor through\n"
                           "   the _Analyze Results_ GUI."
                      )

    #--------------------------- INSERT steps functions --------------------------------------------  
    def _insertAllSteps(self):
        """Insert all steps to calculate the resolution of a 3D reconstruction. """
        
        self.inputVol = _fixMRC(getImageLocation(self.inputVolume.get()))
        if self.doFSC:
            self._insertFunctionStep('calculateFscStep')
        if self.doComputeBfactor:
            self._insertFunctionStep('computeBfactorStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep('createSummaryStep')

    def createOutputStep(self):
        fnFSC = self._defineFscName()
        if exists(fnFSC):
            mData = md.MetaData(fnFSC)
            # Create the FSC object and set the same protocol label
            fsc = FSC(objLabel=self.getRunName())
            fsc.loadFromMd(mData,
                           md.MDL_RESOLUTION_FREQ,
                           md.MDL_RESOLUTION_FRC)
            self._defineOutputs(outputFSC=fsc)
            if not self.useHalves:
                self._defineSourceRelation(self.referenceVolume,fsc)
            self._defineSourceRelation(self.inputVolume,fsc)

    #--------------------------- STEPS steps functions --------------------------------------------
    def calculateFscStep(self):
        """ Calculate the FSC between two volumes"""
        if self.useHalves:
            fn1, fn2 = [_fixMRC(fn) for fn in self.inputVolume.get().getHalfMaps().split(",")]
        else:
            fn1 = self.inputVol
            fn2 = _fixMRC(getImageLocation(self.referenceVolume.get()))

        samplingRate = self.inputVolume.get().getSamplingRate()
        args = "--ref %s -i %s -o %s --sampling_rate %f --do_dpr" % (fn1, fn2, self._defineFscName(), samplingRate)
        self.runJob("xmipp_resolution_fsc", args)

    def computeBfactorStep(self):
        """ Calculate the structure factors of the volume"""
        samplingRate = self.inputVolume.get().getSamplingRate()
        structureFn = self._defineStructFactorName()
        args = "-i %s -o %s --sampling %f" % (self.inputVol, structureFn, samplingRate)
        self.runJob("xmipp_volume_structure_factor", args)
    
    def createSummaryStep(self):
        summary=""
        methodsStr=""
        if self.doFSC.get():
            summary+="FSC file: %s\n" % self.getFileTag(self._defineFscName())
            mData=md.MetaData(self._defineFscName())
            f=self.calculateFSCResolution(mData,0.5)
            summary+="   Resolution FSC=0.5: %3.2f Angstroms\n"%f
            methodsStr+=" The resolution at FSC=0.5 was %3.2f Angstroms."%f
            f=self.calculateFSCResolution(mData,0.143)
            summary+="   Resolution FSC=0.143: %3.2f Angstroms\n"%f
            methodsStr+=" The resolution at FSC=0.143 was %3.2f Angstroms."%f
            f=self.calculateDPRResolution(mData,45)
            summary+="   Resolution DPR=45: %3.2f Angstroms\n"%f
            methodsStr+=" The resolution at DPR=45 was %3.2f Angstroms."%f
        if self.doComputeBfactor:
            summary+="Structure factor file: %s\n" % self.getFileTag(self._defineStructFactorName())
        self.methodsVar.set(methodsStr)
        self.summaryVar.set(summary)
    
    #--------------------------- INFO steps functions --------------------------------------------
    def _validate(self):
        validateMsgs = []
        
        if self.useHalves:
            inputVol = self.inputVolume.get()
            if inputVol and not inputVol.hasHalfMaps():
                validateMsgs.append('The input volume does not have half maps to use')
        if self.doFSC and not self.useHalves:
            referenceVol = self.referenceVolume.get()
            if not referenceVol:
                validateMsgs.append('Please, provide a reference map to compute the FSC')

        return validateMsgs
    
    def _summary(self):
        retval=self.summaryVar.get()
        fnBfactor= self._getPath('bfactor.txt')
        if os.path.exists(fnBfactor):
            f = open(fnBfactor)
            values = map(float, f.readline().split())
            retval+="   Bfactor: %4.3f"%values[4]
        return [retval]
    
    def _methods(self):
        methodsStr=""
        if self.doFSC.get():
            methodsStr+='We obtained the FSC of the volume %s' % self.getObjectTag('inputVolume')
            if self.useHalves:
                methodsStr+=" considering its two associated half maps."
            else:
                methodsStr+=' taking the volume %s' % self.getObjectTag('referenceVolume') + ' as reference.'
            methodsStr+=self.methodsVar.get("")
        if self.doComputeBfactor.get():
            fnBfactor= self._getPath('bfactor.txt')
            if os.path.exists(fnBfactor):
                f = open(fnBfactor)
                values = map(float, f.readline().split())
                methodsStr+=" The corresponding Bfactor was %4.3f."%values[4]
        return [methodsStr]

    def _citations(self):
        return ['Rosenthal2003']
    #--------------------------- UTILS functions ---------------------------------------------------
    def _defineStructFactorName(self):
        return self._getPath('structureFactor.xmd')
    
    def _defineFscName(self):
        return self._getPath('fsc.xmd')

    def calculateFSCResolution(self, mData, threshold):
        xl=-1
        xr=-1
        yl=-1
        yr=-1
        leftSet=False
        rightSet=False
        for objId in mData:
            freq=mData.getValue(md.MDL_RESOLUTION_FREQ,objId)
            FSC=mData.getValue(md.MDL_RESOLUTION_FRC,objId)
            if FSC>threshold:
                xl=freq
                yl=FSC
                leftSet=True
            else:
                xr=freq
                yr=FSC
                rightSet=True
                break
        if leftSet and rightSet:
            x=xl+(threshold-yl)/(yr-yl)*(xr-xl)
            return 1.0/x
        else:
            return -1
    
    def calculateDPRResolution(self, mData, threshold):
        xl=-1
        xr=-1
        yl=-1
        yr=-1
        leftSet=False
        rightSet=False
        for objId in mData:
            freq=mData.getValue(md.MDL_RESOLUTION_FREQ,objId)
            DPR=mData.getValue(md.MDL_RESOLUTION_DPR,objId)
            if DPR<threshold:
                xl=freq
                yl=DPR
                leftSet=True
            else:
                xr=freq
                yr=DPR
                rightSet=True
                break
        if leftSet and rightSet:
            x=xl+(threshold-yl)/(yr-yl)*(xr-xl)
            return 1.0/x
        else:
            return -1

