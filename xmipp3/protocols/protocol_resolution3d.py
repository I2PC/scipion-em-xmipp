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
    """ Computes the resolution of 3D volumes using the Fourier Shell
    Correlation (FSC) criteria. The protocol requires two volumes, which are
    assumed to be independently reconstructed. In addition, the protocol can
    also compute the B-factor for the volumes.

    AI Generated

    ## Overview

    The Resolution 3D protocol estimates the resolution of a 3D reconstruction by
    computing Fourier Shell Correlation, or FSC, between two independent maps.

    FSC is one of the standard criteria used in cryo-EM to assess the agreement
    between two 3D volumes as a function of spatial frequency. If two independently
    derived maps agree at high spatial frequencies, this suggests that the
    corresponding structural features are reproducible. If the FSC drops at lower
    frequencies, the map is supported only at coarser resolution.

    This protocol can compute FSC in two ways. It can compare an input volume with
    a separate reference volume, or it can use the two half maps associated with
    the input volume. The half-map mode is commonly used for resolution assessment
    because half maps are independently reconstructed from two halves of the data.

    The protocol can also compute structure-factor information used for estimating
    a B-factor, which can later be used for map sharpening.

    ## Inputs and General Workflow

    The protocol requires an input volume.

    If FSC calculation is enabled, the protocol needs either:

    - two half maps associated with the input volume; or
    - a separate reference volume.

    The protocol then runs the Xmipp FSC calculation, producing an FSC metadata
    file. In addition to FSC, the calculation also includes DPR information.

    If B-factor computation is enabled, the protocol computes the volume structure
    factor. The resulting file can be used by the analysis tools to estimate or
    apply a B-factor.

    Finally, the protocol creates an output FSC object that can be viewed and
    analyzed in Scipion.

    ## Volume to Compare

    The **Volume to compare** parameter is the main input volume.

    This is the map whose resolution or agreement with another map will be
    evaluated. It may be a reconstruction, a post-processed map, or a volume
    produced by another Scipion protocol.

    The input volume must have a correct sampling rate, because spatial
    frequencies are converted into resolution values in angstroms.

    If the half-map mode is used, this volume must also contain links to its
    associated half maps.

    ## Calculate FSC and DPR

    The **Calculate FSC and DPR?** option controls whether the protocol computes
    the Fourier Shell Correlation and Differential Phase Residual information.

    When this option is enabled, the protocol compares the two selected maps in
    Fourier space and writes an FSC metadata file.

    The summary reports resolution estimates at commonly used thresholds,
    including:

    - FSC = 0.5;
    - FSC = 0.143;
    - DPR = 45 degrees.

    These values provide numerical reference points for interpreting the map
    agreement.

    ## Use Half Maps

    The **Use half maps** option tells the protocol to compute the FSC between the
    two half maps associated with the input volume.

    This is usually the preferred mode when half maps are available, because the
    two maps are expected to be independently reconstructed from separate halves of
    the particle data.

    The protocol checks that the input volume actually has half-map information. If
    no half maps are associated with the input volume, the protocol reports a
    validation error.

    Half-map FSC is useful for estimating the reproducible resolution of a
    reconstruction.

    ## Reference Volume

    The **Reference volume** parameter is used when half maps are not used.

    In this mode, the protocol computes the FSC between the input volume and the
    selected reference volume.

    This comparison is useful when the user wants to compare two independent
    reconstructions, two processing results, or a map against a reference map.
    However, the interpretation depends on how the two volumes were generated. If
    the maps are not independent, the FSC may overestimate reproducible
    information.

    The reference volume should have the same box size, sampling, orientation, and
    position as the input volume for the FSC to be meaningful.

    ## FSC Curve

    The FSC curve describes correlation between the two maps as a function of
    spatial frequency.

    At low spatial frequencies, FSC usually reflects agreement in the overall
    shape of the map. At high spatial frequencies, it reflects agreement in finer
    details.

    A curve that remains high until high spatial frequencies indicates stronger
    agreement at finer resolution. A curve that drops early indicates that the
    maps agree only at lower resolution.

    The protocol stores the FSC curve in an output FSC object, which can be plotted
    and inspected in Scipion.

    ## FSC Thresholds

    The protocol reports resolution values at two FSC thresholds.

    The **FSC = 0.5** criterion is a traditional, stricter threshold. It reports
    the resolution at which the FSC curve falls below 0.5.

    The **FSC = 0.143** criterion is widely used in modern cryo-EM half-map
    validation. It gives a less strict but commonly reported resolution estimate.

    The protocol estimates the crossing point by interpolation between neighboring
    frequency samples. If the curve does not cross the threshold, the reported
    resolution may be unavailable.

    ## DPR Criterion

    The protocol also reports a **DPR = 45 degrees** resolution estimate.

    DPR stands for Differential Phase Residual. It measures phase agreement between
    the two volumes in Fourier space. A smaller DPR indicates better phase
    agreement.

    The reported DPR resolution corresponds to the point where the DPR reaches 45
    degrees.

    This value complements FSC and provides another way to summarize agreement
    between the two maps.

    ## Calculate B-Factor

    The **Calculate B-factor?** option computes structure-factor information from
    the input volume.

    The B-factor describes the falloff of signal amplitude with spatial frequency.
    In cryo-EM, an estimated B-factor can be used to guide sharpening, where
    higher-resolution features are enhanced to compensate for attenuation caused by
    the microscope, detector, reconstruction, and other effects.

    This implementation follows an automated approach based on the methodology of
    Rosenthal and Henderson.

    After the protocol finishes, the B-factor can be applied through the analysis
    GUI.

    ## Structure Factor File

    When B-factor computation is enabled, the protocol writes a structure-factor
    metadata file.

    This file contains the information used to estimate the B-factor. It is not a
    map by itself, but a diagnostic and post-processing support file.

    The protocol summary reports the structure-factor file, and when the B-factor
    is available, it also reports the estimated value.

    ## Output FSC

    The main output is **outputFSC**, created when the FSC file exists.

    This output stores the FSC curve as a Scipion FSC object. It can be visualized
    and used for resolution assessment.

    The output FSC is linked to the input volume and, when a separate reference
    volume is used, also to the reference volume.

    If only B-factor computation is requested and FSC is disabled, the main result
    is the structure-factor information rather than an output FSC object.

    ## Interpreting the Results

    The FSC resolution values should be interpreted as indicators of map agreement,
    not as complete proof of biological correctness.

    A high-resolution FSC estimate is meaningful only when the two maps are
    independent or appropriately separated, properly aligned, and not affected by
    masking or overfitting artifacts.

    Half-map FSC is commonly used because the half maps are reconstructed from
    separate subsets of particles. Comparison with an external reference can be
    useful, but its interpretation depends on the independence and similarity of
    the two maps.

    B-factor estimation is useful for sharpening, but excessive sharpening can
    amplify noise. The sharpened map should always be inspected together with the
    original map and validation information.

    ## Practical Recommendations

    Use half maps when available. This is usually the most appropriate way to
    estimate the resolution of a reconstruction.

    Use a separate reference volume when you specifically want to compare two maps.
    Make sure that they are aligned, have the same box size, and represent the same
    structure.

    Report both FSC = 0.143 and FSC = 0.5 values when useful, but inspect the full
    FSC curve rather than relying only on a single number.

    Use B-factor estimation as a guide for sharpening, not as an automatic
    guarantee of improved interpretability.

    Be cautious when comparing post-processed or heavily masked maps, because
    masking and post-processing can affect FSC behavior.

    ## Final Perspective

    Resolution 3D is a validation and map-assessment protocol.

    For biological users, its main role is to quantify how reproducible a 3D map is
    in Fourier space, either by comparing two half maps or by comparing an input
    map with a reference volume.

    The protocol provides FSC and DPR resolution estimates, and optionally
    structure-factor information for B-factor estimation. These results should be
    used together with visual inspection, local resolution, map-model fit, and
    biological plausibility when assessing a cryo-EM reconstruction.
    """
    _label = 'resolution 3D'
    _devStatus = PROD
      
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
            values = [float(x) for x in f.readline().split()]
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
                values = [float(x) for x in f.readline().split()]
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

