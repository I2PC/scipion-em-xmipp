# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
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

import numpy as np

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, BooleanParam,
                                        IntParam, FileParam, FloatParam)
import pyworkflow.em.metadata as md
from pyworkflow.em.metadata.constants import (MDL_VOLUME_SCORE1, MDL_VOLUME_SCORE2)
from pyworkflow.em.protocol.protocol_3d import ProtAnalysis3D
from pyworkflow.utils import getExt
from pyworkflow.object import Float
from shutil import copyfile
from pyworkflow.em.data import VolumeMask
from pyworkflow.em import ImageHandler
from pyworkflow.em.convert import Ccp4Header
import pyworkflow.em as em
import xmipp3

VALIDATE_METHOD_URL = 'http://github.com/I2PC/scipion-em-xmipp/wiki/XmippProtValFit'
OUTPUT_PDBVOL_FILE = 'pdbVol'
OUTPUT_PDBMRC_FILE = 'pdb_volume.map'
BLOCRES_AVG_FILE = 'blocresAvg'
BLOCRES_HALF_FILE = 'blocresHalf'
RESTA_FILE = 'diferencia.vol'
RESTA_FILE_MRC = 'diferencia.map'
PDB_VALUE_FILE = 'pdb_diferencia.pdb'
MASK_FILE_MRC = 'mask.map'
MASK_FILE = 'mask.vol' 
FN_VOL = 'vol.map'
FN_HALF1 = 'half1.map'
FN_HALF2 = 'half2.map'
MD_MEANS = 'params.xmd'



class XmippProtValFit(ProtAnalysis3D):
    """    
    The protocol evaluates the quality of the fitting.
    """
    _label = 'validate fsc-q'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = 1
        self.mean_init = Float()
        self.meanA_init = Float()
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True,
                      help='Select a volume.')
        
#         form.addParam('inputPDB', PointerParam, pointerClass='PdbFile',
#                       label="Refined PDB", important=True, )
        form.addParam('inputPDB', FileParam,
                      label="PDB File path", important=True,
                      help='Specify a path to desired PDB structure.')
        
        form.addParam('pdbMap', PointerParam, pointerClass='Volume',
                      label="Volume from PDB", allowsNull=True,
                      help='Volume created from the PDB.'
                           ' The volume should be aligned with the reconstruction map.'
                           ' If the volume is not entered,' 
                           ' it is automatically created from the PDB.')        

        form.addParam('Mask', PointerParam, pointerClass='VolumeMask', 
                      allowsNull=True,
                      label="Soft Mask", 
                      help='The mask determines which points are specimen'
                      ' and which are not. If the mask is not passed,' 
                      ' the method creates an automatic mask from the PDB.')
        
        form.addParam('box', IntParam, default=20,
                      label="window size",
                      help='Kernel size for determining'
                      ' local resolution (pixels/voxels).')
        
        form.addParam('setOrigCoord', BooleanParam,
                      label="Set origin of coordinates",
                      help="Option YES:\nA new volume will be created with "
                           "the given ORIGIN of coordinates. ",
                      default=False)       
        
        form.addParam('xcoor', FloatParam, default=0, condition='setOrigCoord',
                      label="x", help="offset along x axis")
        form.addParam('ycoor', FloatParam, default=0, condition='setOrigCoord',
                      label="y", help="offset along y axis")
        form.addParam('zcoor', FloatParam, default=0, condition='setOrigCoord',
                      label="z", help="offset along z axis")
        
        form.addParallelSection(threads=8, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
                 OUTPUT_PDBVOL_FILE: self._getTmpPath('pdb_volume'),
                 OUTPUT_PDBMRC_FILE: self._getExtraPath('pdb_volume.map'),
                 BLOCRES_AVG_FILE: self._getTmpPath('blocres_avg.map'),
                 BLOCRES_HALF_FILE: self._getTmpPath('blocres_half.map'),
                 RESTA_FILE: self._getTmpPath('diferencia.vol'),
                 RESTA_FILE_MRC: self._getExtraPath('diferencia.map'),
                 PDB_VALUE_FILE:  self._getExtraPath('pdb_diferencia.pdb'),
                 MASK_FILE_MRC : self._getExtraPath('mask.map'),  
                 MASK_FILE: self._getTmpPath('mask.vol'), 
                 FN_VOL: self._getTmpPath("vol.map"),
                 FN_HALF1: self._getTmpPath("half1.map"),
                 FN_HALF2: self._getTmpPath("half2.map"),  
                 MD_MEANS: self._getExtraPath('params.xmd')        
                 }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
 
        self._createFilenameTemplates() 
        input = self._insertFunctionStep('convertInputStep')
        Id = []
        for i in range(2):
            Id.append(self._insertFunctionStep('runBlocresStep', i, prerequisites=[input]))   
        input1 = self._insertFunctionStep('substractBlocresStep',prerequisites=Id)    
        input2 = self._insertFunctionStep('assignPdbStep', prerequisites=[input1])  
        self._insertFunctionStep('createOutputStep', prerequisites=[input2])         
           

    def convertInputStep(self):
        """ Read the input volume."""
        self.volume = self.inputVolume.get()
        
        """Read the Origin."""
        if self.setOrigCoord.get():
            self.x = self.xcoor.get()
            self.y = self.ycoor.get()
            self.z = self.zcoor.get()
        else:
            self.x = 0
            self.y = 0
            self.z = 0  
                                       
        self.sampling = self.volume.getSamplingRate()
        self.origin = (-self.x*self.sampling, -self.y*self.sampling, -self.z*self.sampling) 
        
        self.vol = self.volume.getFileName()
        self.half1, self.half2 = self.volume.getHalfMaps().split(',')

        extVol = getExt(self.vol)        
        if (extVol == '.mrc') or (extVol == '.map'):
            self.vol_xmipp = self.vol + ':mrc' 
        else:
            self.vol_xmipp = self.vol     
 
        self.fnvol = self._getFileName(FN_VOL)           
        self.fnvol1 = self._getFileName(FN_HALF1)
        self.fnvol2 = self._getFileName(FN_HALF2)

        Ccp4Header.fixFile(self.vol, self.fnvol, self.origin, self.sampling,
                        Ccp4Header.START)
        Ccp4Header.fixFile(self.half1, self.fnvol1, self.origin, self.sampling,
                        Ccp4Header.START) 
        Ccp4Header.fixFile(self.half2, self.fnvol2, self.origin, self.sampling,
                        Ccp4Header.START)        
        
        """Create map from PDB """         
        if self.pdbMap.hasValue():   
            pdbvolume = self.pdbMap.get()
            self.pdbvol = pdbvolume.getFileName()
            
            self.pdbmap = self._getFileName(OUTPUT_PDBMRC_FILE)
            Ccp4Header.fixFile(self.pdbvol, self.pdbmap, self.origin, self.sampling,
                        Ccp4Header.START) 
      
        else:         
            """ Convert PDB to Map """           
            params = ' --centerPDB '
            params += ' -v 0 '        
            params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()        
            params += ' --size %d' % self.inputVolume.get().getXDim()
            params += ' -i %s' % self.inputPDB.get()        
            params += ' -o %s' % self._getFileName(OUTPUT_PDBVOL_FILE)
            self.runJob('xmipp_volume_from_pdb', params)            
    
            """ Align pdbMap to reconstruction Map """  
              
            params = ' --i1 %s' % self.vol_xmipp        
            params += ' --i2 %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'
            params += ' --local --apply'  
            params += ' %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'                    
            self.runJob('xmipp_volume_align', params)  
            
            """ convert align vol to mrc format """
            
            self.pdbvol = self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol' 
            self.pdbmap = self._getFileName(OUTPUT_PDBMRC_FILE)
            Ccp4Header.fixFile(self.pdbvol, self.pdbmap, self.origin, self.sampling,
                        Ccp4Header.START) 
        
        
        """ Create a mask"""               
        if self.Mask.hasValue():
            self.maskIn = self.Mask.get().getFileName()
            self.maskFn = self._getFileName(MASK_FILE_MRC)           
            Ccp4Header.fixFile(self.maskIn, self.maskFn, self.origin, self.sampling,
                        Ccp4Header.START)

            
            self.mask_xmipp = self.maskFn + ':mrc' 
            
        else:
            
            self.maskFn = self._getFileName(MASK_FILE_MRC)
            self.mask_xmipp = self._getFileName(MASK_FILE) 
            
            if (not self.pdbMap.hasValue()):
                            
                params = ' -i %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'          
                params += ' -o %s' % self.mask_xmipp
                params += ' --select below 0.02 --substitute binarize'                    
                self.runJob('xmipp_transform_threshold', params) 
                 
                params = ' -i %s' % self.mask_xmipp        
                params += ' -o %s' % self.mask_xmipp
                params += ' --binaryOperation dilation --size 3'                    
                self.runJob('xmipp_transform_morphology', params) 
                 
                """ convert mask.vol to mrc format """
                               
                self.maskFn = self._getFileName(MASK_FILE_MRC)           
                ImageHandler().convert(self.mask_xmipp, self.maskFn) 
                
            else:
                """ Convert PDB to Map """           
                params = ' --centerPDB '
                params += ' -v 0 '        
                params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()        
                params += ' --size %d' % self.inputVolume.get().getXDim()
                params += ' -i %s' % self.inputPDB.get()        
                params += ' -o %s' % self._getFileName(OUTPUT_PDBVOL_FILE)
                self.runJob('xmipp_volume_from_pdb', params)            
        
                """ Align pdbMap to reconstruction Map """  
                  
                params = ' --i1 %s' % self.vol_xmipp        
                params += ' --i2 %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'
                params += ' --local --apply'  
                params += ' %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'                    
                self.runJob('xmipp_volume_align', params)  
                
                """ create mask from pdbMap """
                
                params = ' -i %s' % self._getFileName(OUTPUT_PDBVOL_FILE)+'.vol'          
                params += ' -o %s' % self.mask_xmipp
                params += ' --select below 0.02 --substitute binarize'                    
                self.runJob('xmipp_transform_threshold', params) 
                 
                params = ' -i %s' % self.mask_xmipp        
                params += ' -o %s' % self.mask_xmipp
                params += ' --binaryOperation dilation --size 3'                    
                self.runJob('xmipp_transform_morphology', params) 
                 
                """ convert mask.vol to mrc format """
                self.maskFn = self._getFileName(MASK_FILE_MRC)           
                Ccp4Header.fixFile(self.mask_xmipp, self.maskFn, self.origin, self.sampling,
                        Ccp4Header.START)
                       
                              
    def runBlocresStep(self, i):
        # Local import to prevent discovery errors
        import bsoft

        if (i==0):

            """ Calculate FSC map-PDB """

            params = ' -criterio FSC -nofill -smooth -pad 1 '
            params += ' -cutoff 0.67'
            params += ' -maxresolution 2 '
            params += ' -step 1 '
            params += ' -box %d ' % self.box.get()
            params += ' -sampling %f,%f,%f' % (self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate())
#             params += ' -origin %f,%f,%f' % ((self.shifts[0], self.shifts[1], self.shifts[2]))          
            params += ' -Mask %s' % self.maskFn
            params += ' %s  %s' % (self.fnvol, self._getFileName(OUTPUT_PDBMRC_FILE))
            params += ' %s' % self._getFileName(BLOCRES_AVG_FILE)

            self.runJob(bsoft.Plugin.getProgram('blocres'), params,
                        env=bsoft.Plugin.getEnviron())
        else:

            """ Calculate FSC half1-half2 """

            params = ' -criterio FSC -nofill -smooth -pad 1 '
            params += ' -cutoff 0.5'
            params += ' -maxresolution 2 '
            params += ' -step 1 '
            params += ' -box %d ' % self.box.get()
            params += ' -sampling %f,%f,%f' % (self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate()) 
#             params += ' -origin %f,%f,%f' % ((self.shifts[0], self.shifts[1], self.shifts[2])) 
            params += ' -Mask %s' % self.maskFn
            params += ' %s  %s' % (self.fnvol1, self.fnvol2)
            params += ' %s' % self._getFileName(BLOCRES_HALF_FILE)

            self.runJob(bsoft.Plugin.getProgram('blocres'), params,
                        env=bsoft.Plugin.getEnviron())

    def substractBlocresStep(self):
        
        params = ' -i %s' % self._getFileName(BLOCRES_AVG_FILE)+':mrc'  
        params += ' --minus %s' % self._getFileName(BLOCRES_HALF_FILE)+':mrc'     
        params += ' -o %s ' % self._getFileName(RESTA_FILE)       
        self.runJob('xmipp_image_operate', params)
        
        Ccp4Header.fixFile(self._getFileName(RESTA_FILE), self._getFileName(RESTA_FILE_MRC), 
                           self.origin, self.sampling, Ccp4Header.START)
                    
        
    def assignPdbStep(self):
                 
        params = ' --pdb %s ' % self.inputPDB.get()  
        params += ' --vol %s ' % self._getFileName(RESTA_FILE_MRC) 
        params += ' --mask %s ' % self.mask_xmipp         
        params += ' -o %s ' % self._getFileName(PDB_VALUE_FILE)    
        params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()
        params += ' --origin %f %f %f' %(self.x, self.y, self.z)
        params += ' --radius 0.8'  
        params += ' --md %s' % self._getFileName(MD_MEANS)     
        self.runJob('xmipp_pdb_from_volume', params)      
        
    def createOutputStep(self):    
              
        mtd = md.MetaData()
        mtd.read(self._getFileName(MD_MEANS))
            
        self.mean = mtd.getValue(MDL_VOLUME_SCORE1,1)
        self.meanA = mtd.getValue(MDL_VOLUME_SCORE2,1)
        
        self.mean_init.set(round(self.mean*100)/100) 
        self.meanA_init.set(round(self.meanA*100)/100)             
        self._store(self.mean_init)
        self._store(self.meanA_init)

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + VALIDATE_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        summary.append("Mean deviation from the signal of the Half Maps")
        summary.append("Mean FSC-Q: %.2f" % (self.mean_init))  
        summary.append("Absotute Mean FSC-Q: %.2f" % (self.meanA_init))  
        return summary

    def _validate(self):
        errors = []
        if self.inputVolume.hasValue():
            #FIX CSVList:
            # volume.hasHalfMaps() does not work unless you call
            # getHalfMaps() or something else that triggers the CSVList.get()
            # that, populates the objValue. Just a print vol.getHalfMaps() will
            # change the behaviour of hasValue()
            # To review when migrating to Scipion3
            if not self.inputVolume.get().getHalfMaps():
                errors.append("Input Volume needs to have half maps. "
                "If you have imported the volume, be sure to import the half maps.")

        try:
            import bsoft
        except ImportError as e:
            errors.append("This protocol requires bsoft plugin to run.")
            
        return errors    

    def _citations(self):
        return ['Ramirez-Aportela 2020']

