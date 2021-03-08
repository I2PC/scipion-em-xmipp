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


from pyworkflow import VERSION_3_0
from pyworkflow.protocol.params import (PointerParam, BooleanParam,
                                        IntParam, FileParam, FloatParam)
import pwem.emlib.metadata as md
from pwem.emlib.metadata import (MDL_VOLUME_SCORE1, MDL_VOLUME_SCORE2)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.utils import getExt
from pyworkflow.object import (Float, Integer)
from pwem.objects import Volume
from pwem.convert import Ccp4Header
import xmipp3


VALIDATE_METHOD_URL = 'http://github.com/I2PC/scipion-em-xmipp/wiki/XmippProtValFit'
OUTPUT_PDBVOL_FILE = 'pdbVol'
OUTPUT_PDBMRC_FILE = 'pdb_volume.map'
BLOCRES_AVG_FILE = 'blocresAvg'
BLOCRES_HALF_FILE = 'blocresHalf'
RESTA_FILE = 'diferencia.vol'
RESTA_FILE_MRC = 'diferencia.map'
PDB_VALUE_FILE = 'pdb_fsc-q.pdb'
MASK_FILE_MRC = 'mask.map'
MASK_FILE = 'mask.vol' 
FN_VOL = 'vol.map'
FN_HALF1 = 'half1.map'
FN_HALF2 = 'half2.map'
MD_MEANS = 'params.xmd'
MD2_MEANS = 'params2.xmd'
RESTA_FILE_NORM = 'diferencia_norm.map'
PDB_NORM_FILE = 'pdb_fsc-q_norm.pdb'


class XmippProtValFit(ProtAnalysis3D):
    """    
    The protocol assesses the quality of the fit.
    """
    _label = 'validate fsc-q'
    _lastUpdateVersion = VERSION_3_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = 1
    
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

        form.addParam('inputMask', PointerParam, pointerClass='VolumeMask', 
                      allowsNull=True,
                      label="Soft Mask", 
                      help='The mask determines which points are specimen'
                      ' and which are not. If the mask is not passed,' 
                      ' the method creates an automatic mask from the PDB.')
        
        form.addParam('box', IntParam, default=20,
                      label="window size",
                      help='Kernel size (slidding window) for determining'
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
                 RESTA_FILE_NORM: self._getExtraPath('diferencia_norm.map'),
                 PDB_VALUE_FILE:  self._getExtraPath('pdb_fsc-q.pdb'),
                 PDB_NORM_FILE: self._getExtraPath('pdb_fsc-q_norm.pdb'), 
                 MASK_FILE_MRC : self._getExtraPath('mask.map'),  
                 MASK_FILE: self._getTmpPath('mask.vol'), 
                 FN_VOL: self._getTmpPath("vol.map"),
                 FN_HALF1: self._getTmpPath("half1.map"),
                 FN_HALF2: self._getTmpPath("half2.map"),  
                 MD_MEANS: self._getExtraPath('params.xmd'),  
                 MD2_MEANS: self._getExtraPath('params2.xmd')      
                 }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
 
        self._createFilenameTemplates() 
        input = self._insertFunctionStep('convertInputStep')
        id = []
        for i in range(2):
            id.append(self._insertFunctionStep('runBlocresStep', i, prerequisites=[input]))   
        input1 = self._insertFunctionStep('substractBlocresStep',prerequisites=id)    
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
        if self.inputMask.hasValue():
            self.maskIn = self.inputMask.get().getFileName()
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
                Ccp4Header.fixFile(self.mask_xmipp, self.maskFn, self.origin, self.sampling,
                        Ccp4Header.START)
                
            else:
                
                """ create mask from pdbMap """                
                params = ' -i %s' % self._getFileName(OUTPUT_PDBMRC_FILE)+':mrc'          
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
            params += ' -maxresolution 0.5 '
            params += ' -step 1 '
            params += ' -box %d ' % self.box.get()
            params += ' -sampling %f,%f,%f' % (self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate())
#             params += ' -origin %f,%f,%f' % ((self.shifts[0], self.shifts[1], self.shifts[2]))          
            params += ' -Mask %s' % self.maskFn
            params += ' %s  %s' % (self.fnvol, self._getFileName(OUTPUT_PDBMRC_FILE))
            params += ' %s' % self._getFileName(BLOCRES_AVG_FILE)

            self.runJob(bsoft.Plugin.getProgram('blocres', bsoftVersion=bsoft.V1_9_0), params,
                        env=bsoft.Plugin.getEnviron(bsoftVersion=bsoft.V1_9_0))
        else:

            """ Calculate FSC half1-half2 """

            params = ' -criterio FSC -nofill -smooth -pad 1 '
            params += ' -cutoff 0.5'
            params += ' -maxresolution 0.5 '
            params += ' -step 1 '
            params += ' -box %d ' % self.box.get()
            params += ' -sampling %f,%f,%f' % (self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate(),
                                               self.inputVolume.get().getSamplingRate()) 
#             params += ' -origin %f,%f,%f' % ((self.shifts[0], self.shifts[1], self.shifts[2])) 
            params += ' -Mask %s' % self.maskFn
            params += ' %s  %s' % (self.fnvol1, self.fnvol2)
            params += ' %s' % self._getFileName(BLOCRES_HALF_FILE)

            self.runJob(bsoft.Plugin.getProgram('blocres', bsoftVersion=bsoft.V1_9_0), params,
                        env=bsoft.Plugin.getEnviron(bsoftVersion=bsoft.V1_9_0))

    def substractBlocresStep(self):
        
        params = ' -i %s' % self._getFileName(BLOCRES_AVG_FILE)+':mrc'  
        params += ' --minus %s' % self._getFileName(BLOCRES_HALF_FILE)+':mrc'     
        params += ' -o %s ' % self._getFileName(RESTA_FILE)       
        self.runJob('xmipp_image_operate', params)
        
        Ccp4Header.fixFile(self._getFileName(RESTA_FILE), self._getFileName(RESTA_FILE_MRC), 
                           self.origin, self.sampling, Ccp4Header.START)
        
        """Diveded by resolution"""       
        Vx = xmipp3.Image(self._getFileName(RESTA_FILE))
        V=Vx.getData()
        Vmask = xmipp3.Image(self._getFileName(MASK_FILE_MRC)+':mrc').getData()
        Vres = xmipp3.Image(self._getFileName(BLOCRES_HALF_FILE)+':mrc').getData()
        Vt = V
        Zdim, Ydim, Xdim = V.shape
              
        for z in range(0,Zdim):
            for y in range(0,Ydim):
                for x in range(0,Xdim):
                    if (Vmask[z,y,x] > 0.001 and Vres[z,y,x]>0.001): 
                        Vt[z,y,x] = (V[z,y,x]/Vres[z,y,x])
        Vx.setData(Vt) 
        Vx.write(self._getFileName(RESTA_FILE_NORM))
        Ccp4Header.fixFile(self._getFileName(RESTA_FILE_NORM), self._getFileName(RESTA_FILE_NORM), 
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
        self.runJob('xmipp_pdb_label_from_volume', params)   
        
        """Diveded by resolution"""
        params = ' --pdb %s ' % self.inputPDB.get()  
        params += ' --vol %s ' % self._getFileName(RESTA_FILE_NORM) 
        params += ' --mask %s ' % self.mask_xmipp         
        params += ' -o %s ' % self._getFileName(PDB_NORM_FILE)    
        params += ' --sampling %f' % self.inputVolume.get().getSamplingRate()
        params += ' --origin %f %f %f' %(self.x, self.y, self.z)
        params += ' --radius 0.8' 
        params += ' --md %s' % self._getFileName(MD2_MEANS) 
        self.runJob('xmipp_pdb_label_from_volume', params)    
        
    def createOutputStep(self):    
        
        volume=Volume()
        volume.setFileName(self._getFileName(RESTA_FILE_MRC))
        volume.setSamplingRate(self.inputVolume.get().getSamplingRate())
        volume.setOrigin(self.inputVolume.get().getOrigin(True))
        self._defineOutputs(fscq_Volume=volume)
        self._defineTransformRelation(self.inputVolume, volume)
        
        #mean values of FSC-Q
              
        mtd = md.MetaData()
        mtd.read(self._getFileName(MD_MEANS))
            
        mean = mtd.getValue(MDL_VOLUME_SCORE1,1)
        meanA = mtd.getValue(MDL_VOLUME_SCORE2,1)  
        
        #means value for map divided by resolution (FSC-Qr)
        mtd2 = md.MetaData()
        mtd2.read(self._getFileName(MD2_MEANS))
            
        mean2 = mtd2.getValue(MDL_VOLUME_SCORE1,1)
        meanA2 = mtd2.getValue(MDL_VOLUME_SCORE2,1)
              
        #Setting the mean fsc-q for the summary
        self.mean = Float(mean)
        self._store(self) 
        self.meanA = Float(meanA)
        self._store(self) 
        
        self.mean2 = Float(mean2)
        self._store(self) 
        self.meanA2 = Float(meanA2)
        self._store(self)

        
        #statistic from fnal pdb with fsc-q
        #Number of atoms greater or less than 0.5
        total_atom=0
        fscq_greater=0
        fscq_less=0
        with open(self._getFileName(PDB_VALUE_FILE)) as f:
            lines_data = f.readlines()
            for j,lin in enumerate(lines_data):
                
                if ( lin.startswith('ATOM') or lin.startswith('HETATM') ):
                    
                    total_atom=total_atom+1
                    fscq_atom = float(lin[54:60])
                    
                    if (fscq_atom>0.5):
                        fscq_greater=fscq_greater+1
                        
                    if (fscq_atom<-0.5):
                        fscq_less=fscq_less+1
                        
        porc_greater=(fscq_greater*100)/total_atom
        porc_less=(fscq_less*100)/total_atom
 
        self.total_atom = Integer(total_atom)
        self._store(self)
        self.fscq_greater=Integer(fscq_greater)
        self._store(self)
        self.fscq_less=Integer(fscq_less)
        self._store(self)        
        self.porc_greater=Float(porc_greater)
        self._store(self)
        self.porc_less=Float(porc_less)
        self._store(self)           

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + VALIDATE_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        summary.append("Deviation from the signal of the Half Maps")
        if self.hasAttribute('mean'):
            summary.append("Mean FSC-Q: %.2f" % (self.mean.get()))
        if self.hasAttribute('meanA'):    
            summary.append("Absotute Mean FSC-Q: %.2f" % (self.meanA.get()))
             
        summary.append(" ")            
        summary.append("Deviation from the signal of the Half Maps divided by local resolution")
        if self.hasAttribute('mean2'):
            summary.append("Mean FSC-Qr: %.2f" % (self.mean2.get()))
        if self.hasAttribute('meanA2'):    
            summary.append("Absotute Mean FSC-Qr: %.2f" % (self.meanA2.get()))      
             
        summary.append("------------------------------------------")   
        if self.hasAttribute('total_atom'):                    
            summary.append("Total number of atoms analyzed: %d" % (self.total_atom.get()))
        if (self.hasAttribute('fscq_greater') and self.hasAttribute('porc_greater')):            
            summary.append("Number of atoms with FSC-Q>0.5: %d.  Percentage of total: %.2f." 
                       % (self.fscq_greater.get(), self.porc_greater.get()))
        if (self.hasAttribute('fscq_less') and self.hasAttribute('porc_less')):             
            summary.append("Number of atoms with FSC-Q<-0.5: %d.  Percentage of total: %.2f." 
                       % (self.fscq_less.get(), self.porc_less.get()))
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
            if bsoft.__version__ in ["3.0.0", "3.0.1", "3.0.4"]:
                errors.append("This protocol requires bsoft plugin 3.0.5 or above to run."
                              " You have %s. Update it using the plugin manager or command line" % bsoft.__version__)
        except Exception as e:
            errors.append("This protocol requires bsoft plugin 3.0.5 or above to run. Update it using the plugin manager or command line")
            
        return errors    

    def _citations(self):
        return ['Ramirez-Aportela 2020']

