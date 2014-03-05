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
This sub-package contains the XmippCtfMicrographs protocol
"""


from pyworkflow.em import *  

from convert import *
from xmipp3 import XmippMdRow
#from pyworkflow.utils.path import makePath, basename, join, exists
#import xmipp


class XmippProtCTFMicrographs(ProtCTFMicrographs):
    """Protocol to perform CTF estimation on a set of micrographs in the project"""
    _label = 'ctf estimation'
    __prefix = join('%(micDir)s','xmipp_ctf')
    _templateDict = {
        # This templates are relative to a micDir
        'micrographs': 'micrographs.xmd',
        'prefix': __prefix,
        'ctfparam': __prefix +  '.ctfparam',
        'psd': __prefix + '.psd',
        'enhanced_psd': __prefix + '_enhanced_psd.xmp',
        'ctfmodel_quadrant': __prefix + '_ctfmodel_quadrant.xmp',
        'ctfmodel_halfplane': __prefix + '_ctfmodel_halfplane.xmp'
#        'ctffind_ctfparam': join('%(micDir)s', 'ctffind.ctfparam'),
#        'ctffind_spectrum': join('%(micDir)s', 'ctffind_spectrum.mrc')
        }
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def _estimateCTF(self, micFn, micDir):
        """ Run the estimate CTF program """        
        # Create micrograph dir under extra directory
        print "creating path micDir=", micDir
        makePath(micDir)
        if not exists(micDir):
            raise Exception("No created dir: %s " % micDir)
        # Update _params dictionary with mic and micDir
        self._params['micFn'] = micFn
        self._params['micDir'] = self._getFilename('prefix', micDir=micDir)
        # CTF estimation with Xmipp                
        self.runJob(self._program, self._args % self._params)    
    
    def createOutputStep(self):
        # Create the SetOfMicrographs 
        micSet = self._createSetOfMicrographs()
        ctfSet = self._createSetOfCTF()
        defocusList = []
        
        for _, micDir, mic in self._iterMicrographs():
            ctfparam = self._getFilename('ctfparam', micDir=micDir)
            ctfModel = readCTFModel(ctfparam)
            mic.setCTF(ctfModel)
            micSet.append(mic)
            #TODO: WE need to use 2 objects cause same object cannot be inserted on
            #2 different mappers: change this
            
            ctfModel2 = readCTFModel(ctfparam)
            ctfModel2.setMicFile(mic.getFileName())
            ctfModel2.setObjId(mic.getObjId())
            ctfSet.append(ctfModel2)
            
            # save the values of defocus for each micrograph in a list
            defocusList.append(ctfModel2.getDefocusU())
            defocusList.append(ctfModel2.getDefocusV())
 
        self._defocusMaxMin(defocusList)
        #Copy attributes from input to output micrographs
        micSet.copyInfo(self.inputMics)
        # Write as a Xmipp metadata
        mdFn = self._getPath('micrographs_ctf.xmd')
        writeSetOfMicrographs(micSet, mdFn, self.setupMicRow)
                 
        tmpFn = self._getPath('micrographs_backup.xmd')
        writeSetOfMicrographs(micSet, tmpFn, self.setupMicRow)
        # Mark flag of CTF as True
        #micSet.setHasCTF(True)      
        #micSet.write()         
        ctfSet._xmippMd = String(mdFn)

        # Evaluate the PSD and add some criterias
        auxMdFn = self._getTmpPath('micrographs.xmd')
        self.runJob("xmipp_ctf_sort_psds","-i %s -o %s" % (mdFn, auxMdFn))
        # Copy result to output metadata
        moveFile(auxMdFn, mdFn)        
        #micSet._xmippMd.set(mdFn)
        #self._defineOutputs(outputMicrographs=micSet)
        self._defineOutputs(outputCTF=ctfSet)
        #self._defineSourceRelation(self.inputMics, micSet)
        self._defineRelation(RELATION_CTF, ctfSet, self.inputMics)
    

    def _citations(self):
        return ['Vargas2013']
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _prepareCommand(self):
        self._program = 'xmipp_ctf_estimate_from_micrograph'       
        self._args = "--micrograph %(micFn)s --oroot %(micDir)s --fastDefocus"
        
        # Mapping between base protocol parameters and the package specific command options
        self.__params = {'kV': self._params['voltage'],
                'Cs': self._params['sphericalAberration'],
                'sampling_rate': self._params['samplingRate'], 
                'ctfmodelSize': self._params['windowSize'],
                'Q0': self._params['ampContrast'],
                'min_freq': self._params['lowRes'],
                'max_freq': self._params['highRes'],
                'pieceDim': self._params['windowSize'],
                'defocus_range': (self._params['maxDefocus']-self._params['minDefocus'])/2,
                'defocusU': (self._params['maxDefocus']+self._params['minDefocus'])/2
                }
        
        for par, val in self.__params.iteritems():
            self._args += " --%s %s" % (par, str(val))

    def setupMicRow(self, mic, micRow):
        """ Add extra labels to the micrograph row, not present
        in the base class. 
        """
        micDir = self._getMicrographDir(mic)
        labels = [xmipp.MDL_PSD, xmipp.MDL_PSD_ENHANCED, xmipp.MDL_IMAGE1, xmipp.MDL_IMAGE2]
        # Filenames key related to ctf estimation
        ctfParam = self._getFilename('ctfparam', micDir=micDir)
        ctfRow = XmippMdRow()
        ctfRow.readFromFile(ctfParam)
        keys = ['psd', 'enhanced_psd', 'ctfmodel_quadrant', 'ctfmodel_halfplane']
        values = [self._getFilename(key, micDir=micDir) for key in keys]
        for l, v in zip(labels, values):
            micRow.setValue(l, v)
            
        micRow.copyFromRow(ctfRow)  
