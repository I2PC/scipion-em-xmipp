# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume
from pyworkflow.utils import getExt
import pwem.emlib.metadata as md

from pwem.emlib import (MDL_XCOOR, MDL_YCOOR, MDL_ZCOOR,
                      MDL_ANGLE_ROT, MDL_ANGLE_TILT,
                      MDL_MAX, MDL_MIN, MDL_INTSCALE)


MONORES_METHOD_URL = 'https://github.com/I2PC/scipion/wiki/XmippProtMonoDir'


OUTPUT_RADIAL_AVERAGES = 'Radial_averages.xmd'
OUTPUT_RESOLUTION_FILE = 'monoresResolutionMap.mrc'
OUTPUT_RESOLUTION_FILE_CHIMERA = 'monoresResolutionChimera.mrc'
OUTPUT_MASK_FILE = 'output_Mask.vol'
FN_MEAN_VOL = 'mean_volume.vol'
METADATA_ANGLES_FILE = 'angles_md.xmd'
OUTPUT_DOA_FILE = 'local_anisotropy.vol'
OUTPUT_VARIANCE_FILE = 'resolution_variance.vol'
OUTPUT_DIRECTIONS_FILE = 'ellipsoids.xmd'
OUTPUT_MD_RADIAL_FILE = 'radial_resolution.xmd'
OUTPUT_MD_AZIMUTHAL_FILE = 'azimuthal_resolution.xmd'
OUTPUT_DESCR = 'ellipsoid.descr'
OUTPUT_ELLIP = 'ellipsoid.vol'
OUTPUT_RADIAL_FILE = 'radial_resolution.vol'
OUTPUT_AZIMUTHAL_FILE = 'azimuthal_resolution.vol'
OUTPUT_MEANRES_FILE = 'mean_resolution.vol'
OUTPUT_HIGHESTRES_FILE = 'highestResolution.vol'
OUTPUT_DOA1_FILE = 'doaMetric.vol'
OUTPUT_DOA2_FILE = 'meanResdoa.vol'
OUTPUT_LOWESTRES_FILE = 'lowestResolution.vol'
OUTPUT_THRESHOLDS_FILE = 'thresholds.xmd'
OUTPUT_MD_MINDIRECTIONAL_FILE = 'hist_prefdir.xmd'
OUTPUT_ZSCOREMAP_FILE = 'zscoreMap.vol'


class XmippProtMonoDir(ProtAnalysis3D):
    """    
    Asseses directional local resolution values of a 3D map. Enables identifying angular assignment errors and possible preferential directions. This method uses monores local resolution algorithm in a directional manner.
    """
    _label = 'directional resolution MonoDir'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolumes', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True,
                      help='Select a volume for determining its local resolution.')

        form.addParam('Mask', PointerParam, pointerClass='VolumeMask',
                      label="Binary Mask", allowsNull=True,
                      help='The mask determines which points are specimen and which ones not')

        group = form.addGroup('Extra parameters')
#         group.addParam('angularsampling', FloatParam, default=15, expertLevel=LEVEL_ADVANCED,
#                       label="Angular Sampling",
#                       help='Angular sampling to cover the projection sphere')
        
        group.addParam('significance', FloatParam, default=0.95, expertLevel=LEVEL_ADVANCED,
                      label="Significance",
                      help='Relution is computed using hipothesis tests, this value determines'
                      'the significance of that test')
        
        group.addParam('resstep', FloatParam, default=0.5, expertLevel=LEVEL_ADVANCED,
                      label="Resolution Step",
                      help='The resolution will be sought in steps of this values, '
                      'with step = 0.3, then 1A, 1.3A, 1.6A,...')
        
        group.addParam('fast', BooleanParam, default=False, 
                      label="Fast Computation",
                      help='Fast computation is recommended for large volumes.')
        
        group.addParam('isPremasked', BooleanParam, default=False,
                      label="Is the original premasked?",
                      help='Sometimes the original volume is masked inside a spherical mask. In this case'
                      'please select yes')
        
        group.addParam('volumeRadius', FloatParam, default=-1,
                      label="Spherical mask radius (px)",
                      condition = 'isPremasked', 
                      help='When the original volume is originally premasked, the noise estimation ought'
                      'to be performed inside that premask, and out of the provieded mask asked in the previus'
                      'box. The radius value, determines the radius of the spherical premask. By default'
                      'radius = -1 use the half of the volume size as radius')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        
        self.micsFn = self._getPath()

        self.vol0Fn = self.inputVolumes.get().getFileName()
        self.maskFn = self.Mask.get().getFileName()

        # Convert input into xmipp Metadata format
        convertId = self._insertFunctionStep('convertInputStep')

        self._insertFunctionStep('MonoResStep')      
          
        self._insertFunctionStep('directionalResolutionStep',
                                      prerequisites=[convertId])

        self._insertFunctionStep('createOutputStep')

#         self._insertFunctionStep("createEllipsoid")
        self._insertFunctionStep("createHistrogramStep")
        
        
    def convertInputStep(self):
        """ Read the input volume.
        """
        extVol0 = getExt(self.vol0Fn)
        if (extVol0 == '.mrc') or (extVol0 == '.map'):
            self.vol0Fn = self.vol0Fn + ':mrc'

        extMask = getExt(self.maskFn)
        if ((extMask == '.mrc') or (extMask == '.map')):
            self.maskFn = self.maskFn + ':mrc'


    def directionalResolutionStep(self):

        if self.isPremasked:
            if self.volumeRadius == -1:
                xdim, _ydim, _zdim = self.inputVolumes.get().getDim()
                xdim = xdim*0.5
            else:
                xdim = self.volumeRadius.get()
        else:
            xdim, _ydim, _zdim = self.inputVolumes.get().getDim()
            xdim = xdim*0.5

        # Number of frequencies
        #Nfreqs = xdim
                
        params = ' --vol %s' % self.vol0Fn
        params += ' --mask %s' % self.maskFn
        params += ' -o %s' % self._getExtraPath(OUTPUT_RESOLUTION_FILE)
        params += ' --sampling_rate %f' % self.inputVolumes.get().getSamplingRate()
        params += ' --volumeRadius %f' % xdim
        params += ' --significance %f' % self.significance.get()
        params += ' --resStep %f' % self.resstep.get()
        params += ' --radialRes %s' % self._getExtraPath(OUTPUT_RADIAL_FILE)
        params += ' --azimuthalRes %s' % self._getExtraPath(OUTPUT_AZIMUTHAL_FILE)
        params += ' --highestResolutionVol %s' % self._getExtraPath(OUTPUT_HIGHESTRES_FILE)
        params += ' --lowestResolutionVol %s' % self._getExtraPath(OUTPUT_LOWESTRES_FILE)
        params += ' --doa1 %s' % self._getExtraPath(OUTPUT_DOA1_FILE)
        params += ' --doa2 %s' % self._getExtraPath(OUTPUT_DOA2_FILE)
        params += ' --radialAzimuthalThresholds %s' % self._getExtraPath(OUTPUT_THRESHOLDS_FILE)
        params += ' --radialAvG %s' % self._getExtraPath(OUTPUT_RADIAL_AVERAGES)
        params += ' --prefMin %s' % self._getExtraPath(OUTPUT_MD_MINDIRECTIONAL_FILE)
        params += ' --zScoremap %s' % self._getExtraPath(OUTPUT_ZSCOREMAP_FILE)
        params += ' --threads %i' % self.numberOfThreads.get()
        params += ' --monores %s' % self._getExtraPath(OUTPUT_RESOLUTION_FILE)
        if (self.fast.get() is True):
            params += ' --fast'

        self.runJob('xmipp_resolution_directional', params)
        
        #TODO: Take a metadata and set maxRes minRes, idem with azimuthal and tangencial


    def MonoResStep(self):

        params = ' --vol %s' % self.vol0Fn
        params += ' --mask %s' % self.maskFn
        params += ' -o %s' % self._getExtraPath()
        params += ' --sampling_rate %f' % self.inputVolumes.get().getSamplingRate()
        params += ' --step %f' % 0.25
        params += ' --minRes %f' % (2.0*self.inputVolumes.get().getSamplingRate())
        params += ' --maxRes %f' % 18.0
        params += ' --significance %f' % self.significance.get()
        params += ' --threads %i' % self.numberOfThreads.get()  
        self.runJob('xmipp_resolution_monogenic_signal', params)

    def createEllipsoid(self):

        xdim, ydim, zdim = self.inputVolumes.get().getDim()
        f = open(self._getExtraPath(OUTPUT_DESCR),'w') 
        str_ = '%i %i %i 0\n' %(xdim, ydim, zdim)     
        f.write(str_)
        
        mtd = md.MetaData()
        mtd.read(self._getExtraPath(OUTPUT_DIRECTIONS_FILE))
        for objId in mtd:
            xcoor = mtd.getValue(MDL_XCOOR, objId)
            ycoor = mtd.getValue(MDL_YCOOR, objId)
            zcoor = mtd.getValue(MDL_ZCOOR, objId)
            rot = mtd.getValue(MDL_ANGLE_ROT, objId)
            tilt = mtd.getValue(MDL_ANGLE_TILT, objId)
            len_max = mtd.getValue(MDL_MAX, objId)
            len_min = mtd.getValue(MDL_MIN, objId)
            doa = mtd.getValue(MDL_INTSCALE, objId)
            str_ = 'ell = %f %i %i %i %f %f %f %f %f 0\n' %(doa, xcoor, ycoor, zcoor, 
                                            len_max, len_min, len_min, rot, tilt)
            f.write(str_)
        
        f.close()

        params = ' -i %s' % self._getExtraPath(OUTPUT_DESCR)
        params += ' -o %s' % self._getExtraPath(OUTPUT_ELLIP)
        
        self.runJob('xmipp_phantom_create', params)


    def createHistrogram(self, fnVol, fnOut, doa):

        params = ' -i %s' % fnVol
        params += ' --mask binary_file %s' % self.maskFn
        params += ' --steps %f' % 30
        params += ' -o %s' % fnOut
        if doa is True:
            params += ' --range %f %f' % (0, 1)#(self.minRes.get(), self.maxRes.get())
        else:
            params += ' --range %f %f' % (0, 30)
        self.runJob('xmipp_image_histogram', params)
        
    
    def createHistrogramStep(self):
        self.createHistrogram(self._getExtraPath(OUTPUT_DOA1_FILE), self._getExtraPath('hist_DoA.xmd'), True)
        self.createHistrogram(self._getExtraPath(OUTPUT_DOA2_FILE), self._getExtraPath('hist_DoA2.xmd'), False)
        self.createHistrogram(self._getExtraPath(OUTPUT_RADIAL_FILE), self._getExtraPath('hist_radial.xmd'), False)
        self.createHistrogram(self._getExtraPath(OUTPUT_AZIMUTHAL_FILE), self._getExtraPath('hist_azimuthal.xmd'), False)
        
    def createOutputStep(self):
        
        volume=Volume()
        volume.setFileName(self._getExtraPath(OUTPUT_DOA1_FILE))
        volume.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(outputVolume_doa=volume)
        self._defineSourceRelation(self.inputVolumes, volume)
        
        volume.setFileName(self._getExtraPath(OUTPUT_AZIMUTHAL_FILE))
        volume.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(azimuthalVolume=volume)
        self._defineSourceRelation(self.inputVolumes, volume)

        volume.setFileName(self._getExtraPath(OUTPUT_RADIAL_FILE))
        volume.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(radialVolume=volume)
        self._defineSourceRelation(self.inputVolumes, volume)        

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'doaVol'):
            messages.append(
                'Information about the method/article in ' + MONORES_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []

        return summary

    def _citations(self):
        return ['Not yet']
