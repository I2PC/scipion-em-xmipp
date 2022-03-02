# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
# *              Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
Protocol to split a volume in two volumes based on a set of images
"""

from numpy import average
from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL
from pyworkflow.protocol.params import PointerParam, FloatParam, IntParam, StringParam, BooleanParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range
from pyworkflow.utils.path import copyFile, makePath, createLink, cleanPattern, cleanPath, moveFile

from pwem.protocols import ProtClassify3D
from pwem.objects import Volume
from pwem.constants import (ALIGN_NONE, ALIGN_2D, ALIGN_3D, ALIGN_PROJ)

from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes, readSetOfVolumes

import random as rand

class XmippProtSplitvolume(ProtClassify3D):
    """Split volume in two"""
    _label = 'split volume'
    _devStatus = BETA
    
    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL
        self._createFilenames()

    def _createFilenames(self):
        """ Centralize the names of the files. """
        recFmt='rec%(rec)06d'

        myDict = {
            'average_root': f'average/',
            'average_particles': f'average/directional_classes.xmd',
            'average_volume': f'average/volume.vol',
            'reconstruction_root': f'{recFmt}/', 
            'reconstruction_particles': f'{recFmt}/directional_classes.xmd',
            'reconstruction_volume': f'{recFmt}/volume.vol',
            'volume_pca_root': f'volume_pca/',
            'volume_pca_md': f'volume_pca/volumes.xmd',
            'volume_pca_projections': f'volume_pca/projections.xmd',
            'volume_pca_percentiles': f'volume_pca/percentiles.vol',
            'volume_pca_extremes': f'volume_pca/extremes.vol',
            'volume_pca_basis': f'volume_pca/basis.vol',
        }
        self._updateFilenamesDict(myDict)
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('directionalClasses', PointerParam, label="Directional classes", 
                      pointerClass='SetOfAverages', pointerCondition='hasAlignmentProj',
                      important=True, 
                      help='Select a set of particles with angles. Preferrably the output of a run of directional classes')
        form.addParam('averageVolume', PointerParam, label="Average volume", 
                      pointerClass='Volume', important=True, 
                      help='Volume used as the coordinate origin of the PCA analysis')
        form.addParam('symmetryGroup', StringParam, label="Symmetry group",
                      default='c1',
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp') 
        form.addParam('mask', PointerParam, label="Mask", pointerClass='VolumeMask', allowsNull=True,
                      help='The mask values must be binary: 0 (remove these voxels) and 1 (let them pass).')

        form.addSection(label='Reconstruction')
        form.addParam('reconstructionCount', IntParam, label="Number of reconstructions", 
                      default=5000, validators=[GE(2)], expertLevel=LEVEL_ADVANCED, 
                      help="Number of random reconstructions to perform")
        form.addParam('reconstructionSamples', IntParam, label="Number of images per reconstruction", 
                      default=15, validators=[GT(0)], expertLevel=LEVEL_ADVANCED, 
                      help="Number of images per reconstruction. Consider that reconstructions with symmetry c1 will be performed")
        form.addParam('reconstructionMaxResolution', FloatParam, label="Maximum resolution", 
                      default=0.25, validators=[Range(0, 1)], expertLevel=LEVEL_ADVANCED, 
                      help="Maximum resolution in terms of the Nyquist frequency")

        form.addSection(label='Volumetric PCA Splitting')
        form.addParam('volumePca_enable', BooleanParam, label='Enable', default=True)
        form.addParam('volumePca_alpha', FloatParam, label="Confidence level (%)", 
                      default=90, validators=[Range(50, 100)], expertLevel=LEVEL_ADVANCED, 
                      help="This parameter is alpha. Two volumes, one at alpha/2 and another one at 1-alpha/2, will be generated")
        form.addParam('volumePca_beta', FloatParam, label="Extreme volume separation", 
                      default=10, validators=[GE(0.0)], expertLevel=LEVEL_ADVANCED, 
                      help="This parameter defines how far extreme volumes are generated")

        form.addParallelSection()
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        nRec = self.reconstructionCount.get()

        # Start converting the input
        convertInputIdx = self._insertFunctionStep('convertInputStep')

        # Generate reconstructions in parallel with a random particle subset
        reconstructSteps = []
        for i in range(nRec):
            reconstructSteps.append(self._insertFunctionStep('generateVolumeStep', i, prerequisites=[convertInputIdx]))

        # Compare the volumes to select the most different ones
        if self.volumePca_enable.get():
            self._insertFunctionStep('volumePcaStep', nRec)
        
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        self._createParticleList()

    def generateAverageVolumeStep(self):
        self._createAverageWorkingDir()
        fnParticles = self._getExtraPath(self._getFileName('average_particles'))
        fnVolume = self._getExtraPath(self._getFileName('average_volume'))
        sym = self.symmetryGroup.get()
        maxRes = self.reconstructionMaxResolution.get()
        
        # Write a metadata file with all the particles
        particles = self.directionalClasses.get()
        writeSetOfParticles(particles, fnParticles)

        # Execute the reconstruction job
        command='xmipp_reconstruct_fourier'
        args  = f'-i {fnParticles} '
        args += f'-o {fnVolume} '
        args += f'--max_resolution {maxRes} '
        args += f'--sym {sym} '
        args += f'-v 0'
        self.runJob(command, args)
    
    def generateVolumeStep(self, rec):
        self._createReconstructionWorkingDir(rec)
        fnParticles = self._getExtraPath(self._getFileName('reconstruction_particles', rec=rec))
        fnVolume = self._getExtraPath(self._getFileName('reconstruction_volume', rec=rec))
        maxRes = self.reconstructionMaxResolution.get()
        
        # Write a metadata file with a random subset of particles
        particles = self.directionalClasses.get()
        nSamples = self.reconstructionSamples.get()
        subset = rand.sample(self.particleList, k=nSamples)
        writeSetOfParticles(subset, fnParticles, alignType=particles.getAlignment())

        # Execute the reconstruction job
        command='xmipp_reconstruct_fourier'
        args  = f'-i {fnParticles} '
        args += f'-o {fnVolume} '
        args += f'--max_resolution {maxRes} '
        args += f'-v 0'
        self.runJob(command, args)

    def volumePcaStep(self, nRec):
        self._createVolumePcaWorkingDir()
        alpha = self.volumePca_alpha.get()
        beta = self.volumePca_beta.get()
        fnAverageVolume = self.averageVolume.get().getFileName()
        fnMask = self.mask.get().getFileName() if self.mask.get() else None
        fnsVolumes = [self._getExtraPath(self._getFileName('reconstruction_volume', rec=rec)) for rec in range(nRec)]
        fnVolumesMd = self._getExtraPath(self._getFileName('volume_pca_md'))
        fnOutProjections = self._getExtraPath(self._getFileName('volume_pca_projections'))
        fnOutExtremes = self._getExtraPath(self._getFileName('volume_pca_extremes'))
        fnOutPca = self._getExtraPath(self._getFileName('volume_pca_percentiles'))
        fnOutBasis = self._getExtraPath(self._getFileName('volume_pca_basis'))

        # Index all volumes into the metadata file
        writeSetOfVolumes(
            [Volume(fn, objId=i) for i, fn in enumerate(fnsVolumes)], 
            fnVolumesMd, 
            alignType=ALIGN_NONE
        )

        # Execute the analysis job
        command='xmipp_volume_pca'
        args  = f'-i {fnVolumesMd} '
        args += f'-o {fnOutProjections} '
        args += f'--avgVolume {fnAverageVolume} '
        args += f'--saveBasis {fnOutBasis} '
        args += f'--generatePCAVolumes {alpha} {100-alpha} '
        args += f'--opca {fnOutPca} '
        args += f'--generateExtremeVolumes {beta} '
        args += f'--oext {fnOutExtremes} '
        if fnMask:
            args += f'--mask binary_file {fnMask} '
        args += f'-v 0'
        self.runJob(command, args)

    def createOutputStep(self):
        outputs = []
        sources = [self.directionalClasses]

        # Only add the mask if defined
        if self.mask.get():
            sources.append(self.mask)

        # Generate requested outputs
        if self.volumePca_enable.get():
            outputs.append(self._createVolumePcaPercentilesOutput())
            outputs.append(self._createVolumePcaExtremesOutput())

        # Define source-output relations
        for output in outputs:
            for source in sources:
                self._defineSourceRelation(source, output)

        
    #--------------------------- UTILS functions ---------------------------------------------------
    def _createParticleList(self):
        """ Concurrent iteration of SetOfParticles is problematic.
            Therefore create a list with clones of its elements 
        """        
        result = []

        for particle in self.directionalClasses.get():
            result.append(particle.clone())

        self.particleList = result
    
    def _createAverageWorkingDir(self):
        root = self._getExtraPath(self._getFileName('average_root'))
        makePath(root)
        return root

    def _createReconstructionWorkingDir(self, rec):
        root = self._getExtraPath(self._getFileName('reconstruction_root', rec=rec))
        makePath(root)
        return root
    
    def _createVolumePcaWorkingDir(self):
        root = self._getExtraPath(self._getFileName('volume_pca_root'))
        makePath(root)
        return root

    def _createVolumePcaPercentilesOutput(self):
        assert(self.volumePca_enable.get())
        volumes = self._createSetOfVolumes('volumePcaPercentiles')
        volumes.copyInfo(self.directionalClasses.get())

        for i in range(2):
            path = self._getExtraPath(self._getFileName('volume_pca_percentiles'))
            loc = (i+1, path)
            vol = Volume(loc)
            volumes.append(vol)

        self._defineOutputs(outputVolumetricPcaPercentiles=volumes)
        return volumes
    
    def _createVolumePcaExtremesOutput(self):
        assert(self.volumePca_enable.get())
        volumes = self._createSetOfVolumes('volumePcaExtremes')
        volumes.copyInfo(self.directionalClasses.get())

        for i in range(2):
            path = self._getExtraPath(self._getFileName('volume_pca_extremes'))
            loc = (i+1, path)
            vol = Volume(loc)
            volumes.append(vol)

        self._defineOutputs(outputVolumetricPcaExtremes=volumes)
        return volumes