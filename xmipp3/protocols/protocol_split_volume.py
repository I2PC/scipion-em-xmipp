# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
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

from asyncore import write
from re import sub
from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL
from pyworkflow.protocol.params import PointerParam, FloatParam, IntParam, StringParam, BooleanParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range
from pyworkflow.utils.path import copyFile, makePath, createLink, cleanPattern, cleanPath, moveFile

from pwem.protocols import ProtClassify3D
from pwem.objects import Volume
from pwem.constants import (ALIGN_NONE, ALIGN_2D, ALIGN_3D, ALIGN_PROJ)

from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes

import random as rand

class XmippProtSplitvolume(ProtClassify3D):
    """Split volume in two"""
    _label = 'split volume'
    _devStatus = BETA
    
    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)
        #self.stepsExecutionMode = STEPS_PARALLEL
        self._createFilenames()

    def _createFilenames(self):
        """ Centralize the names of the files. """
        recFmt='rec%(rec)05d'

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
        }
        self._updateFilenamesDict(myDict)
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('directionalClasses', PointerParam, label="Directional classes", 
                      pointerClass='SetOfAverages', pointerCondition='hasAlignmentProj',
                      important=True, 
                      help='Select a set of particles with angles. Preferrably the output of a run of directional classes')
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

        form.addParallelSection()
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        nRec = self.reconstructionCount.get()

        reconstructSteps = []

        # Reconstruct the average volume with all the particles
        self._insertFunctionStep('generateAverageVolumeStep', prerequisites=[])
        reconstructSteps.append(len(self._steps))

        # Generate reconstructions in parallel with a random particle subset
        for i in range(nRec):
            self._insertFunctionStep('generateVolumeStep', i, prerequisites=[])
            reconstructSteps.append(len(self._steps))

        # Compare the volumes to select the most different ones
        if self.volumePca_enable.get():
            self._insertFunctionStep('volumePcaStep', nRec)
        
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
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
        args = f'-i {fnParticles} -o {fnVolume} --max_resolution {maxRes} --sym {sym} -v 0'
        self.runJob(command, args)
    
    def generateVolumeStep(self, rec):
        self._createReconstructionWorkingDir(rec)
        fnParticles = self._getExtraPath(self._getFileName('reconstruction_particles', rec=rec))
        fnVolume = self._getExtraPath(self._getFileName('reconstruction_volume', rec=rec))
        maxRes = self.reconstructionMaxResolution.get()
        
        # Write a metadata file with a random subset of particles
        particles = self.directionalClasses.get()
        nSamples = self.reconstructionSamples.get()
        subset = self._createRandomParticleSubset(particles, nSamples)
        writeSetOfParticles(subset, fnParticles, alignType=particles.getAlignment())

        # Execute the reconstruction job
        command='xmipp_reconstruct_fourier'
        args = f'-i {fnParticles} -o {fnVolume} --max_resolution {maxRes} -v 0'
        self.runJob(command, args)

    def volumePcaStep(self, nRec):
        self._createVolumePcaWorkingDir()
        fnAverageVolume = self._getExtraPath(self._getFileName('average_volume'))
        fnsVolumes = [self._getExtraPath(self._getFileName('reconstruction_volume', rec=rec)) for rec in range(nRec)]
        fnVolumesMd = self._getExtraPath(self._getFileName('volume_pca_md'))
        fnProjections = self._getExtraPath(self._getFileName('volume_pca_projections'))
        fnOutPca = self._getExtraPath(self._getFileName('volume_pca_percentiles'))
        alpha = self.volumePca_alpha.get()

        # Index all volumes into the metadata file
        writeSetOfVolumes(
            [Volume(fn, objId=i) for i, fn in enumerate(fnsVolumes)], 
            fnVolumesMd, 
            alignType=ALIGN_NONE
        )

        # Execute the analysis job
        command='xmipp_volume_pca'
        args = f'-i {fnVolumesMd} -o {fnProjections} --generatePCAVolumes {alpha} {1-alpha} --opca {fnOutPca} --avgVolume {fnAverageVolume} -v 0'
        self.runJob(command, args)

    def createOutputStep(self):
        outputs = []
        sources = [self.directionalClasses]

        # Generate requested outputs
        if self.volumePca_enable.get():
            outputs.append(self._createVolumePcaOutput())

        # Define source-output relations
        for output in outputs:
            for source in sources:
                self._defineSourceRelation(source, output)

        
    #--------------------------- UTILS functions ---------------------------------------------------
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

    def _createRandomParticleSubset(self, particles, nSamples):
        result = []

        # Create a random selection mask (nSample Trues and False for the rest)
        particleMask = [True]*nSamples + [False]*(len(particles)-nSamples)
        rand.shuffle(particleMask)

        # Add the selected items to the result
        assert(len(particles) == len(particleMask))
        for particle, selected in zip(particles, particleMask):
            if selected:
                result.append(particle.clone())

        return result

    def _createVolumePcaOutput(self):
        assert(self.volumePca_enable.get())
        volumes = self._createSetOfVolumes('volumePca')
        volumes.copyInfo(self.directionalClasses.get())

        for i in range(2):
            path = self._getExtraPath(self._getFileName('volume_pca_percentiles'))
            loc = (i+1, path)
            vol = Volume(loc)
            volumes.append(vol)

        self._defineOutputs(outputVolumetricPca=volumes)
        return volumes
