# **************************************************************************
# *
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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


from pwem import emlib
from pwem.protocols import ProtClassify3D
from pwem.objects import (Class3D, SetOfParticles, SetOfClasses3D, Particle,
                          Volume, SetOfVolumes )

from pyworkflow import BETA
import pyworkflow.utils as pwutils
from pyworkflow.protocol.params import (Form, PointerParam, FloatParam,
                                        IntParam, StringParam, BooleanParam,
                                        GT,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )

import xmipp3
from xmipp3.convert import writeSetOfParticles

import pickle
import numpy as np
import scipy.stats
import sklearn.mixture

class XmippProtHetReconstruct(ProtClassify3D, xmipp3.XmippProtocol):
    OUTPUT_CLASSES_NAME = 'classes'
    OUTPUT_VOLUMES_NAME = 'volumes'
    
    _label = 'heterogeneous reconstruct'
    _devStatus = BETA
    _possibleOutputs = {
        OUTPUT_CLASSES_NAME: SetOfClasses3D,
        OUTPUT_VOLUMES_NAME: SetOfVolumes
    }

    def __init__(self, *args, **kwargs):
        ProtClassify3D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form: Form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label='Particles', important=True,
                      pointerClass=SetOfParticles, pointerCondition='hasAlignmentProj',
                      help='Input particle set')
        form.addParam('symmetryGroup', StringParam, label='Symmetry group', default='c1')
        form.addParam('resolution', FloatParam, label='Resolution', default=8)
        form.addSection(label='Classification')
        form.addParam('classCount', IntParam, label='Class count',
                      default=2, validators=[GT(0)],
                      help='Number of classes to be reconstructed')
        form.addParam('principalComponents', StringParam, label='Principal components',
                      expertLevel=LEVEL_ADVANCED,
                      help='Principal components used in evaluation')
        
    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('classifyStep')
    
        for i in range(self.classCount.get()):
            self._insertFunctionStep('reconstructStep', i+1)

        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions -------------------------------
    def convertInputStep(self):
        inputParticles = self._getInputParticles()
        
        EXTRA_LABELS = [
            emlib.MDL_DIMRED
        ]
        
        writeSetOfParticles(
            inputParticles, 
            self._getInputParticleMdFilename(),
            extraLabels=EXTRA_LABELS
        )
    
    def classifyStep(self):
        classificationMd = emlib.MetaData(self._getInputParticleMdFilename())
        projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
        
        principalComponents = self._getPrincipalComponents()
        writeProjections = len(principalComponents) > 0
        if writeProjections:
            projections = projections[:,principalComponents]
        
        gmm = sklearn.mixture.GaussianMixture(n_components=self.classCount.get())
        ref3d = gmm.fit_predict(projections) + 1
        
        if writeProjections:
            classificationMd.setColumnValues(emlib.MDL_DIMRED, projections.tolist())
        classificationMd.setColumnValues(emlib.MDL_REF3D, ref3d.tolist())
        classificationMd.write(self._getClassificationMdFilename())
        self._writeGaussianMixtureModel(gmm)
    
    def reconstructStep(self, cls):
        # Select particles
        classification = emlib.MetaData(self._getClassificationMdFilename())
        classification.removeObjects(emlib.MDValueNE(emlib.MDL_REF3D, cls))
        classification.write(self._getClassMdFilename(cls))
        
        maxResolution = self._getSamplingRate() / self.resolution.get()
        
        args = []
        args += ['-i', self._getClassMdFilename(cls)]
        args += ['-o', self._getClassVolumeFilename(cls)]
        args += ['--sym', self._getSymmetryGroup()]
        args += ['--max_resolution', maxResolution]
        
        # Determine the execution parameters
        numberOfMpi = self.numberOfMpi.get()
        if self.useGpu.get():
            reconstructProgram = 'xmipp_cuda_reconstruct_fourier'

            gpuList = self.getGpuList()
            if self.numberOfMpi.get() > 1:
                numberOfGpus = len(gpuList)
                numberOfMpi = numberOfGpus + 1
                args += ['-gpusPerNode', numberOfGpus]
                args += ['-threadsPerGPU', max(self.numberOfThreads.get(), 4)]
            else:
                args += ['--device', ','.join(map(str, gpuList))]
                
            """
            count=0
            GpuListCuda=''
            if self.useQueueForSteps() or self.useQueue():
                GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                GpuList = GpuList.split(",")
                for elem in GpuList:
                    GpuListCuda = GpuListCuda+str(count)+' '
                    count+=1
            else:
                GpuListAux = ''
                for elem in self.getGpuList():
                    GpuListCuda = GpuListCuda+str(count)+' '
                    GpuListAux = GpuListAux+str(elem)+','
                    count+=1
                os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux
            """
                
            args += ['--thr', self.numberOfThreads.get()]
        
        else:
            reconstructProgram = 'xmipp_reconstruct_fourier_accel'
        
        # Run
        self.runJob(reconstructProgram, args, numberOfMpi=numberOfMpi)
        
    def createOutputStep(self):
        classificationMd = emlib.MetaData(self._getClassificationMdFilename())
        
        # Create volumes
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(self._getSamplingRate())
        for i in range(self.classCount.get()):
            objId = i+1
            volumeFilename = self._getClassVolumeFilename(objId)
            volume = Volume(objId=objId)
            volume.setFileName(volumeFilename)
            volumes.append(volume)
        
        # Classify particles
        def updateItem(item: Particle, row: emlib.metadata.Row):
            assert(item.getObjId() == row.getValue(emlib.MDL_ITEM_ID))
            item.setClassId(row.getValue(emlib.MDL_REF3D))
            
        def updateClass(cls: Class3D):
            clsId = cls.getObjId()
            representative = volumes[clsId].clone()
            cls.setRepresentative(representative)
            cls.setAlignmentProj()
            
        classes3d: SetOfClasses3D = self._createSetOfClasses3D(self.inputParticles)
        classes3d.classifyItems(
            updateItemCallback=updateItem,
            updateClassCallback=updateClass,
            itemDataIterator=emlib.metadata.iterRows(classificationMd)
        )
        
        # Define the output
        self._defineOutputs(**{self.OUTPUT_CLASSES_NAME: classes3d})
        self._defineOutputs(**{self.OUTPUT_VOLUMES_NAME: volumes})
        self._defineSourceRelation(self.inputParticles, classes3d)
        self._defineSourceRelation(self.inputParticles, volumes)


        
    # --------------------------- UTILS functions -------------------------------
    def _getDeviceList(self):
        gpus = self.getGpuList()
        return list(map('cuda:{:d}'.format, gpus))
    
    def _getInputParticles(self) -> SetOfParticles:
        return self.inputParticles.get()
    
    def _getSamplingRate(self):
        return float(self.inputParticles.get().getSamplingRate())
    
    def _getSymmetryGroup(self):
        return self.symmetryGroup.get()
    
    def _getPrincipalComponents(self):
        if self.principalComponents.get():
            return pwutils.getListFromRangeString(self.principalComponents.get())
        else:
            return []
            
    def _getInputParticleMdFilename(self):
        return self._getPath('input_particles.xmd')
    
    def _getInputParticleStackFilename(self):
        return self._getTmpPath('input_particles.mrc')
   
    def _getWienerParticleMdFilename(self):
        if self.considerInputCtf:
            return self._getExtraPath('particles_wiener.xmd')
        else:
            return self._getInputParticleMdFilename()

    def _getWienerParticleStackFilename(self):
        return self._getTmpPath('particles_wiener.mrc')

    def _getClassificationMdFilename(self):
        return self._getExtraPath('classification.xmd')
    
    def _getClassMdFilename(self, cls: int):
        return self._getExtraPath('class_%02d.xmd' % cls)

    def _getClassVolumeFilename(self, cls: int):
        return self._getExtraPath('class_%02d.mrc' % cls)

    def _getGaussianMixtureModelFilename(self) -> str:
        return self._getExtraPath('gmm.pkl')

    def _writeGaussianMixtureModel(self, gmm: sklearn.mixture.GaussianMixture):
        with open(self._getGaussianMixtureModelFilename(), 'wb') as f:
            pickle.dump(gmm, f)
        
    def _readGaussianMixtureModel(self):
        with open(self._getGaussianMixtureModelFilename(), 'rb') as f:
            return pickle.load(f)
