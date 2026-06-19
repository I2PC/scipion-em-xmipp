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
                                        EnumParam, GT,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )

import xmipp3
from xmipp3.convert import writeSetOfParticles

import numpy as np
import sklearn.neighbors
import matplotlib.pyplot as plt


class XmippProtHetReconstructInteractive(ProtClassify3D, xmipp3.XmippProtocol):
    OUTPUT_CLASSES_NAME = 'classes'
    OUTPUT_VOLUMES_NAME = 'volumes'
    
    _label = 'heterogeneous reconstruct interactive'
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
        
    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('interactiveStep')
        self._insertFunctionStep('classifyStep')
        self._insertFunctionStep('reconstructStep')
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
        
    def interactiveStep(self):
        fig, ax = plt.subplots()
        
        particlesMd = emlib.MetaData(self._getInputParticleMdFilename())
        points = np.array(particlesMd.getColumnValues(emlib.MDL_DIMRED))
        
        ax.hist2d(points[:,0], points[:,1], bins=96)
        
        centroids = []
        scatter = ax.scatter([], [], picker=True, c='white')
        
        def onClick(event):
            if event.inaxes == ax and event.button == 1:
                centroids.append((event.xdata, event.ydata))
                scatter.set_offsets(centroids)
                fig.canvas.draw()
        
        def onPick(event):
            if event.artist == scatter and event.mouseevent.button == 3:
                ind = event.ind[0]
                centroids.pop(ind)
                scatter.set_offsets(centroids)
                fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', onClick)
        fig.canvas.mpl_connect('pick_event', onPick)
        
        plt.show()
        self._writeCentroids(np.array(centroids))
    
    def classifyStep(self):
        classificationMd = emlib.MetaData(self._getInputParticleMdFilename())
        points = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
        centroids = self._readCentroids()
        
        classifier = sklearn.neighbors.KDTree(centroids)
        distance, ref3d = classifier.query(points)
        ref3d = ref3d[:,0]+1
        distance = distance[:,0]

        classificationMd.setColumnValues(emlib.MDL_REF3D, ref3d.tolist())
        classificationMd.write(self._getClassificationMdFilename())
        
    def reconstructStep(self):
        centroids = self._readCentroids()
        for i, _ in enumerate(centroids, start=1):
            # Select particles
            classification = emlib.MetaData(self._getClassificationMdFilename())
            classification.removeObjects(emlib.MDValueNE(emlib.MDL_REF3D, i))
            classification.write(self._getClassMdFilename(i))
            
            maxResolution = self._getSamplingRate() / self.resolution.get()
            
            args = []
            args += ['-i', self._getClassMdFilename(i)]
            args += ['-o', self._getClassVolumeFilename(i)]
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
        centroids = self._readCentroids()
        classificationMd = emlib.MetaData(self._getClassificationMdFilename())
        
        # Create volumes
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(self._getSamplingRate())
        for i, _ in enumerate(centroids, start=1):
            objId = i
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

    def _getCentroidsFilename(self):
        return self._getExtraPath('points.npy')
    
    def _readCentroids(self) -> np.ndarray:
        return np.load(self._getCentroidsFilename())
    
    def _writeCentroids(self, points: np.ndarray):
        np.save(self._getCentroidsFilename(), points)
        