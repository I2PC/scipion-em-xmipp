# ******************************************************************************
# *
# * Authors:     Daniel Marchan Torres (da.marchan@cnb.csic.es)
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
# ******************************************************************************

import os.path
from pwem.objects.data import Class2D, Particle, SetOfClasses2D, SetOfAverages
from pwem.protocols import ProtAnalysis2D
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        EnumParam, LEVEL_ADVANCED, LT, GT)
from pyworkflow import NEW, BETA
from xmipp3 import XmippProtocol

FN = "class_representatives"
RESULT_FILE = 'best_clusters_with_names.txt'
OUTPUT_CLASSES = 'outputClasses'
OUTPUT_AVERAGES = 'outputAverages'


class XmippProtCL2DClustering(ProtAnalysis2D, XmippProtocol):
    """Groups similar 2D class averages using clustering. This process helps identify homogeneous subsets within the dataset, improving classification and downstream analysis. """

    _label = 'clustering 2d classes'
    _devStatus = BETA

    _possibleOutputs = {OUTPUT_CLASSES: SetOfClasses2D,
                        OUTPUT_AVERAGES: SetOfAverages}

    CLASSES = 0
    AVERAGES = 1
    BOTH = 2

    def __init__(self, **args):
        ProtAnalysis2D.__init__(self, **args)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet2D', PointerParam,
                      label="Input 2D images",
                      important=True, pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Select the input classes or input averages to be clustered.')
        form.addParam('min_cluster', IntParam, label='Minimum number of clusters',
                      default=10, expertLevel=LEVEL_ADVANCED,
                      validators=[GT(1, 'Error must be greater than 1')],
                      help=' This number will limit the search for the optimum number of clusters. '
                               'By default, the 2D averages will start searching for the optimum number of clusters'
                               'with a minimum number of 10 classes.')
        form.addParam('max_cluster', IntParam, label='Maximum number of clusters',
                      default=-1, expertLevel=LEVEL_ADVANCED,
                      validators=[LT(50, 'Error must be smaller than the number of classes - 2.')],
                      help='This number will limit the search for the optimum number of clusters. '
                              'If -1 then it will act as default. By default, the 2D averages will end searching'
                              'for the optimum number of clusters until a maximum number of N_classes - 2.')
        form.addParam('compute_threads', IntParam, label='Number of computational threads',
                      default=8, expertLevel=LEVEL_ADVANCED,
                      validators=[
                          GT(0, 'Error must be greater than 0.')],
                      help=' By default, the program will use 8 threads for computation.'
                               'The higher the number the fastest the computation will be')

        form.addSection(label='Output')
        form.addParam('extractOption', EnumParam,
                      choices=['Classes', 'Averages', 'Both'],
                      default=self.CLASSES,
                      label="Extraction option", display=EnumParam.DISPLAY_COMBO,
                      help='Select an option to extract from the 2D Classes: \n '
                           '_Classes_: Create a new set of 2D classes with the respective cluster distribution. \n '
                           '_Averages_: Extract the representatives of each cluster. This are the most representative averages. \n'
                           '_Both_: Create a new set of 2D classes and extract their representatives.')


    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        convertStep = self._insertFunctionStep(self.convertStep)
        clusterStep = self._insertFunctionStep(self.clusterClasses, prerequisites=convertStep)
        self._insertFunctionStep(self.createOutputStep, prerequisites=clusterStep)


    def convertStep(self):
        self.info('Writting class representatives')
        self.directoryPath = self._getExtraPath()
        self.imgsFn = os.path.join(self.directoryPath, FN + ".mrcs")
        self.refIdsFn = os.path.join(self.directoryPath, FN + ".txt")

        inputSet2D = self.inputSet2D.get()
        classes_refIds = []

        if isinstance(inputSet2D, SetOfClasses2D):
            self.samplingRate = inputSet2D.getFirstItem().getRepresentative().getSamplingRate()
            for rep in inputSet2D.iterRepresentatives():
                idClass, _ = rep.getLocation()
                classes_refIds.append(idClass)
        else: # In case the input is a SetOfAverages
            self.samplingRate = inputSet2D.getSamplingRate()
            for rep in inputSet2D.iterItems():
                idClass, _ = rep.getLocation()
                classes_refIds.append(idClass)

        # Save the corresponding .mrcs file
        inputSet2D.writeStack(self.imgsFn) # The same method for SetOfClasses and SetOfAverages
        # Save the original ref ids
        with open(self.refIdsFn, "w") as file:
            for item in classes_refIds:
                file.write(f"{item}\n")


    def clusterClasses(self):
        'xmipp_cl2d_clustering -i path/to/inputAverages.mrcs -o path/to/outputDir -m 10 -M 20 -j 8'

        min_cluster = self.min_cluster.get()
        max_cluster = self.max_cluster.get()
        compute_threads = self.compute_threads.get()

        args = (" -i %s -o %s -m %d -M %d -j %d"
                %(self.imgsFn, self.directoryPath, min_cluster, max_cluster, compute_threads))

        self.runJob("xmipp_cl2d_clustering", args)


    def createOutputStep(self):
        output_dict = {}
        inputSet2D = self.inputSet2D.get()
        inputSet2D.loadAllProperties()

        result_dict_file = os.path.join(self.directoryPath, RESULT_FILE)
        result_dict = self.read_clusters_from_txt(result_dict_file)

        message = ("Classify original input set of %d images into %d groups of structural different images"
                   % (self.inputSet2D.get().getSize(), len(result_dict)))

        self.summaryVar.set(message)

        if self.extractOption.get() == self.CLASSES or self.extractOption.get() == self.BOTH:
            output_dict = self.createOutputSetOfClasses(inputSet2D, result_dict, output_dict)

        if self.extractOption.get() == self.AVERAGES or self.extractOption.get() == self.BOTH:
            output_dict = self.createOutputSetOfAverages(inputSet2D, result_dict, output_dict)

        self._defineOutputs(**output_dict)

        if self.extractOption.get() == self.CLASSES or self.extractOption.get() == self.BOTH:
            self._defineSourceRelation(inputSet2D.getImagesPointer(), output_dict[OUTPUT_CLASSES])
        if self.extractOption.get() == self.AVERAGES or self.extractOption.get() == self.BOTH:
            self._defineSourceRelation(self.inputSet2D, output_dict[OUTPUT_AVERAGES])

        self._store()


    def createOutputSetOfAverages(self, inputSet2D, result_dict, output_dict):
        outputRefs = self._createSetOfAverages()  # We need to create always an empty set since we need to rebuild it

        for cluster, classesRef in result_dict.items():
            self.info('For cluster %d' % cluster)
            self.info('We have the following ref classes: %s' % classesRef)
            firstTime = True
            for classRef in classesRef:
                if firstTime: # Just want to get the first ref
                    if isinstance(inputSet2D, SetOfClasses2D):
                        classTmp = inputSet2D.getItem("id", classRef).clone()
                        rep = classTmp.getRepresentative().clone()
                    else:
                        rep = inputSet2D.getItem("id", classRef).clone()
                        self.samplingRate = inputSet2D.getSamplingRate()
                    self.info('Using centroid to create new Average %s' % classRef)
                    newAvg = Particle()
                    newAvg.copyInfo(rep)
                    newAvg.setObjId(int(classRef))
                    newAvg.setClassId(int(classRef))
                    outputRefs.append(newAvg)
                    firstTime = False

        outputRefs.setSamplingRate(self.samplingRate)
        output_dict[OUTPUT_AVERAGES] = outputRefs

        return output_dict


    def createOutputSetOfClasses(self, inputSet2D, result_dict, output_dict):
        classes2DSet = self._createSetOfClasses2D(inputSet2D.getImagesPointer())
        dictClasses = {}

        for cluster, classesRef in result_dict.items():
            self.info('For cluster %d' % cluster)
            self.info('We have the following ref classes: %s' % classesRef)
            firstTime = True
            newParticles = []

            for classRef in classesRef:
                classTmp = inputSet2D.getItem("id", classRef)
                if firstTime:
                    self.info('First iter, using centroid to create new Class: %s' % classRef)
                    newClass = Class2D()
                    newClass.copyInfo(classTmp)
                    newClass.setObjId(int(classRef))
                    newClassId = newClass.getObjId()
                    firstTime = False

                for particle in classTmp.iterItems():
                    particle.setClassId(newClassId)
                    newParticles.append(particle.clone())

            dictClasses[newClassId] = newParticles
            self.info('Class particles size: %d' % len(newParticles))
            classes2DSet.append(newClass)

        for classId, particles in dictClasses.items():
            self.info('Filling Class with ID %d with %d particles' % (classId, len(particles)))
            class2D = classes2DSet[classId]
            class2D.enableAppend()
            for particle in particles:
                class2D.append(particle)

            classes2DSet.update(class2D)

        classes2DSet.write()

        output_dict[OUTPUT_CLASSES] = classes2DSet

        return output_dict


    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, OUTPUT_CLASSES) and not hasattr(self, OUTPUT_AVERAGES):
            summary.append("Output set not ready yet.")
        else:
            summary.append(self.summaryVar.get())

        return summary


    def _validate(self):
        errors = []
        if ((self.extractOption.get() == self.CLASSES or self.extractOption.get() == self.BOTH)
                and not isinstance(self.inputSet2D.get(), SetOfClasses2D)):
            errors.append("The input 2D must be a SetOfClasses2D to generate a SetOfClasses2D.")

        return errors

    # ------------------------------------ Utils ----------------------------------------------
    def read_clusters_from_txt(self, file_path):
        """
        Reads a cluster dictionary from a .txt file formatted as:
        Cluster 0:
            6
        Cluster 1:
            36
            34
            44
        ...

        Args:
            file_path (str): The path to the .txt file.

        Returns:
            dict: A dictionary where the key is the cluster number and the value is a list of associated numbers.
        """
        clusters = {}
        current_cluster = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("Cluster"):
                    # Extract the cluster number
                    current_cluster = int(line.split()[1].replace(':', ''))
                    clusters[current_cluster] = []
                elif current_cluster is not None and line.isdigit():
                    # Append numbers to the current cluster's list
                    clusters[current_cluster].append(line)

        return clusters

    # --------------------------------- Viewer functions ---------------------------------
    def getClusterPlot(self):
        return self._getExtraPath('best_cluster_visualization_with_images.png')


    def getClusterImagesPlot(self):
        return self._getExtraPath('all_clusters_with_labels.png')