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
from pwem.protocols import ProtAnalysis2D
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED, LT, GT)
from xmipp3 import XmippProtocol

FN = "class_representatives"
RESULT_FILE = 'best_clusters_with_names.txt'


class XmippProtCL2DClustering(ProtAnalysis2D, XmippProtocol):
    """ 2D clustering protocol to group similar classes """

    _label = '2D classes clustering'
    _conda_env = 'xmipp_cl2dClustering'

    def __init__(self, **args):
        ProtAnalysis2D.__init__(self, **args)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', PointerParam,
                      label="Input 2D classes",
                      important=True, pointerClass='SetOfClasses2D',
                      help='Select the input classes to be mapped.')
        form.addParam('min_cluster', IntParam, label='Minimum number of clusters',
                      default=10, expertLevel=LEVEL_ADVANCED, validators=[GT(1, 'Error must be greater than 1')],
                      help=''' By default, the 2D averages will start searching for the optimum number of clusters '''
                           ''' with a minimum number of 10 classes.''')

        form.addParam('max_cluster', IntParam, label='Maximum number of clusters',
                      default=-1, expertLevel=LEVEL_ADVANCED,
                      validators=[LT(50, 'Error must be smaller than the number of classes - 2.')],
                      help=''' By default, the 2D averages will end searching for the optimum number of clusters '''
                           ''' until a maximum number of N classes - 2. If -1 then it will act as default.''')

        form.addParam('compute_threads', IntParam, label='Number of computational threads',
                      default=8, expertLevel=LEVEL_ADVANCED,
                      validators=[
                          GT(0, 'Error must be greater than 0.')],
                      help=''' By default, the program will use 8 threads for computation.''')

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

        inputClasses = self.inputClasses.get()

        classes_refIds = []
        for rep in inputClasses.iterRepresentatives():
            idClass, fn = rep.getLocation()
            classes_refIds.append(idClass)

        # Save the corresponding .mrcs file
        inputClasses.writeStack(self.imgsFn)
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
        print('New classes')
        classes2DSet = self._createSetOfClasses2D(self.inputClasses.get().getImages())
        result_dict_file = os.path.join(self.directoryPath, RESULT_FILE)
        result_dict = self.read_clusters_from_txt(result_dict_file)
        # self._fillClassesFromLevel(classes2DSet)
        result = {'outputClasses': classes2DSet}
        self._defineOutputs(**result)

    # ------------------------ Utils -----------------------------
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