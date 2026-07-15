# **************************************************************************
# *
# * Authors:     
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


from pathlib import Path

import emtable
import pwem
from pyworkflow import VERSION_3_0
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        IntParam, BooleanParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils import Message
from pyworkflow.utils.path import createLink
from pwem.protocols import ProtAnalysis2D, ProtFlexBase
from pwem.objects import SetOfClasses2D, SetOfParticlesFlex, ParticleFlex, String
from xmipp3.convert import readSetOfParticles, writeSetOfParticles, readSetOfClasses2D
import os
import xmipp3
from pyworkflow import BETA, UPDATED, NEW, PROD
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow import Config

import numpy as np

def getJaxXmippEnvActivation():
    """ Remove the scipion home and activate the conda environment. """
    activation = "conda activate xmipp_jax"
    scipionHome = Config.SCIPION_HOME + os.path.sep

    return activation.replace(scipionHome, "", 1)


def getJaxXmippActivationCommand():
    """ Return the activation command. """
    return '%s %s' % (
        xmipp3.Plugin.getCondaActivationCmd(),
        getJaxXmippEnvActivation()
    )

class XmippProtClusterFlexParticles(ProtAnalysis2D, ProtFlexBase, xmipp3.XmippProtocol):
    """Train a rotational and shift invariant embedding for images"""
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_jax'
    _label = 'cluster flex particles'
    _devStatus = NEW

    def __init__(self, **args):
        ProtAnalysis2D.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addParallelSection(threads=1, mpi=4)

        form.addSection(label=Message.LABEL_INPUT)

        form.addParam(
            'inputParticles',
            PointerParam,
            label="Input images",
            pointerClass='SetOfParticlesFlex',
            help='Set of particles to cluster'
        )
        
        form.addParam(
            'numClusters',
            IntParam,
            label='Clusters',
            help='Number of Clusters'
        )
        
        form.addSection(label='Compute')
        
    def _prepareInputs(self):
        particles: SetOfParticlesFlex = self.inputParticles.get()
        assert isinstance(particles, SetOfParticlesFlex)

        latents = []
        for particle in particles.iterItems():
            particle: ParticleFlex = particle
            assert isinstance(particle, ParticleFlex)

            latent = particle.getZFlex()
            latents.append(latent)

        latents = np.array(latents).astype(np.int32)
        np.savetxt(self.fnLatents, latents)

    def _outputStep(self):
        def __updateItem(item: ParticleFlex, row):
            class2D = int(row) + 1 # Index 0 cancels the iterations
            assert class2D != 0, "Class cannot be 0 as it cancels the loop"

            item.setClassId(class2D)


        classLabels = np.loadtxt(self.fnOutputs)
        
        outputSet: SetOfClasses2D = self._createSetOfClasses2D(imgSet=self.inputParticles.get())
        assert isinstance(outputSet, SetOfClasses2D)

        print(len(classLabels), len(outputSet))

        outputSet.classifyItems(
            updateItemCallback=__updateItem,
            updateClassCallback=None,
            itemDataIterator=iter(classLabels),
            doClone=False,
            raiseOnNextFailure=False,
        )

        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputParticles.get(), outputSet)


    def _kMeansStep(self):

        inputsPath = os.path.abspath(self.fnLatents)
        outputsPath = os.path.abspath(self.fnOutputs)

        assert os.path.exists(inputsPath)

        args = (
            "-i {inputs} --olabels {outputs} --embeddingK {k}"
                .format(
                    inputs=inputsPath,
                    outputs=outputsPath,
                    k=self.numClusters
                )
        )

        self.runJob(
            f"{getJaxXmippActivationCommand()} && xmipp_cluster_latents",
            args,
            numberOfMpi=1,
            env=self.getCondaEnv()
        )


    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.fnLatents = self._getExtraPath("z_space.txt")
        self.fnOutputs = self._getExtraPath("clusters.txt")

        self._insertFunctionStep(self._prepareInputs)
        self._insertFunctionStep(self._kMeansStep)
        self._insertFunctionStep(self._outputStep)
        


    