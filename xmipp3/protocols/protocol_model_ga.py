# *****************************************************************************
# *
# * Authors:     David Herreros Calero         dherreros@cnb.csic.es
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
# *****************************************************************************

import numpy as np
import sys

from pwem.objects import Volume
from pwem.protocols import ProtAnalysis3D
from pwem.emlib.image import ImageHandler

import pyworkflow.protocol.params as params


class XmippProtModelGA(ProtAnalysis3D):
    """Modeling implemented through genetic algorithm"""
    _label = 'genetic algortihm modeling'

    # --------------------------- DEFINE param functions ---------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputMask', params.PointerParam, pointerClass='Volume', label='Input Masks',
                      important=True, help='Input mask with identifiers (not binary).')
        form.addParam('inputSeqs', params.MultiPointerParam, pointerClass='Sequence', label='Input Sequence', important=True,
                      help='Secuences to be modeled in the EM map (it can be a protein a chain...)')
        form.addParam('population', params.IntParam, label='Population Size', default=100, expertlevel=params.LEVEL_ADVANCED,
                      help='Number of indivuals in the population for the genetic algorithm')
        form.addParam('generations', params.IntParam, label='Number of Generations', default=5, expertlevel=params.LEVEL_ADVANCED,
                      help='Number of generations to be computed by the genetic algorithm')
        form.addParam('parents', params.IntParam, label='Number of Parents', default=5, expertlevel=params.LEVEL_ADVANCED,
                      help='Number of parents to be mated')
        form.addParam('p_mutation', params.FloatParam, label='Mutation Probability', default=0.3, expertlevel=params.LEVEL_ADVANCED,
                      help='Probability of introducing a mutation in an individual from the offspring')

    # --------------------------- INSERT steps functions ---------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('geneticAlgorithm')
        self._insertFunctionStep('createOutputStep')

    def geneticAlgorithm(self):
        # import time
        # time.sleep(10)
        ih = ImageHandler()
        self.seqs = [seq.get().getSequence() for seq in self.inputSeqs]
        self.seqs = np.asarray(self.seqs)
        self.idMask = ih.read(self.inputMask.get().getFileName()).getData()
        self.idMask = np.squeeze(self.idMask)

        sol_per_population = self.population.get()
        self.regions_id = np.unique(np.reshape(self.idMask, (1, -1)))
        self.regions_id = np.delete(self.regions_id, 0)
        num_regions = len(self.regions_id)
        pop_size = (sol_per_population, num_regions)
        new_population = np.random.random_integers(low=1, high=len(self.seqs), size=pop_size)
        num_generations = self.generations.get()
        num_parents = self.parents.get()
        cMat = self.connectivityMatrix()

        for generation in range(num_generations):
            print('Generation: ', (generation+1))
            score_population = self.scorePopulation(new_population)
            parents = self.matingPool(new_population, score_population, num_parents)
            offspring_size = (pop_size[0] - parents.shape[0], num_regions)
            offspring_crossover = self.crossover(new_population, offspring_size)
            offspring_mutation = self.mutation(offspring_crossover)
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation

            # FIXME: Probably this can be removed
            score_population = self.scorePopulation(new_population)
            print('Best result after generation %d: %f' % ((generation+1), np.amin(score_population)))
            sys.stdout.flush()

        print(new_population[score_population.argmin()])
        self.bestIndividual = new_population[score_population.argmin()]

    def createOutputStep(self):
        ih = ImageHandler()
        outMask = ih.createImage()
        outData = np.zeros(self.idMask.shape, float)
        for pos, idm in enumerate(self.regions_id):
            logic_mask = self.idMask == idm
            outData += self.bestIndividual[pos] * (self.idMask * logic_mask / idm)
        outMask.setData(outData)
        ih.write(outMask, self._getExtraPath('outMask.mrc'))
        volume = Volume()
        volume.setSamplingRate(self.inputMask.get().getSamplingRate())
        volume.setLocation(self._getExtraPath('outMask.mrc'))
        self._defineOutputs(outputMask=volume)
        self._defineSourceRelation(self.inputMask, volume)


    # --------------------------- DEFINE utils functions ----------------------
    def scorePopulation(self, population):
        mean_density_prot = 8.1325e-04  # KDa / (A^3)
        mean_mass_aa = 0.110  # KDa
        sampling_rate = self.inputMask.get().getSamplingRate() ** 3  # A^3 / voxel
        mean_density_prot *= sampling_rate

        submass = [mean_mass_aa * len(subseq) for subseq in self.seqs]
        submass = np.asarray(submass)

        map_region_mass = [mean_density_prot * np.sum(self.idMask[self.idMask == idr]) / idr for idr in self.regions_id]
        map_region_mass = np.asarray(map_region_mass)

        score_population = np.zeros(len(population))
        for idx in range((len(self.seqs)-1)):
            for idi, individual in enumerate(population):
                map_regions = np.where(individual == (idx + 1))
                score_population[idi] += np.abs(submass[idx] - np.sum(map_region_mass[map_regions]))

        return score_population

    def matingPool(self, population, score, num_parents):
        parents = np.empty((num_parents, population.shape[1]))

        for idp in range(num_parents):
            parents[idp, :] = population[np.argmin(score), :]
            score[np.argmin(score)] = np.inf

        return parents

    def crossover(self, population, offspring_size):
        offspring = np.empty(offspring_size)

        for k in range(offspring_size[0]):
            crossover_point = np.random.random_integers(0, offspring_size[1] - 1)
            parent1_idx = np.random.randint(0, population.shape[0])
            parent2_idx = np.random.randint(0, population.shape[0])
            offspring[k, 0:crossover_point] = population[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = population[parent2_idx, crossover_point:]

        return offspring

    def mutation(self, offspring):
        p_mutation = self.p_mutation.get()
        for idx in range(offspring.shape[0]):
            num_rnd = np.random.uniform()
            idg_rnd = np.random.random_integers(0, offspring.shape[1] - 1)
            chain_rnd = np.random.random_integers(1, len(self.seqs))
            if num_rnd <= p_mutation:
                offspring[idx, idg_rnd] = float(chain_rnd)

        return offspring

    def connectivityMatrix(self):
        num_regions = len(self.regions_id)
        cMat = np.zeros((num_regions, num_regions))
        for idr in range(num_regions):
            row = self.neighbours(self.regions_id[idr], num_regions)
            cMat[idr] = row
        return cMat

    def neighbours(self, region_id, num_regions):
        voxels = np.asarray(np.where(self.idMask == region_id))
        row = np.zeros(num_regions)
        for idv in range(voxels.shape[1]):
            coords = voxels[:,idv]
            submat = self.idMask[coords[0]-1:coords[0]+2, coords[1]-1:coords[1]+2, coords[2]-1:coords[2]+2]
            submat = submat.reshape(-1)
            for id in submat:
                if id != region_id and id !=0:
                    row[int(id-1)] += 1
        return row





    # --------------------------- DEFINE info functions ----------------------
    def _methods(self):
        pass

    def _summary(self):
        pass