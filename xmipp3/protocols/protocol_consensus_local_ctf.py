# coding=utf-8
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *              Carlos Oscar Sanchez Sorzano
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia, CSIC
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

import numpy as np

from pyworkflow import VERSION_2_0
from pyworkflow.object import Float
from pyworkflow.protocol.params import MultiPointerParam, PointerParam

from pwem.protocols import ProtAnalysis3D
from pwem import emlib

from xmipp3.convert import setXmippAttribute


class XmippProtConsensusLocalCTF(ProtAnalysis3D):
    """This protocol compares the estimations of local defocus computed by different protocols for a set of particles"""
    _label = 'consensus local defocus'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam,
                      label="Input particles to assign the consensus defocus",
                      pointerClass='SetOfParticles',
                      help="Particle set of interest to estimate the defocus")
        form.addParam('inputSets', MultiPointerParam,
                      label="Input defocus estimations",
                      pointerClass='SetOfParticles',
                      help="Sets of particles with different local defocus estimations to compare")

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("compareDefocus")
        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def compareDefocus(self):
        """compare with median, mad and correlation matrix the local defocus value for each pixel estimated by
        different programs (Xmipp, Relion, gCTF, ...)"""
        allParticlesDef = {}
        for defsetP in self.inputSets:
            defset = defsetP.get()
            for particle in defset:
                pIDs = []
                pDefAs = []
                pIDs.append(particle.getObjId())
                defU = particle.getCTF().getDefocusU()
                defV = particle.getCTF().getDefocusV()
                pDefAs.append((defU+defV)/2)
                for pId, pDefA in zip(pIDs,pDefAs):
                    if not pId in allParticlesDef.keys():
                        allParticlesDef[pId]=[]
                    allParticlesDef[pId].append(pDefA)
        self.defMatrix = np.full(shape=(len(allParticlesDef.keys()),len(self.inputSets)),fill_value=np.NaN)
        i=0
        self.pMatrixIdx={}
        for pId,pdefi in allParticlesDef.items():
            self.pMatrixIdx[pId]=i
            for j in enumerate(pdefi):
                self.defMatrix[i,j[0]]=j[1]
            i+=1

        self.median = np.nanmedian(self.defMatrix, axis=1)
        self.mad = np.empty_like(self.median)
        for k in enumerate(self.median):
            self.mad[k[0]] = np.nanmedian(np.abs(self.defMatrix[k[0],:] - self.median[k[0]]))
        self.defMatrix = np.hstack((self.defMatrix, self.median.reshape(len(self.median),1)))
        self.corrMatrix = np.zeros((self.defMatrix.shape[0],self.defMatrix.shape[1]))
        defMatrixInvalid = np.ma.masked_invalid(self.defMatrix)
        self.corrMatrix = np.ma.corrcoef(defMatrixInvalid,rowvar=False)
        np.savetxt(self._getExtraPath("defocusMatrix.txt"),self.defMatrix)
        np.savetxt(self._getExtraPath("correlationMatrix.txt"),self.corrMatrix)

    def createOutputStep(self):
        """create as output a setOfParticles with a consensus estimation of local defocus (median) and its median
        standard deviation (mad)"""
        imgSet = self.inputSet.get()

        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(imgSet)
        # Loop through input set of
        for part in imgSet.iterItems():

            id = part.getObjId()
            if id in self.pMatrixIdx:
                index = self.pMatrixIdx[id]
                newPart = part.clone()
                pMedian = Float(self.median[index])
                pMad = Float(self.mad[index])
                setXmippAttribute(newPart.getCTF(), emlib.MDL_CTF_DEFOCUSA, pMedian)
                setXmippAttribute(newPart.getCTF(), emlib.MDL_CTF_DEFOCUS_RESIDUAL, pMad)
                outputSet.append(newPart)

        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)


    def _updateItem(self, particle, row):
        """update each particle in the set with the values computed"""
        pId = particle.getObjId()
        setXmippAttribute(particle,emlib.MDL_CTF_DEFOCUSA,self.median[self.pMatrixIdx[pId]])
        setXmippAttribute(particle,emlib.MDL_CTF_DEFOCUS_RESIDUAL,self.mad[self.pMatrixIdx[pId]])

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Consensus local defocus and residual computed for %s particles" % self.getObjectTag('outputParticles'))
        return summary

    def _methods(self):
        methods = []
        methods.append("The results obtained when local CTF is estimated by different methods are compared here "
                       "computing the median, MAD and correlation matrix.")
        return methods
