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
from pwem.objects import SetOfParticles

from xmipp3.convert import setXmippAttribute
from pyworkflow import BETA, UPDATED, NEW, PROD

OUTPUTNAME = "outputParticles"
CITE ='Fernandez-Gimenez2023b'

class XmippProtConsensusLocalCTF(ProtAnalysis3D):
    """This protocol compares local defocus estimations obtained from multiple protocols for a set of particles. It evaluates the consistency among different CTF estimates and generates a set of particles with a consensus defocus."""
    _label = 'consensus local defocus'
    _devStatus = PROD

    _lastUpdateVersion = VERSION_2_0
    _possibleOutputs = {OUTPUTNAME:SetOfParticles}

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
        self._insertFunctionStep(self.compareDefocus)
        self._insertFunctionStep(self.createOutputStep)

    #--------------------------- STEPS functions ---------------------------------------------------
    def compareDefocus(self):
        """compare with median, mad and correlation matrix the local defocus value for each pixel estimated by
        different programs (Xmipp, Relion, gCTF, ...)"""
        allParticlesDefU = {}
        allParticlesDefV = {}
        allParticlesDefAngle = {}
        for defsetP in self.inputSets:
            defset = defsetP.get()
            for particle in defset:
                pIDs = []
                pDefUs = []
                pDefVs = []
                pDefAngles = []
                pIDs.append(particle.getObjId())
                pDefUs.append(particle.getCTF().getDefocusU())
                pDefVs.append(particle.getCTF().getDefocusV())
                pDefAngles.append(particle.getCTF().getDefocusAngle())
                for pId, pDefU, pDefV, pDefAngle in zip(pIDs, pDefUs, pDefVs, pDefAngles):
                    if not pId in allParticlesDefU.keys():
                        allParticlesDefU[pId]=[]
                        allParticlesDefV[pId] = []
                        allParticlesDefAngle[pId] = []
                    allParticlesDefU[pId].append(pDefU)
                    allParticlesDefV[pId].append(pDefV)
                    allParticlesDefAngle[pId].append(pDefAngle)
        self.defMatrixU = np.full(shape=(len(allParticlesDefU.keys()),len(self.inputSets)),fill_value=np.NaN)
        self.defMatrixV = np.full(shape=(len(allParticlesDefV.keys()),len(self.inputSets)),fill_value=np.NaN)
        self.defMatrixAngle = np.full(shape=(len(allParticlesDefAngle.keys()),len(self.inputSets)),fill_value=np.NaN)
        i=0
        self.pMatrixIdx={}
        for pIdU, pdefiU in allParticlesDefU.items():
            self.pMatrixIdx[pIdU]=i
            for j in enumerate(pdefiU):
                self.defMatrixU[i,j[0]]=j[1]
            i+=1
        l=0
        for pIdV, pdefiV in allParticlesDefV.items():
            self.pMatrixIdx[pIdV] = l
            for k in enumerate(pdefiV):
                self.defMatrixV[l, k[0]] = k[1]
            l+=1
        n=0
        for pIdAngle, pdefiAngle in allParticlesDefAngle.items():
            self.pMatrixIdx[pIdAngle] = n
            for p in enumerate(pdefiAngle):
                self.defMatrixAngle[n, p[0]] = p[1]
            n+=1

        self.medianU = np.nanmedian(self.defMatrixU, axis=1)
        self.medianV = np.nanmedian(self.defMatrixV, axis=1)
        self.medianAngle = np.nanmedian(self.defMatrixAngle, axis=1)
        print(self.medianAngle)

        self.madU = np.empty_like(self.medianU)
        self.madV = np.empty_like(self.medianV)
        self.madAngle = np.empty_like(self.medianAngle)

        for l in enumerate(self.medianU):
            self.madU[l[0]] = np.nanmedian(np.abs(self.defMatrixU[l[0],:] - self.medianU[l[0]]))
        for m in enumerate(self.medianV):
            self.madV[m[0]] = np.nanmedian(np.abs(self.defMatrixV[m[0], :] - self.medianV[m[0]]))
        for o in enumerate(self.medianAngle):
            self.madAngle[o[0]] = np.nanmedian(np.abs(self.defMatrixAngle[o[0], :] - self.medianAngle[o[0]]))

        self.defMatrixU = np.hstack((self.defMatrixU, self.medianU.reshape(len(self.medianU),1)))
        self.corrMatrixU = np.zeros((self.defMatrixU.shape[0],self.defMatrixU.shape[1]))
        defMatrixInvalidU = np.ma.masked_invalid(self.defMatrixU)
        self.corrMatrixU = np.ma.corrcoef(defMatrixInvalidU,rowvar=False)
        np.savetxt(self._getExtraPath("defocusMatrixU.txt"),self.defMatrixU)
        np.savetxt(self._getExtraPath("correlationMatrixU.txt"),self.corrMatrixU)

        self.defMatrixV = np.hstack((self.defMatrixV, self.medianV.reshape(len(self.medianV),1)))
        self.corrMatrixV = np.zeros((self.defMatrixV.shape[0],self.defMatrixV.shape[1]))
        defMatrixInvalidV = np.ma.masked_invalid(self.defMatrixV)
        self.corrMatrixV = np.ma.corrcoef(defMatrixInvalidV,rowvar=False)
        np.savetxt(self._getExtraPath("defocusMatrixV.txt"),self.defMatrixV)
        np.savetxt(self._getExtraPath("correlationMatrixV.txt"),self.corrMatrixV)

        self.defMatrixAngle = np.hstack((self.defMatrixAngle, self.medianAngle.reshape(len(self.medianAngle),1)))
        self.corrMatrixAngle = np.zeros((self.defMatrixAngle.shape[0],self.defMatrixAngle.shape[1]))
        defMatrixInvalidAngle = np.ma.masked_invalid(self.defMatrixAngle)
        self.corrMatrixAngle = np.ma.corrcoef(defMatrixInvalidAngle,rowvar=False)
        np.savetxt(self._getExtraPath("defocusMatrixAngle.txt"),self.defMatrixAngle)
        np.savetxt(self._getExtraPath("correlationMatrixAngle.txt"),self.corrMatrixAngle)

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
                pMedianU = Float(self.medianU[index]) # consensus defocus U
                pMedianV = Float(self.medianV[index]) # consensus defocus V
                pMedianAngle = Float(self.medianAngle[index]) # consensus defocus Angle

                pMadU = Float(self.madU[index]) # residual consensus defocus U
                newPart._ctfModel._defocusU.set(pMedianU)
                newPart._ctfModel._defocusV.set(pMedianV)
                newPart._ctfModel._defocusAngle.set(pMedianAngle)

                setXmippAttribute(newPart.getCTF(), emlib.MDL_CTF_DEFOCUS_RESIDUAL, pMadU)
                outputSet.append(newPart)

        self._defineOutputs(**{OUTPUTNAME:outputSet})
        self._defineSourceRelation(self.inputSet, outputSet)


    def _updateItem(self, particle, row):
        """update each particle in the set with the values computed"""
        pId = particle.getObjId()
        setXmippAttribute(particle,emlib.MDL_CTF_DEFOCUSA,self.medianU[self.pMatrixIdx[pId]])
        setXmippAttribute(particle,emlib.MDL_CTF_DEFOCUS_RESIDUAL,self.madU[self.pMatrixIdx[pId]])
        setXmippAttribute(particle,emlib.MDL_CTF_DEFOCUS_ANGLE,self.medianAngle[self.pMatrixIdx[pId]])


    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Consensus local defocus and residual computed for %s particles" % self.getObjectTag(OUTPUTNAME))
        return summary

    def _methods(self):
        methods = []
        methods.append("The results obtained when local CTF is estimated by different methods are compared here "
                       "computing the median, MAD and correlation matrix.")
        return methods
