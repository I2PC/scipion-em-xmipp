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
from pyworkflow.protocol.params import PointerParam

from pwem import emlib
from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md

from xmipp3.convert import readSetOfMicrographs, writeSetOfMicrographs


class XmippProtAnalyzeLocalCTF(ProtAnalysis3D):
    """Assigns to each micrograph a coefficient (R2) which evaluates the result of the
        local defocus adjustment and displays the local defocus for all the particles in each micrograph."""
    _label = 'analyze local defocus'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputMics', PointerParam, label="Input micrographs",
                      pointerClass='SetOfMicrographs')
        form.addParam('inputSet', PointerParam, label="Input images",
                      pointerClass='SetOfParticles', help="Set of particles with assigned local defocus")

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("analyzeDefocus")
        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def analyzeDefocus(self):
        micIds=[]
        particleIds=[]
        x=[]
        y=[]
        defocusU=[]
        defocusV=[]
        for particle in self.inputSet.get():
            micIds.append(particle.getMicId())
            particleIds.append(particle.getObjId())
            xi, yi = particle.getCoordinate().getPosition()
            x.append(xi)
            y.append(yi)
            defocusU.append(particle.getCTF().getDefocusU())
            defocusV.append(particle.getCTF().getDefocusV())

        uniqueMicIds = list(set(micIds))
        self.R2={}

        md = emlib.MetaData()

        for micId in uniqueMicIds:
            idx = [i for i,j in enumerate(micIds) if j==micId]
            defocusUbyId = []
            defocusVbyId = []
            meanDefocusbyId = []
            xbyId = []
            ybyId = []
            particleIdsbyMicId = []
            for idxi in idx:
                defocusUbyId.append(defocusU[idxi])
                defocusVbyId.append(defocusV[idxi])
                meanDefocus = (defocusU[idxi]+defocusV[idxi])/2
                meanDefocusbyId.append(meanDefocus)
                xbyId.append(x[idxi])
                ybyId.append(y[idxi])
                particleIdsbyMicId.append(particleIds[idxi])

            #defocus = c*y + b*x + a = A * X; A=[x(i),y(i)]
            A = np.column_stack([np.ones(len(xbyId)),xbyId,ybyId])
            polynomial, residuals, _, _ = np.linalg.lstsq(A,meanDefocusbyId,rcond=None)
            meanDefocusbyIdArray = np.asarray(meanDefocusbyId)
            coefficients = np.asarray(polynomial)
            self.R2[micId] = 1 - residuals / sum((meanDefocusbyIdArray - meanDefocusbyIdArray.mean()) ** 2)
            mdBlock = emlib.MetaData()
            for xi, yi, deltafi, parti in zip(xbyId,ybyId,meanDefocusbyId,particleIdsbyMicId):
                objId = mdBlock.addObject()
                mdBlock.setValue(emlib.MDL_ITEM_ID,long(parti),objId)
                mdBlock.setValue(emlib.MDL_XCOOR,xi,objId)
                mdBlock.setValue(emlib.MDL_YCOOR,yi,objId)
                mdBlock.setValue(emlib.MDL_CTF_DEFOCUSA,deltafi,objId)
                estimatedVal = coefficients[2]*yi + coefficients[1]*xi + coefficients[0]
                residuali = deltafi - estimatedVal
                mdBlock.setValue(emlib.MDL_CTF_DEFOCUS_RESIDUAL,residuali,objId)
            mdBlock.write("mic_%d@%s"%(micId,self._getExtraPath("micrographDefoci.xmd")),emlib.MD_APPEND)
            objId = md.addObject()
            md.setValue(emlib.MDL_CTF_DEFOCUS_COEFS,coefficients.tolist(),objId)
            md.write(self._getExtraPath("micrographCoef.xmd"),emlib.MD_APPEND)


    def createOutputStep(self):
        inputMicSet = self.inputMics.get()
        fnMics = self._getExtraPath('input_mics.xmd')
        writeSetOfMicrographs(inputMicSet, fnMics)
        mdMics = md.MetaData(fnMics)
        for objId in mdMics:
            micId = mdMics.getValue(emlib.MDL_ITEM_ID,objId)
            if micId in self.R2:
                micR2 = float(self.R2[micId])
                mdMics.setValue(emlib.MDL_CTF_DEFOCUS_R2, micR2, objId)
        mdMics.write(fnMics)
        outputSet = self._createSetOfMicrographs()
        outputSet.copyInfo(inputMicSet)
        readSetOfMicrographs(fnMics, outputSet, extraLabels=[emlib.MDL_CTF_DEFOCUS_R2])

        self._defineOutputs(outputMicrographs=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)
        self._defineSourceRelation(inputMicSet, outputSet)


    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Local defocus analyzed for %i particles" % self.inputSet.get().getSize())
        return summary
    
    def _methods(self):
        methods = []
        methods.append("The results obtained when local CTF is calculated are analyzed here. The adjust coefficients, "
                       "residues and R2 are calculated for each micrograph.")
        return methods
