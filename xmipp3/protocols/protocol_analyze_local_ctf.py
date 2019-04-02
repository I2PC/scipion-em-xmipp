# coding=utf-8
# **************************************************************************
# *
# * Authors:     Carlos Óscar Sánchez Sorzano
# *              Estrella Fernández Giménez
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

from math import floor
import os

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import PointerParam, StringParam, FloatParam, BooleanParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.em.constants import ALIGN_PROJ
from pyworkflow.utils.path import cleanPath
from pyworkflow.em.protocol import ProtAnalysis3D
from pyworkflow.em.data import Image, SetOfParticles
from pyworkflow.em.metadata.utils import iterRows

import pyworkflow.em.metadata as md

import numpy as np

import xmippLib
from xmipp3.convert import setXmippAttributes, xmippToLocation, rowToAlignment
from xmipp3.convert import readSetOfMicrographs, writeSetOfMicrographs, setOfMicrographsToMd


        
class XmippProtAnalyzeLocalCTF(ProtAnalysis3D):
    """Assigns to each micrograph a coefficient (R2) which evaluates the result of the
        adjustment of the defocus made by protocol_local_ctf. It computes the minimun
        defocus between defocusU and defocusV."""
    _label = 'analyze local defocus'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images",
                      pointerClass='SetOfParticles')
        form.addParam('inputMics', PointerParam, label="Input micrographs",
                      pointerClass='SetOfMicrographs')

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("analyzeDefocus")
        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def analyzeDefocus(self):
        micIds=[]
        x=[]
        y=[]
        defocusU=[]
        defocusV=[]
        # For each particle, take the micId, coordinates x,y and defocusU,V
        for particle in self.inputSet.get():
            micIds.append(particle.getMicId())
            xi, yi = particle.getCoordinate().getPosition()
            x.append(xi)
            y.append(yi)
            defocusU.append(particle.getCTF().getDefocusU())
            defocusV.append(particle.getCTF().getDefocusV())

        uniqueMicIds = list(set(micIds))
        self.R2={}
        # for each unique micId take all the defocusU,V and coordinates x,y of the particles of that mic
        for micId in uniqueMicIds:
            idx = [i for i,j in enumerate(micIds) if j==micId]
            defocusUbyId = []
            defocusVbyId = []
            xbyId = []
            ybyId = []
            for idxi in idx:

                defocusUbyId.append(defocusU[idxi])
                defocusVbyId.append(defocusV[idxi])
                xbyId.append(x[idxi])
                ybyId.append(y[idxi])

            #defocus = c*y + b*x + a = A * X; A=[x(i),y(i)], X=[a,b,c]
            A = np.column_stack([np.ones(len(xbyId)),xbyId,ybyId])

            polynomialU, residualsU, _, _ = np.linalg.lstsq(A,defocusUbyId,rcond=None)
            defocusUbyIdArray = np.asarray(defocusUbyId)
            R2u = 1 - residualsU / sum((defocusUbyIdArray - defocusUbyIdArray.mean()) ** 2)

            polynomialV, residualsV, _, _ = np.linalg.lstsq(A, defocusVbyId,rcond=None)
            defocusVbyIdArray = np.asarray(defocusVbyId)
            R2v = 1 - residualsV / sum((defocusVbyIdArray - defocusVbyIdArray.mean()) ** 2)
            self.R2[micId]=min(R2u,R2v)


    def createOutputStep(self):
        inputMicSet = self.inputMics.get()
        fnMics = self._getExtraPath('input_mics.xmd')
        writeSetOfMicrographs(inputMicSet, fnMics)
        mdMics = md.MetaData(fnMics)
        for objId in mdMics:
            micId = mdMics.getValue(xmippLib.MDL_ITEM_ID,objId)
            if micId in self.R2:
                micR2 = float(self.R2[micId])
                mdMics.setValue(xmippLib.MDL_CTF_DEFOCUS_R2, micR2, objId)
        mdMics.write(fnMics)
        outputSet = self._createSetOfMicrographs()
        readSetOfMicrographs(fnMics, outputSet, readCTF=True)
        self._defineOutputs(outputMicrographs=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)
        self._defineSourceRelation(inputMicSet, outputSet)

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Local defocus analyzed of %i particles" % self.inputSet.get().getSize())
        return summary
    
    def _methods(self):
        methods = []
        methods.append("The results obtained when local CTF is calculated are analyzed here. The adjust coefficients, residues and R2 are calculated for each micrograph.")
        return methods
