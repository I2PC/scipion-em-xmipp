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
    """This protocol compares local defocus estimations obtained from multiple
    protocols for a set of particles. It evaluates the consistency among
    different CTF estimates and generates a set of particles with a consensus
    defocus.

    AI Generated:

    Overview

    The Consensus Local Defocus protocol is designed for a common practical
    problem in cryo-EM processing: you may estimate local CTF/defocus per
    particle using different programs (for example Xmipp, Relion-based workflows,
    gCTF-derived approaches, or other local CTF estimators) and obtain slightly
    different values. This protocol compares those alternative estimates and
    produces a single consensus defocus per particle, together with a measure
    of disagreement.

    From a user perspective, it answers:

    If several methods estimate local defocus for the same particles, what is
    the most reliable “central” value—and how consistent are the methods with each other?

    This is particularly useful when you want to:
    - reduce method-specific noise or bias by combining estimates,
    - detect problematic particles where defocus estimation is unstable,
    - generate a unified particle set for downstream refinement using
    consistent CTF parameters.

    Inputs and General Workflow

    The protocol uses two types of particle sets:
    - Input particles to assign the consensus defocus. This is the target
    particle set you want to update with the final consensus defocus values.
    Think of it as your “main dataset” that will continue through the pipeline.

    - Input defocus estimations (multiple particle sets). Here you provide
    several particle sets, each containing the same particles but with
    different local defocus estimations already stored in their CTF models.

    Important practical point: these sets must correspond to the same images
    (same particle identities). The protocol matches particles by their
    internal identifiers. If one set is missing particles or uses different
    identifiers, those particles may simply not receive a consensus value.

    What the Protocol Computes

    For each particle, the protocol collects the defocus parameters from all
    the provided estimations:
    - Defocus U
    - Defocus V
    - Defocus angle (astigmatism angle)

    It then computes:
    1) Consensus defocus (median). The protocol uses the median across methods
    as the consensus value (for U, V, and angle). The median is a very robust
    choice: it is much less sensitive than the mean to an outlier method or to
    occasional failed estimates.

    Biologically/practically, this means the consensus is aiming to represent
    the “typical” defocus estimate supported by most methods.

    2) Residual disagreement (MAD). To quantify how consistent the estimators
    are, the protocol computes the MAD (median absolute deviation) for defocus
    U (and internally also for V and angle, although the output explicitly
    stores a residual value associated with defocus).

    Interpretation:
    - low MAD → the different methods largely agree → defocus is stable and
    reliable
    - high MAD → the methods disagree → particle defocus may be ambiguous (
    low SNR, contamination, bad local background, poor particle windowing, etc.)

    In practice, this residual is extremely useful as a quality indicator.

    3) Correlation matrices between methods. The protocol also computes and
    saves correlation matrices comparing the defocus estimates across all
    particles:
    - correlation between methods for Defocus U
    - correlation between methods for Defocus V
    - correlation between methods for Defocus angle

    These summaries help you understand whether two estimators behave similarly
    (high correlation) or systematically differ (low correlation).

    Outputs and Their Interpretation

    Output: particles with consensus defocus

    The main output is a new SetOfParticles in which each particle has updated
    CTF parameters:
    - Defocus U = median across methods
    - Defocus V = median across methods
    - Defocus angle = median across methods

    In addition, the protocol stores a defocus residual (derived from the MAD)
    in the particle’s CTF metadata. This residual acts as a per-particle
    indicator of uncertainty or disagreement among methods.

    This output is typically what you would feed into:
    - CTF-aware refinement
    - polishing / per-particle corrections (depending on your pipeline)
    - quality filtering steps (e.g., removing particles with highly unstable
    defocus)

    Saved diagnostic files

    In the protocol’s extra output directory, it also writes text files with:
    - defocus matrices (per particle × per method, plus the median column)
    - correlation matrices between methods

    These are mainly intended for diagnosis and reporting. For example, they
    allow you to identify whether one estimator is systematically shifted
    relative to the others.

    Practical Recommendations

    When to use it

    This protocol is most useful when you have:
    - two or more alternative local defocus estimations for the same dataset,
    and
    - you want a single consistent particle set for downstream processing.

    It is also useful as a sanity check: if estimators strongly disagree
    globally, that often indicates an underlying problem (incorrect pixel size,
    wrong voltage/spherical aberration settings, wrong micrograph grouping, or
    poor preprocessing).

    How many methods should you include?

    Two can already be helpful, but three or more is often better because the
    median becomes more informative and robust.

    How to interpret the residual

    Particles with very high residual disagreement are often:
    - low contrast / low SNR
    - partially contaminated or overlapping
    - near carbon edges or gradients
    - poorly centered or poorly windowed
    - affected by local ice thickness changes

    A common downstream strategy is to:
    - plot or inspect the distribution of residuals,
    - remove the worst tail (for example top 5–10%) if it is clearly separated.

    (Exact thresholds are dataset-dependent; the residual is best used
    comparatively.)

    Caution with defocus angle

    Angle can be less stable than defocus values, especially when astigmatism
    is weak. If you see strong disagreement specifically in angles, it may not
    always be biologically meaningful—often it is a symptom of low astigmatism
    signal.

    Final Perspective

    Local CTF estimation is one of those steps where small numerical
    differences can propagate into refinement quality and interpretability.
    The Consensus Local Defocus protocol provides a simple and robust way to
    combine multiple estimators, producing a single defocus per particle while
    also reporting how reliable that consensus is.

    For many users, its main value is not only the consensus defocus itself,
    but also the per-particle disagreement indicator, which helps detect
    problematic particles and increase confidence in downstream structural
    conclusions.
    """
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
