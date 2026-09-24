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

CITE ='Fernandez-Gimenez2023b'


class XmippProtAnalyzeLocalCTF(ProtAnalysis3D):
    """Assigns to each micrograph a coefficient (R2) which evaluates the result of the
       local defocus adjustment and displays the local defocus for all the
       particles in each micrograph.

       AI Generated:

       What this protocol is for

        Analyze local defocus is a quality-control and diagnostic protocol
        intended for workflows where you have already estimated a local defocus
        per particle (for example, after running a local CTF
        refinement/adjustment step). Its purpose is not to refine CTF again,
        but to evaluate how coherent and interpretable the local defocus field
        is within each micrograph, and to help you detect micrographs where
        local defocus estimates behave suspiciously or are inconsistent with
        a simple physical model.

        From a biological user’s viewpoint, the protocol answers two practical
        questions. First, “do the local defocus values across particles in a
        micrograph follow a plausible smooth trend (like a tilt plane)?”
        Second, “which micrographs show a reliable local defocus pattern and
        which ones may be problematic and should be inspected or down-weighted?”

        Inputs and what they must contain

        You provide two inputs: a set of micrographs and a set of particles.
        The particle set must contain two essential pieces of information:
        each particle must be linked to a micrograph (so the protocol knows
        which micrograph it comes from), and each particle must already have
        a local defocus assigned in its CTF parameters. The protocol also uses
        particle coordinates within the micrograph (x, y), because it evaluates
        how defocus varies spatially across the image.

        In practice, this protocol is typically placed immediately after a step
        that assigns per-particle local CTF/defocus (or after importing
        particles that already include local defocus). If the particle set
        does not actually contain local defocus values, the analysis will not
        be meaningful.

        What the protocol computes (conceptually)

        For each micrograph, the protocol collects all particles belonging to
        it and looks at the mean defocus per particle, computed as the average
        of DefocusU and DefocusV. It then fits a very simple spatial model
        where defocus is approximated by a plane over the micrograph
        coordinates, i.e., defocus changes linearly with x and y. This is a
        physically sensible first-order approximation for common causes of
        local defocus variation such as specimen tilt or smooth ice thickness
        gradients.

        Once this plane is fitted, the protocol measures how well the
        particle-wise defocus values follow that plane. The main summary metric
        it produces per micrograph is an R² coefficient, which you can
        interpret as “how much of the observed variation in local defocus can
        be explained by a smooth planar trend.” Values closer to 1 indicate
        that local defocus estimates behave consistently and follow a coherent
        spatial pattern; values near 0 indicate that the estimates are poorly
        explained by a plane, which can happen when local defocus is noisy,
        unstable, or driven by artifacts.

        In addition to the global per-micrograph score, the protocol also
        stores per-particle information describing the fitted plane and the
        residual for each particle, meaning the difference between the
        particle’s local defocus and the value predicted by the plane at that
        coordinate. Biologically, residual maps are useful because they
        highlight localized regions where local CTF behaves oddly—often
        corresponding to contamination, thick ice patches, drift/charging
        effects, carbon edges, or other micrograph-level problems.

        Outputs and how to use them

        The protocol produces an output SetOfMicrographs that is essentially
        the same as your input micrograph set, but with an additional
        per-micrograph attribute: the R² score of the local defocus analysis.
        In Scipion, you can use this output as a convenient way to sort,
        filter, or select micrographs based on how well-behaved local defocus
        appears to be.

        A typical biological workflow is to inspect the distribution of R²
        values and then decide whether to exclude the worst micrographs, to
        inspect them visually, or to reconsider local CTF settings upstream.
        For example, if you observe a subset of micrographs with very low R²,
        those micrographs often correspond to acquisition problems (poor CTF
        estimation, strong astigmatism not captured well, drift, charging, ice
        gradients too complex, or simply too few particles to support a stable
        local model). Conversely, a dataset where most micrographs show high
        R² usually indicates that local defocus is being estimated consistently
        and that per-particle CTF refinement is likely providing a meaningful
        correction.

        Although the protocol’s primary formal output is the micrograph set
        with R², the analysis also prepares the underlying per-particle spatial
        defocus/residual information for visualization. In practice, this
        enables viewers to display, micrograph by micrograph, how local defocus
        values are distributed across the particle coordinates and where
        residuals concentrate. For a biological user, this is often the most
        intuitive diagnostic: you can immediately see whether defocus changes
        smoothly across the field of view or whether it looks patchy and
        incoherent.

        Interpreting R² biologically (what high and low values usually mean)

        A high R² typically means that local defocus values vary smoothly
        across the micrograph in a way consistent with a simple planar trend.
        This is what you would expect if local defocus is mostly driven by tilt
        or other smooth gradients and if per-particle CTF estimation is stable.
        Such micrographs are generally “safe” with respect to local CTF
        behaviour.

        A low R² means that the planar model explains little of the observed
        variation. This does not automatically imply that the micrograph is
        unusable, but it is a warning sign that local defocus estimates may be
        dominated by noise or by effects more complex than a plane. Common
        biological/experimental reasons include strong non-planar ice thickness
        variation, local distortions, contamination, carbon edges, charging,
        poor signal-to-noise, or simply that the micrograph contains too few
        particles (so the fit becomes unstable). Low R² micrographs are often
        good candidates for targeted inspection and may be candidates for
        exclusion or down-weighting depending on the downstream sensitivity.

        One subtlety to keep in mind is that if the particles in a micrograph
        happen to have nearly constant estimated defocus (very little spread),
        the R² can be uninformative; in those cases, you should rely more on
        visual inspection and on upstream CTF diagnostics. Practically, this
        tends to happen when local defocus estimation collapses to a constant
        value or when the particle distribution does not cover the micrograph
        broadly enough.

        Practical ways to use this protocol in a workflow

        This protocol is most powerful as part of a quality-control loop.
        After running local defocus estimation, you run Analyze local defocus,
        then you examine micrographs ranked by R², and you decide on one of
        three actions: keep everything if R² is consistently high; manually
        inspect and possibly exclude a subset of poor micrographs if there is
        a tail of low values; or, if many micrographs have poor scores, revisit
        the upstream local CTF settings, particle picking/extraction quality,
        or micrograph preprocessing.

        For biological projects where subtle high-resolution features matter,
        this analysis can be particularly useful because unstable local defocus
        often translates into weaker high-frequency signal in 2D/3D refinement.
        Filtering out the worst micrographs (or at least understanding why they
        behave poorly) can make the difference between a dataset that refines
        cleanly and one that stalls.
        """
    _label = 'analyze local defocus'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputMics', PointerParam, label="Input micrographs",
                      pointerClass='SetOfMicrographs')
        form.addParam('inputSet', PointerParam, label="Input images",
                      pointerClass='SetOfParticles', help="Set of particles with assigned local defocus")

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("analyzeDefocus")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ---------------------------------------------------
    def analyzeDefocus(self):
        """compute R2 coefficient of each micrograph and prepare data to be later displayed in the viewer as a 3D
        representation of the distribution of particles in the micrograph"""
        micIds = []
        particleIds = []
        x = []
        y = []
        defocusU = []
        defocusV = []
        for particle in self.inputSet.get():
            micIds.append(particle.getMicId())
            particleIds.append(particle.getObjId())
            xi, yi = particle.getCoordinate().getPosition()
            x.append(xi)
            y.append(yi)
            defocusU.append(particle.getCTF().getDefocusU())
            defocusV.append(particle.getCTF().getDefocusV())
        uniqueMicIds = list(set(micIds))
        self.R2 = {}
        md = emlib.MetaData()

        for micId in uniqueMicIds:
            idx = [i for i, j in enumerate(micIds) if j == micId]
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

            # defocus = c*y + b*x + a = A * X; A=[x(i),y(i)]
            A = np.column_stack([np.ones(len(xbyId)), xbyId, ybyId])
            polynomial, _, _, _ = np.linalg.lstsq(A, meanDefocusbyId, rcond=None)
            residuals = 0
            for Ai, bi in zip(A, meanDefocusbyId):
                residuali = bi - (Ai[0]*polynomial[0] + Ai[1]*polynomial[1] + Ai[2]*polynomial[2])
                residuals += residuali*residuali
            meanDefocusbyIdArray = np.asarray(meanDefocusbyId)
            coefficients = np.asarray(polynomial)
            den = sum((meanDefocusbyIdArray - meanDefocusbyIdArray.mean()) ** 2)
            if den == 0:
                self.R2[micId] = 0
            else:
                self.R2[micId] = 1 - residuals / den
            mdBlock = emlib.MetaData()
            for xi, yi, deltafi, parti in zip(xbyId, ybyId, meanDefocusbyId, particleIdsbyMicId):
                objId = mdBlock.addObject()
                mdBlock.setValue(emlib.MDL_ITEM_ID, parti, objId)
                mdBlock.setValue(emlib.MDL_XCOOR, xi, objId)
                mdBlock.setValue(emlib.MDL_YCOOR, yi, objId)
                mdBlock.setValue(emlib.MDL_CTF_DEFOCUSA, deltafi, objId)
                estimatedVal = coefficients[2]*yi + coefficients[1]*xi + coefficients[0]
                residuali = deltafi - estimatedVal
                mdBlock.setValue(emlib.MDL_CTF_DEFOCUS_RESIDUAL, residuali, objId)
            mdBlock.write("mic_%d@%s" % (micId, self._getExtraPath("micrographDefoci.xmd")), emlib.MD_APPEND)
            objId = md.addObject()
            md.setValue(emlib.MDL_CTF_DEFOCUS_COEFS, coefficients.tolist(), objId)
            md.write(self._getExtraPath("micrographCoef.xmd"), emlib.MD_APPEND)

    def createOutputStep(self):
        """create as output a setOfParticles and add the columns of corresponding computed metadata"""
        inputMicSet = self.inputMics.get()
        fnMics = self._getExtraPath('input_mics.xmd')
        writeSetOfMicrographs(inputMicSet, fnMics)
        mdMics = md.MetaData(fnMics)
        for objId in mdMics:
            micId = mdMics.getValue(emlib.MDL_ITEM_ID, objId)
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

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Local defocus analyzed for %i particles" % self.inputSet.get().getSize())
        return summary
    
    def _methods(self):
        methods = []
        methods.append("The results obtained when local CTF is calculated are analyzed here. The adjust coefficients, "
                       "residues and R2 are calculated for each micrograph.")
        return methods
