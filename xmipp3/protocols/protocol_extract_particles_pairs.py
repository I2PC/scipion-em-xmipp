# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Laura del Cano (ldelcano@cnb.csic.es)
# *              Adrian Quintana (aquintana@cnb.csic.es)
# *              Javier Vargas (jvargas@cnb.csic.es)
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

try:
    from itertools import izip
except ImportError:
    izip = zip

from os.path import exists

import pwem.emlib.metadata as md
import pyworkflow.utils as pwutils
from pyworkflow.protocol.constants import (STEPS_PARALLEL, LEVEL_ADVANCED,
                                           STATUS_FINISHED)
from pyworkflow.protocol.params import (PointerParam, EnumParam, FloatParam,
                                        IntParam, BooleanParam, Positive, GE)
from pwem.protocols import ProtExtractParticlesPair
from pwem.objects import ParticlesTiltPair, TiltPair, SetOfParticles

from xmipp3.base import XmippProtocol
from xmipp3.convert import (writeSetOfCoordinates, readSetOfParticles,
                            micrographToCTFParam)
from xmipp3.constants import OTHER


class XmippProtExtractParticlesPairs(ProtExtractParticlesPair, XmippProtocol):
    """Protocol to extract particles from a set of tilted pairs coordinates"""
    _label = 'extract particle pairs'

    def __init__(self, **kwargs):
        ProtExtractParticlesPair.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinatesTiltedPairs', PointerParam,
                      important=True, label="Coordinates tilted pairs",
                      pointerClass='CoordinatesTiltPair',
                      help='Select the CoordinatesTiltPairs')
        form.addParam('downsampleType', EnumParam,
                      choices=['same as picking', 'other'],
                      default=0, important=True, label='Micrographs source',
                      display=EnumParam.DISPLAY_HLIST,
                      help='By default the particles will be extracted '
                           'from the micrographs used in the picking '
                           'step ( _same as picking_ option ).\n'
                           'If you select _other_ option, you must provide '
                           'a different set of micrographs to extract from.\n'
                           '*Note*: In the _other_ case, ensure that provided '
                           'micrographs and coordinates are related '
                           'by micName or by micId. Difference in pixel size '
                           'will be handled automatically.')
        form.addParam('inputMicrographsTiltedPair', PointerParam,
                      pointerClass='MicrographsTiltPair',
                      condition='downsampleType != 0',
                      important=True, label='Input tilt pair micrographs',
                      help='Select the tilt pair micrographs from which to '
                           'extract.')
        form.addParam('ctfUntilt', PointerParam, allowsNull=True,
                      # expertLevel=LEVEL_ADVANCED,
                      pointerClass='SetOfCTF',
                      label='CTF estimation (untilted mics)',
                      help='Choose some CTF estimation related to input '
                           'UNTILTED micrographs. \n CTF estimation is needed '
                           'if you want to do phase flipping or you want to '
                           'associate CTF information to the particles.')
        form.addParam('ctfTilt', PointerParam, allowsNull=True,
                      # expertLevel=LEVEL_ADVANCED,
                      pointerClass='SetOfCTF',
                      label='CTF estimation (tilted mics)',
                      help='Choose some CTF estimation related to input '
                           'TILTED micrographs. \n CTF estimation is needed '
                           'if you want to do phase flipping or you want to '
                           'associate CTF information to the particles.')

        # downFactor should always be 1.0 or greater
        geOne = GE(1.0, error='Value should be greater or equal than 1.0')

        form.addParam('downFactor', FloatParam, default=1.0,
                      validators=[geOne],
                      label='Downsampling factor',
                      help='Select a value greater than 1.0 to reduce the size '
                           'of micrographs before extracting the particles. '
                           'If 1.0 is used, no downsample is applied. '
                           'Non-integer downsample factors are possible. ')
        form.addParam('boxSize', IntParam, default=0,
                      label='Particle box size', validators=[Positive],
                      help='In pixels. The box size is the size of the boxed '
                           'particles, actual particles may be smaller than '
                           'this. If you do downsampling after extraction, '
                           'provide final box size here.')
        form.addParam('doBorders', BooleanParam, default=False,
                      label='Fill pixels outside borders',
                      help='Xmipp by default skips particles whose boxes fall '
                           'outside of the micrograph borders. Set this '
                           'option to True if you want those pixels outside '
                           'the borders to be filled with the closest pixel '
                           'value available')

        form.addSection(label='Preprocess')
        form.addParam('doRemoveDust', BooleanParam, default=True,
                      important=True, label='Dust removal (Recommended)',
                      help='Sets pixels with unusually large values to random '
                           'values from a Gaussian with zero-mean and '
                           'unity-standard deviation.')
        form.addParam('thresholdDust', FloatParam, default=3.5,
                      condition='doRemoveDust', expertLevel=LEVEL_ADVANCED,
                      label='Threshold for dust removal',
                      help='Pixels with a signal higher or lower than this '
                           'value times the standard deviation of the image '
                           'will be affected. For cryo, 3.5 is a good value. '
                           'For high-contrast negative stain, the signal '
                           'itself may be affected so that a higher value may '
                           'be preferable.')
        form.addParam('doInvert', BooleanParam, default=None,
                      label='Invert contrast',
                      help='Invert the contrast if your particles are black '
                           'over a white background. Xmipp, Spider, Relion '
                           'and Eman require white particles over a black '
                           'background, Frealign (up to v9.07) requires black '
                           'particles over a white background')
        form.addParam('doFlip', BooleanParam, default=False,
                      # expertLevel=LEVEL_ADVANCED,
                      label='Phase flipping',
                      help='Use the information from the CTF to compensate for '
                           'phase reversals.\n'
                           'Phase flip is recommended in Xmipp or Eman\n'
                           '(even Wiener filtering and bandpass filter are '
                           'recommended for obtaining better 2D classes)\n'
                           'Otherwise (Frealign, Relion, Spider, ...), '
                           'phase flip is not recommended.')
        form.addParam('doNormalize', BooleanParam, default=True,
                      label='Normalize (Recommended)',
                      help='It subtract a ramp in the gray values and '
                           'normalizes so that in the background there is 0 '
                           'mean and standard deviation 1.')
        form.addParam('normType', EnumParam,
                      choices=['OldXmipp', 'NewXmipp', 'Ramp'], default=2,
                      condition='doNormalize', expertLevel=LEVEL_ADVANCED,
                      display=EnumParam.DISPLAY_COMBO,
                      label='Normalization type',
                      help='OldXmipp (mean(Image)=0, stddev(Image)=1).\n'
                           'NewXmipp (mean(background)=0, '
                           'stddev(background)=1)\n'
                           'Ramp (subtract background+NewXmipp).')
        form.addParam('backRadius', IntParam, default=-1,
                      condition='doNormalize',
                      label='Background radius', expertLevel=LEVEL_ADVANCED,
                      help='Pixels outside this circle are assumed to be '
                           'noise and their stddev is set to 1. Radius for '
                           'background circle definition (in pix.). If this '
                           'value is 0, then half the box size is used.')
        form.addParallelSection(threads=4, mpi=1)

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._setupBasicProperties()
        # Write pos files for all micrographs
        firstStepId = self._insertFunctionStep('writePosFilesStep')
        # For each micrograph insert the steps, run in parallel
        deps = []

        def _insertMicStep(mic):
            localDeps = [firstStepId]
            fnLast = mic.getFileName()
            micName = pwutils.removeBaseExt(mic.getFileName())

            def getMicTmp(suffix):
                return self._getTmpPath(micName + suffix)

            # Create a list with micrographs operations (programs in xmipp) and
            # the required command line parameters (except input/ouput files)
            micOps = []

            # Check if it is required to downsample your micrographs
            downFactor = self.downFactor.get()

            if self.notOne(downFactor):
                fnDownsampled = getMicTmp("_downsampled.xmp")
                args = "-i %s -o %s --step %f --method fourier"
                micOps.append(('xmipp_transform_downsample',
                               args % (fnLast, fnDownsampled, downFactor)))
                fnLast = fnDownsampled

            if self.doRemoveDust:
                fnNoDust = getMicTmp("_noDust.xmp")
                args = " -i %s -o %s --bad_pixels outliers %f"
                micOps.append(('xmipp_transform_filter',
                               args % (fnLast, fnNoDust, self.thresholdDust)))
                fnLast = fnNoDust

            if self._useCTF() and self.ctfDict[mic] is not None:
                # We need to write a Xmipp ctfparam file
                # to perform the phase flip on the micrograph
                fnCTF = self._getTmpPath("%s.ctfParam" % micName)
                mic.setCTF(self.ctfDict[mic])
                micrographToCTFParam(mic, fnCTF)
                # Insert step to flip micrograph
                if self.doFlip:
                    fnFlipped = getMicTmp('_flipped.xmp')
                    args = " -i %s -o %s --ctf %s --sampling %f"
                    micOps.append(('xmipp_ctf_phase_flip',
                                   args % (fnLast, fnFlipped, fnCTF,
                                           self._getNewSampling())))
                    fnLast = fnFlipped
            else:
                fnCTF = None

            # Actually extract
            deps.append(self._insertFunctionStep('extractParticlesStep',
                                                 mic.getObjId(), micName,
                                                 fnCTF, fnLast, micOps,
                                                 self.doInvert.get(),
                                                 self._getNormalizeArgs(),
                                                 self.doBorders.get(),
                                                 prerequisites=localDeps))

        for mic in self.ctfDict:
            _insertMicStep(mic)

        metaDeps = self._insertFunctionStep('createMetadataImageStep',
                                            prerequisites=deps)

        # Insert step to create output objects
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=[metaDeps], wait=False)

    # --------------------------- STEPS functions ------------------------------
    def writePosFilesStep(self):
        """ Write the pos file for each micrograph in metadata format
        (both untilted and tilted). """
        writeSetOfCoordinates(self._getExtraPath(),
                              self.inputCoords.getUntilted(),
                              scale=self.getBoxScale())
        writeSetOfCoordinates(self._getExtraPath(),
                              self.inputCoords.getTilted(),
                              scale=self.getBoxScale())

        # We need to find the mapping by micName (without ext) between the
        #  micrographs in the SetOfCoordinates and the Other micrographs
        if self._micsOther():
            micDict = {}
            for micU, micT in izip(self.inputCoords.getUntilted().getMicrographs(),
                                   self.inputCoords.getTilted().getMicrographs()):
                micBaseU = pwutils.removeBaseExt(micU.getFileName())
                micPosU = self._getExtraPath(micBaseU + ".pos")
                micDict[pwutils.removeExt(micU.getMicName())] = micPosU
                micBaseT = pwutils.removeBaseExt(micT.getFileName())
                micPosT = self._getExtraPath(micBaseT + ".pos")
                micDict[pwutils.removeExt(micT.getMicName())] = micPosT

            # now match micDict and other mics (in self.ctfDict)
            if any(pwutils.removeExt(mic.getMicName()) in micDict for mic in self.ctfDict):
                micKey = lambda mic: pwutils.removeExt(mic.getMicName())
            else:
                raise Exception('Could not match input micrographs and coordinates '
                                'by micName.')

            for mic in self.ctfDict:
                mk = micKey(mic)
                if mk in micDict:
                    micPosCoord = micDict[mk]
                    if exists(micPosCoord):
                        micBase = pwutils.removeBaseExt(mic.getFileName())
                        micPos = self._getExtraPath(micBase + ".pos")
                        if micPos != micPosCoord:
                            self.info('Moving %s -> %s' % (micPosCoord, micPos))
                            pwutils.moveFile(micPosCoord, micPos)

    def extractParticlesStep(self, micId, baseMicName, fnCTF,
                             micrographToExtract, micOps,
                             doInvert, normalizeArgs, doBorders):
        """ Extract particles from one micrograph """
        outputRoot = str(self._getExtraPath(baseMicName))
        fnPosFile = self._getExtraPath(baseMicName + ".pos")

        # If it has coordinates extract the particles
        particlesMd = 'particles@%s' % fnPosFile
        boxSize = self.boxSize.get()
        boxScale = self.getBoxScale()
        print("boxScale: ", boxScale)

        if exists(fnPosFile):
            # Apply first all operations required for the micrograph
            for program, args in micOps:
                self.runJob(program, args)

            args = " -i %s --pos %s" % (micrographToExtract, particlesMd)
            args += " -o %s --Xdim %d" % (outputRoot, boxSize)

            if doInvert:
                args += " --invert"

            if fnCTF:
                args += " --ctfparam " + fnCTF

            if doBorders:
                args += " --fillBorders"

            self.runJob("xmipp_micrograph_scissor", args)

            # Normalize
            if normalizeArgs:
                self.runJob('xmipp_transform_normalize',
                            '-i %s.stk %s' % (outputRoot, normalizeArgs))
        else:
            self.warning("The micrograph %s hasn't coordinate file! "
                         % baseMicName)
            self.warning("Maybe you picked over a subset of micrographs")

        # Let's clean the temporary mrc micrographs
        if not pwutils.envVarOn("SCIPION_DEBUG_NOCLEAN"):
            pwutils.cleanPattern(self._getTmpPath(baseMicName) + '*')

    def createMetadataImageStep(self):
        mdUntilted = md.MetaData()
        mdTilted = md.MetaData()
        # for objId in mdPairs:
        for uMic, tMic in izip(self.uMics, self.tMics):
            umicName = pwutils.removeBaseExt(uMic.getFileName())
            fnMicU = self._getExtraPath(umicName + ".xmd")
            fnPosU = self._getExtraPath(umicName + ".pos")
            # Check if there are picked particles in these micrographs
            if pwutils.exists(fnMicU):
                mdMicU = md.MetaData(fnMicU)
                mdPosU = md.MetaData('particles@%s' % fnPosU)
                mdPosU.merge(mdMicU)
                mdUntilted.unionAll(mdPosU)
                tmicName = pwutils.removeBaseExt(tMic.getFileName())
                fnMicT = self._getExtraPath(tmicName + ".xmd")
                fnPosT = self._getExtraPath(tmicName + ".pos")
                mdMicT = md.MetaData(fnMicT)
                mdPosT = md.MetaData('particles@%s' % fnPosT)
                mdPosT.merge(mdMicT)
                mdTilted.unionAll(mdPosT)

        # Write image metadata (check if it is really necessary)
        fnTilted = self._getExtraPath("images_tilted.xmd")
        fnUntilted = self._getExtraPath("images_untilted.xmd")
        mdUntilted.write(fnUntilted)
        mdTilted.write(fnTilted)

    def createOutputStep(self):
        fnTilted = self._getExtraPath("images_tilted.xmd")
        fnUntilted = self._getExtraPath("images_untilted.xmd")

        # Create outputs SetOfParticles both for tilted and untilted
        imgSetU = self._createSetOfParticles(suffix="Untilted")
        imgSetU.copyInfo(self.uMics)
        imgSetT = self._createSetOfParticles(suffix="Tilted")
        imgSetT.copyInfo(self.tMics)

        sampling = self.getMicSampling() if self._micsOther() else self.getCoordSampling()
        if self._doDownsample():
            sampling *= self.downFactor.get()
        imgSetU.setSamplingRate(sampling)
        imgSetT.setSamplingRate(sampling)

        # set coords from the input, will update later if needed
        imgSetU.setCoordinates(self.inputCoordinatesTiltedPairs.get().getUntilted())
        imgSetT.setCoordinates(self.inputCoordinatesTiltedPairs.get().getTilted())

        # Read untilted and tilted particles on a temporary object (also disabled particles)
        imgSetAuxU = SetOfParticles(filename=':memory:')
        imgSetAuxU.copyInfo(imgSetU)
        readSetOfParticles(fnUntilted, imgSetAuxU, removeDisabled=False)

        imgSetAuxT = SetOfParticles(filename=':memory:')
        imgSetAuxT.copyInfo(imgSetT)
        readSetOfParticles(fnTilted, imgSetAuxT, removeDisabled=False)

        # calculate factor for coords scaling
        factor = 1 / self.samplingFactor
        if self._doDownsample():
            factor /= self.downFactor.get()

        coordsT = self.getCoords().getTilted()
        # For each untilted particle retrieve micId from SetOfCoordinates untilted
        for imgU, coordU in izip(imgSetAuxU, self.getCoords().getUntilted()):
            # FIXME: Remove this check when sure that objIds are equal
            id = imgU.getObjId()
            if id != coordU.getObjId():
                raise Exception('ObjIds in untilted img and coord are not the same!!!!')
            imgT = imgSetAuxT[id]
            coordT = coordsT[id]

            # If both particles are enabled append them
            if imgU.isEnabled() and imgT.isEnabled():
                if self._micsOther() or self._doDownsample():
                    coordU.scale(factor)
                    coordT.scale(factor)
                imgU.setCoordinate(coordU)
                imgSetU.append(imgU)
                imgT.setCoordinate(coordT)
                imgSetT.append(imgT)

        if self.doFlip:
            imgSetU.setIsPhaseFlipped(self.ctfUntilt.hasValue())
            imgSetU.setHasCTF(self.ctfUntilt.hasValue())
            imgSetT.setIsPhaseFlipped(self.ctfTilt.hasValue())
            imgSetT.setHasCTF(self.ctfTilt.hasValue())
        imgSetU.write()
        imgSetT.write()

        # Define output ParticlesTiltPair
        outputset = ParticlesTiltPair(filename=self._getPath('particles_pairs.sqlite'))
        outputset.setTilted(imgSetT)
        outputset.setUntilted(imgSetU)
        for imgU, imgT in izip(imgSetU, imgSetT):
            outputset.append(TiltPair(imgU, imgT))

        outputset.setCoordsPair(self.inputCoordinatesTiltedPairs.get())
        self._defineOutputs(outputParticlesTiltPair=outputset)
        self._defineSourceRelation(self.inputCoordinatesTiltedPairs, outputset)

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        # doFlip can only be selected if CTF information
        # is available on picked micrographs
        if self.doFlip and not self._useCTF():
            errors.append('Phase flipping cannot be performed unless '
                          'CTF information is provided.')

        if self.doNormalize:
            if self.backRadius > int(self.boxSize.get() / 2):
                errors.append("Background radius for normalization should be "
                              "equal or less than half of the box size.")
        return errors

    def _citations(self):
        return ['delaRosaTrevin2013']

    def _summary(self):
        summary = []
        summary.append("Micrographs source: %s"
                       % self.getEnumText('downsampleType'))
        summary.append("Particle box size: %d" % self.boxSize)

        if not hasattr(self, 'outputParticlesTiltPair'):
            summary.append("Output images not ready yet.")
        else:
            summary.append("Particle pairs extracted: %d" %
                           self.outputParticlesTiltPair.getSize())

        if self.doFlip:
            if self.ctfTilt.hasValue() and self.ctfUntilt.hasValue():
                summary.append('Phase flipped for both untilted and tilted particles.')
            elif self.ctfUntilt.hasValue():
                summary.append('Phase flipped only for untilted particles.')
            else:
                summary.append('Phase flipped only for tilted particles.')

        return summary

    def _methods(self):
        methodsMsgs = []

        if self.getStatus() == STATUS_FINISHED:
            msg = "A total of %d particle pairs of size %d were extracted" % \
                  (self.getOutput().getSize(), self.boxSize)
            if self._micsOther():
                msg += " from another set of micrographs: %s" % \
                       self.getObjectTag('inputMicrographsTiltedPair')

            msg += " using coordinates %s" % self.getObjectTag('inputCoordinatesTiltedPairs')
            msg += self.methodsVar.get('')
            methodsMsgs.append(msg)

            if self.doRemoveDust:
                methodsMsgs.append("Removed dust over a threshold of %s." % (self.thresholdDust))
            if self.doInvert:
                methodsMsgs.append("Inverted contrast on images.")
            if self._doDownsample():
                methodsMsgs.append("Particles downsampled by a factor of %0.2f." % self.downFactor)
            if self.doNormalize:
                methodsMsgs.append("Normalization performed of type %s." %
                                   (self.getEnumText('normType')))
            if self.doFlip:
                methodsMsgs.append("Phase flipping was performed.")

        return methodsMsgs

    # --------------------------- UTILS functions ------------------------------
    def _setupBasicProperties(self):
        # Get sampling rate and inputMics according to micsSource type
        self.inputCoords = self.getCoords()
        self.uMics, self.tMics = self.getInputMicrographs()
        ctfUntilt = self.ctfUntilt.get() if self.ctfUntilt.hasValue() else None
        ctfTilt = self.ctfTilt.get() if self.ctfTilt.hasValue() else None
        self.samplingFactor = float(self.getMicSampling() / self.getCoordSampling())

        # create a ctf dict with unt/tilt mics
        self.ctfDict = {}
        for micU in self.uMics:
            if ctfUntilt is not None:
                micBase = pwutils.removeExt(self.getMicNameOrId(micU))
                for ctf in ctfUntilt:
                    ctfMicName = self.getMicNameOrId(ctf.getMicrograph())
                    ctfMicBase = pwutils.removeExt(ctfMicName)
                    if micBase == ctfMicBase:
                        self.ctfDict[micU.clone()] = ctf.clone()
                        break
            else:
                self.ctfDict[micU.clone()] = None

        for micT in self.tMics:
            if ctfTilt is not None:
                micBase = pwutils.removeExt(self.getMicNameOrId(micT))
                for ctf in ctfTilt:
                    ctfMicName = self.getMicNameOrId(ctf.getMicrograph())
                    ctfMicBase = pwutils.removeExt(ctfMicName)
                    if micBase == ctfMicBase:
                        self.ctfDict[micT.clone()] = ctf.clone()
                        break
            else:
                self.ctfDict[micT.clone()] = None

    def getInputMicrographs(self):
        """ Return pairs of micrographs associated to the SetOfCoordinates or
        Other micrographs. """
        if not self._micsOther():

            return self.inputCoordinatesTiltedPairs.get().getUntilted().getMicrographs(), \
                   self.inputCoordinatesTiltedPairs.get().getTilted().getMicrographs()
        else:
            return self.inputMicrographsTiltedPair.get().getUntilted(), \
                   self.inputMicrographsTiltedPair.get().getTilted()

    def getCoords(self):
        return self.inputCoordinatesTiltedPairs.get()

    def getOutput(self):
        if self.hasAttribute('outputParticlesTiltPair') and self.outputParticlesTiltPair.hasValue():
            return self.outputParticlesTiltPair
        else:
            return None

    def getCoordSampling(self):
        return self.getCoords().getUntilted().getMicrographs().getSamplingRate()

    def getMicSampling(self):
        return self.getInputMicrographs()[0].getSamplingRate()

    def _getNewSampling(self):
        newSampling = self.getMicSampling()

        if self._doDownsample():
            # Set new sampling, it should be the input sampling of the used
            # micrographs multiplied by the downFactor
            newSampling *= self.downFactor.get()

        return newSampling

    def notOne(self, value):
        return abs(value - 1) > 0.0001

    def _doDownsample(self):
        return self.downFactor > 1.0

    def _getNormalizeArgs(self):
        if not self.doNormalize:
            return ''

        normType = self.getEnumText("normType")
        args = "--method %s " % normType

        if normType != "OldXmipp":
            bgRadius = self.backRadius.get()
            if bgRadius <= 0:
                bgRadius = int(self.boxSize.get() / 2)
            args += " --background circle %d" % bgRadius

        return args

    def getBoxScale(self):
        """ Computing the sampling factor between input and output.
        We should take into account the differences in sampling rate between
        micrographs used for picking and the ones used for extraction.
        The downsampling factor could also affect the resulting scale.
        """
        f = self.getCoordSampling() / self.getMicSampling()
        return f / self.downFactor.get() if self._doDownsample() else f

    def _micsOther(self):
        """ Return True if other micrographs are used for extract. """
        return self.downsampleType == OTHER

    def _useCTF(self):
        return self.ctfUntilt.hasValue() or self.ctfTilt.hasValue()

    def getMicNameOrId(self, mic):
        return mic.getMicName() or mic.getObjId()
