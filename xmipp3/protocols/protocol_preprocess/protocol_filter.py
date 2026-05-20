# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Roberto Marabini (roberto@cnb.csic.es)
# *
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
"""
Protocols for particles filter operations.
"""

from pyworkflow.object import Float
from pyworkflow.protocol.params import (FloatParam, EnumParam, DigFreqParam,
                                        BooleanParam, PointerParam)

from pwem.objects import ImageDim
from pwem.constants import FILTER_LOW_PASS, FILTER_HIGH_PASS, FILTER_BAND_PASS
from pwem.protocols import ProtFilterParticles, ProtFilterVolumes

from xmipp3.constants import (FILTER_SPACE_FOURIER, FILTER_SPACE_REAL,
                              FILTER_SPACE_WAVELET)
from xmipp3.convert import writeCTFModel
from .protocol_process import XmippProcessParticles, XmippProcessVolumes


class XmippFilterHelper():
    """ Filter operations such as: Fourier or Gaussian. """
    # Some Filter Modes constants to be used locally
    # the special cases of low pass, high pass and band pass
    # should preserve the em.constants values 0, 1 and 2 respectively
    # for properly working of the wizards
    #Fourier filters
    FM_LOW_PASS  = FILTER_LOW_PASS   #0
    FM_HIGH_PASS = FILTER_HIGH_PASS  #1
    FM_BAND_PASS = FILTER_BAND_PASS  #2
    FM_CTF       = 3
    # Real Space Filters
    FM_MEDIAN = 0
    #Wavelets decomposition base
    FM_DAUB4   = 0
    FM_DAUB12  = 1
    FM_DAUB20  = 2
    #Wavelet filters
    FM_REMOVE_SCALE      = 0
    FM_BAYESIAN          = 1
    FM_SOFT_THRESHOLDING = 2
    FM_ADAPTIVE_SOFT     = 3
    FM_CENTRAL           = 4

    #--------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineProcessParams(cls, form, fourierChoices=['low pass', 'high pass', 'band pass', 'ctf']):
        form.addParam('filterSpace', EnumParam, choices=['fourier', 'real', 'wavelets'],
                      default=FILTER_SPACE_FOURIER,
                      label="Filter space")
        form.addParam('filterModeFourier', EnumParam, choices=fourierChoices,
                      default=cls.FM_BAND_PASS,
                      condition='filterSpace == %d' % FILTER_SPACE_FOURIER,
                      label="Filter mode",
                      help='Depending on the filter mode some frequency (freq.) components\n'
                           'are kept and some are removed.\n '
                           '_low pass_: components below *High freq.* are preserved.\n '
                           '_high pass_: components above *Low freq.* are preserved.\n '
                           '_band pass_: components between *Low freq.* and *High freq.* '
                           'are preserved. \n'
                           'ctf: apply first CTF in CTFset to all the particles. This is normally for simulated data.\n'
                           '   : This is not a CTF correction.'
        )

        form.addParam('filterModeReal', EnumParam, choices=['median'],
                      default=cls.FM_MEDIAN,
                      condition='filterSpace == %d' % FILTER_SPACE_REAL,
                      label="Filter mode",
                      help='median: replace each pixel with the median of neighboring pixels.\n'
        )
        form.addParam('filterModeWavelets', EnumParam, choices=['daub4','daub12','daub20'],
                      default=cls.FM_DAUB4,
                      condition='filterSpace == %d' % FILTER_SPACE_WAVELET,
                      label="Filter mode",
                      help='DAUB4: filter using the DAUB4 wavelet transform.\n '
        )

        #String that identifies filter in Fourier Space
        fourierCondition = 'filterSpace == %d and (%s)' % (FILTER_SPACE_FOURIER,
                                                           cls.getModesCondition('filterModeFourier',
                                                                                  cls.FM_LOW_PASS,
                                                                                  cls.FM_HIGH_PASS,
                                                                                  cls.FM_BAND_PASS))
        #String that identifies filter in Real Space
        realCondition    = 'filterSpace == %d and (%s)' % (FILTER_SPACE_REAL,
                                                           cls.getModesCondition('filterModeReal',
                                                                                  cls.FM_MEDIAN))
        #String that identifies filter in Real Space
        waveletCondition = 'filterSpace == %d and (%s)' % (FILTER_SPACE_WAVELET,
                                                           cls.getModesCondition('filterModeWavelets',
                                                                                  cls.FM_DAUB4,
                                                                                  cls.FM_DAUB12,
                                                                                  cls.FM_DAUB20))
        #fourier

        form.addParam('freqInAngstrom', BooleanParam, default=True,
                      condition='filterSpace == %d and filterModeFourier != %d' % (FILTER_SPACE_FOURIER, cls.FM_CTF),
                      label='Provide resolution in Angstroms?',
                      help='If *Yes*, the resolution values for the filter\n'
                           'should be provided in Angstroms. If *No*, the\n'
                           'values should be in normalized frequencies (between 0 and 0.5).')
        # Resolution in Angstroms (inverse of frequencies)
        line = form.addLine('Resolution (A)',
                            condition=fourierCondition + ' and freqInAngstrom',
                            help='Range of resolutions to use in the filter')
        line.addParam('lowFreqA', FloatParam, default=60,
                      condition='(' + cls.getModesCondition('filterModeFourier',
                                                       cls.FM_BAND_PASS, cls.FM_HIGH_PASS) + ') and freqInAngstrom',
                      label='Lowest')
        line.addParam('highFreqA', FloatParam, default=10,
                      condition='(' + cls.getModesCondition('filterModeFourier',
                                                       cls.FM_BAND_PASS, cls.FM_LOW_PASS) + ') and freqInAngstrom',
                      label='Highest')

        form.addParam('freqDecayA', FloatParam, default=100,
                      condition=fourierCondition + ' and freqInAngstrom',
                      label='Decay length',
                      help=('Amplitude decay in a [[https://en.wikipedia.org/'
                            'wiki/Raised-cosine_filter][raised cosine]]'))

        # Normalized frequencies ("digital frequencies")
        line = form.addLine('Frequency (normalized)',
                            condition=fourierCondition + ' and (not freqInAngstrom)',
                            help='Range of frequencies to use in the filter')
        line.addParam('lowFreqDig', DigFreqParam, default=0.02,
                      condition='(' + cls.getModesCondition('filterModeFourier',
                                                       cls.FM_BAND_PASS, cls.FM_HIGH_PASS) + ') and (not freqInAngstrom)',
                      label='Lowest')
        line.addParam('highFreqDig', DigFreqParam, default=0.35,
                      condition='(' + cls.getModesCondition('filterModeFourier',
                                                       cls.FM_BAND_PASS, cls.FM_LOW_PASS) + ') and (not freqInAngstrom)',
                      label='Highest')

        form.addParam('freqDecayDig', FloatParam, default=0.02,
                      condition=fourierCondition + ' and (not freqInAngstrom)',
                      label='Frequency decay',
                      help=('Amplitude decay in a [[https://en.wikipedia.org/'
                            'wiki/Raised-cosine_filter][raised cosine]]'))

        form.addParam('inputCTF', PointerParam, allowsNull=True,
                      condition='filterModeFourier == %d' % cls.FM_CTF,
                      label='CTF Object',
                      pointerClass='CTFModel',
                      help='Object with CTF information if empty it will take the CTF information related with the first particle.\n'
                           'Note that this is normally used with simulated data.')

        #wavelets
        form.addParam('waveletMode',  EnumParam, choices=['remove_scale',
                                                          'bayesian(not implemented)',
                                                          'soft_thresholding',
                                                          'adaptive_soft',
                                                          'central'],
                      default=cls.FM_REMOVE_SCALE,
                      condition='filterSpace == %d' % FILTER_SPACE_WAVELET,
                      label='mode',
                      help='filter mode to be applied in wavelet space')

    @classmethod
    def getLowFreq(cls, protocol):
        if protocol.freqInAngstrom:
            return protocol.getInputSampling() / protocol.lowFreqA.get()
        else:
            return protocol.lowFreqDig.get()

    @classmethod
    def getHighFreq(cls,protocol):
        if protocol.freqInAngstrom:
            return protocol.getInputSampling() / protocol.highFreqA.get()
        else:
            return protocol.highFreqDig.get()

    @classmethod
    def getFreqDecay(cls,protocol):
        if protocol.freqInAngstrom:
            return protocol.getInputSampling() / protocol.freqDecayA.get()
        else:
            return protocol.freqDecayDig.get()

    @classmethod
    def getModesCondition(cls, filterMode, *filterModes):
        return ' or '.join('%s==%d' % (filterMode,fm) for fm in filterModes)
    #        return ' or '.join(['%s==%d' % (fmKey,fmValue) for fmKey,fmValue in zip(filterMode,filterModes)])

    #--------------------------- INSERT steps functions --------------------------------------------
    @classmethod
    def _insertProcessStep(cls, protocol):

        if protocol.filterSpace == FILTER_SPACE_FOURIER:
            lowFreq = cls.getLowFreq(protocol)
            highFreq = cls.getHighFreq(protocol)
            freqDecay = cls.getFreqDecay(protocol)
            print("protocol.freqInAngstrom: ", protocol.freqInAngstrom.get())
            print("lowFreq, highFreq, freqDecay: ", lowFreq, highFreq, freqDecay)
            
            mode = protocol.filterModeFourier.get()

            if mode == cls.FM_LOW_PASS:
                filterStr = " low_pass %f %f " % (highFreq, freqDecay)
            elif mode == cls.FM_HIGH_PASS:
                filterStr = " high_pass %f %f " % (lowFreq, freqDecay)
            elif mode == cls.FM_BAND_PASS:
                filterStr = " band_pass %f %f %f " % (lowFreq, highFreq, freqDecay)
            elif mode == cls.FM_CTF:
                ctfModel = protocol._getTmpPath(protocol.tmpCTF)
                sampling = protocol.getInputSampling()
                filterStr = " ctf %s --sampling %f" % (ctfModel,sampling)
                # Save CTF model too
                protocol._insertFunctionStep("convertCTFXmippStep", ctfModel)
            else:
                raise Exception("Unknown fourier filter mode: %d" % mode)

            args = " --fourier " + filterStr
        elif protocol.filterSpace == FILTER_SPACE_REAL:
            mode = protocol.filterModeReal.get()
            if mode == cls.FM_MEDIAN:
                filterStr = " --median "
            else:
                raise Exception("Unknown real filter mode: %d" % mode)

            args = filterStr
        elif protocol.filterSpace == FILTER_SPACE_WAVELET:
            filterMode = protocol.filterModeWavelets.get()
            filterStr = " --wavelet "
            if filterMode == cls.FM_DAUB4:
                filterStr += " DAUB4 "
            elif filterMode == cls.FM_DAUB12:
                filterStr += " DAUB12 "
            elif filterMode == cls.FM_DAUB20:
                filterStr += " DAUB20 "
            else:
                raise Exception("Unknown wavelets filter mode: %d" % filterMode)

            waveletMode = protocol.waveletMode.get()
            if waveletMode == cls.FM_REMOVE_SCALE:
                filterStr += " remove_scale "
            elif waveletMode == cls.FM_BAYESIAN:
                raise Exception("Bayesian filter not implemented")
            elif waveletMode == cls.FM_SOFT_THRESHOLDING:
                filterStr += " soft_thresholding "
            elif waveletMode == cls.FM_ADAPTIVE_SOFT:
                filterStr += " adaptive_soft "
            elif waveletMode == cls.FM_CENTRAL:
                filterStr += " central "

            args = filterStr
        else:
            raise Exception("Unknown filter space: %d" % protocol.filterSpace.get())

        command = (protocol._args % {'inputFn': protocol.inputFn}) + args
        protocol._insertFunctionStep("filterStep", command)

    @classmethod
    def _summary(cls, protocol):
        summary = []
        def add(x): summary.append('  ' + x)  # short notation
        if protocol.filterSpace == FILTER_SPACE_FOURIER:
            add('Space: *Fourier*')
            Ts = protocol.getInputSampling()
            inA = protocol.freqInAngstrom  # short notation
            lowA, highA = protocol.lowFreqA.get(), protocol.highFreqA.get()
            lowAN = Ts/lowA
            highAN = Ts/highA
            lowN, highN = protocol.lowFreqDig.get(), protocol.highFreqDig.get()
            lowNA = Ts/lowN
            highNA = Ts/highN
            mode = protocol.filterModeFourier.get()
            decayStr = ('*%g* A' % protocol.freqDecayA.get() if inA else
                        '*%g*' % protocol.freqDecayDig.get())
            if mode == cls.FM_LOW_PASS:
                add('Mode: *Low-pass* ( %s, decay at %s )' %
                    ('resolution > *%g* A (normalized freq. %g)' % (highA,highAN) if inA else
                     'normalized frequency < *%g* (%g A)' % (highN,highNA), decayStr))
            elif mode == cls.FM_HIGH_PASS:
                add('Mode: *High-pass* ( %s, decay at %s )' %
                    ('resolution < *%g* A (normalized freq. %g)' % (lowA,lowAN) if inA else
                     'normalized frequency > *%g* (%g A)' % (lowN,lowNA), decayStr))
            elif mode == cls.FM_BAND_PASS:
                add('Mode: *Band-pass* ( %s, decay at %s )' %
                    ('*%g* A (normalized freq. %g) < resolution < *%g* A (normalized freq. %g)' % (highA, highAN, lowA, lowAN) if inA else
                     '*%g* (%g A) < normalized frequency < *%g* (%g A)' % (lowN, lowNA, highN, highNA),
                     decayStr))
        elif protocol.filterSpace == FILTER_SPACE_REAL:
            add('Mode: *Median*')
        elif protocol.filterSpace == FILTER_SPACE_WAVELET:
            add('Space: *Wavelet*')
            filterMode = protocol.filterModeWavelets.get()
            if   filterMode == cls.FM_DAUB4:   add('Wavelet: *Daubechies 4*')
            elif filterMode == cls.FM_DAUB12:  add('Wavelet: *Daubechies 12*')
            elif filterMode == cls.FM_DAUB20:  add('Wavelet: *Daubechies 20*')

            fm = protocol.waveletMode.get()
            if   fm == cls.FM_REMOVE_SCALE:       add('Mode: *Remove scale*')
            elif fm == cls.FM_SOFT_THRESHOLDING:  add('Mode: *Soft thresholding*')
            elif fm == cls.FM_ADAPTIVE_SOFT:      add('Mode: *Adaptive soft*')
            elif fm == cls.FM_CENTRAL:            add('Mode: *Central*')

        return summary

    @classmethod
    def _methods(cls, protocol):
        return ["We filtered the volume(s) using Xmipp [Sorzano2007a]."]

    def getInputSampling(self):
        """ Function to return the sampling rate of input objects.
        Should be implemented for filter volumes and particles.
        """
        pass


class XmippProtFilterParticles(ProtFilterParticles, XmippProcessParticles):
    """ Applies Fourier-based filters to a set of particles to enhance or
    suppress specific frequency components. This helps improve contrast or
    remove noise, preparing particles for further analysis such as picking.

    AI Generated

    ## Overview

    The Filter Particles protocol applies frequency-domain, real-space, or
    wavelet-based filters to a set of particle images.

    Filtering is commonly used in cryo-EM to suppress noise, remove unwanted
    frequency components, enhance contrast, or prepare particles for downstream
    processing. This protocol provides several filtering modes, including low-pass,
    high-pass, band-pass, CTF-based filtering, median filtering, and wavelet
    filtering.

    The main output is a new particle set containing the filtered particle images.

    ## Inputs and General Workflow

    The input is a set of particles.

    The protocol converts the input particles to Xmipp metadata format, applies the
    selected filter using `xmipp_transform_filter`, and writes a new image stack
    and metadata file. The output particle set keeps the input metadata columns
    whenever possible and points to the filtered images.

    Depending on the selected filter space, the user defines frequency limits,
    real-space filter mode, or wavelet parameters.

    ## Input Particles

    The **Input particles** parameter defines the particle set to be filtered.

    The particles can be raw, extracted, aligned, classified, or otherwise
    processed. The protocol does not change particle coordinates, orientations, or
    CTF metadata, except when a CTF model is used to define a filter.

    The output particles preserve the input metadata and replace the image
    locations with the filtered image stack.

    ## Filter Space

    The **Filter space** parameter defines the mathematical domain where the
    filter is applied.

    There are three options:

    **Fourier** applies filters in frequency space. This is the most common choice
    for low-pass, high-pass, band-pass, or CTF-like filtering.

    **Real** applies a real-space filter. In this protocol, the available real-space
    mode is median filtering.

    **Wavelets** applies filtering after a wavelet decomposition.

    The appropriate choice depends on the goal of the filtering.

    ## Fourier Filtering

    Fourier filtering modifies the frequency content of the particle images.

    The available Fourier modes for particles are:

    - low pass;
    - high pass;
    - band pass;
    - CTF.

    Low-pass filtering keeps low-frequency information and suppresses high
    frequencies. High-pass filtering keeps high-frequency information and
    suppresses low frequencies. Band-pass filtering keeps only a selected frequency
    range.

    The CTF option applies a CTF filter to the particles, normally for simulated
    data. It should not be interpreted as a CTF correction.

    ## Low-Pass Filter

    The **low pass** mode preserves frequency components below the selected high
    frequency cutoff.

    In resolution terms, this removes details finer than the selected highest
    resolution. It is often used to reduce high-frequency noise or to prepare
    particles for coarse alignment or classification.

    For example, a low-pass filter at 20 Å keeps broad structural information and
    suppresses finer details.

    ## High-Pass Filter

    The **high pass** mode preserves frequency components above the selected low
    frequency cutoff.

    This removes very low-frequency background or slowly varying intensity trends.

    High-pass filtering should be used cautiously because excessive removal of
    low-frequency information can damage particle contrast and global shape.

    ## Band-Pass Filter

    The **band pass** mode preserves frequency components between the selected low
    and high frequency limits.

    This is useful when the user wants to focus on a specific frequency range,
    removing both very low-frequency background and high-frequency noise.

    The band-pass filter is the default Fourier filtering mode.

    ## Resolution or Normalized Frequency

    The **Provide resolution in Angstroms?** option controls how Fourier cutoffs
    are entered.

    If enabled, the user provides limits as resolutions in angstroms.

    If disabled, the user provides normalized digital frequencies between 0 and
    0.5, where 0.5 corresponds to Nyquist.

    For biological users, angstrom values are usually easier to interpret. Digital
    frequencies are useful for technical workflows where the user wants direct
    control of the Fourier sampling.

    ## Resolution Parameters

    When values are provided in angstroms, the protocol uses:

    - **Lowest** resolution;
    - **Highest** resolution;
    - **Decay length**.

    For a band-pass filter, the preserved range lies between the highest and
    lowest resolution limits. For a low-pass filter, the highest-resolution cutoff
    is used. For a high-pass filter, the lowest-resolution cutoff is used.

    The decay length controls the smooth transition of the filter.

    ## Normalized Frequency Parameters

    When values are provided as normalized frequencies, the protocol uses:

    - **Lowest** frequency;
    - **Highest** frequency;
    - **Frequency decay**.

    Normalized frequencies range from 0 to 0.5.

    A larger normalized frequency corresponds to finer detail. A smaller normalized
    frequency corresponds to coarser information.

    ## Decay Length

    The decay parameter controls the smoothness of the filter transition.

    The filter uses a raised-cosine-like transition rather than an abrupt cutoff.
    This helps reduce ringing artifacts caused by sharp frequency truncation.

    A larger decay produces a smoother transition. A smaller decay produces a
    sharper cutoff.

    ## CTF Filter

    The **ctf** Fourier mode applies a CTF filter to the particles.

    The user may provide a **CTF Object**. If no CTF object is provided, the
    protocol uses the CTF information associated with the first input particle.

    The protocol converts the CTF information to Xmipp format and applies it as a
    filter. This option is normally intended for simulated data and should not be
    used as a replacement for CTF correction.

    ## Real-Space Median Filter

    When **real** filter space is selected, the available mode is **median**.

    Median filtering replaces each pixel by the median of neighboring pixels. This
    can reduce isolated outlier values or salt-and-pepper-like noise while
    preserving edges better than a simple average filter.

    Median filtering modifies particle appearance directly in real space and
    should be used conservatively.

    ## Wavelet Filtering

    When **wavelets** filter space is selected, the protocol performs a wavelet
    decomposition.

    The available wavelet bases are:

    - DAUB4;
    - DAUB12;
    - DAUB20.

    The available wavelet modes are:

    - remove scale;
    - soft thresholding;
    - adaptive soft;
    - central.

    The Bayesian wavelet mode is listed in the interface but is not implemented.

    Wavelet filtering requires particle dimensions to be powers of two. The
    protocol validates this condition.

    ## Output Particles

    The main output is **outputParticles**.

    This output contains the filtered particle images. It preserves the input
    particle metadata and stores the new image locations in the filtered stack.

    The output can be used in downstream workflows such as classification,
    alignment, reconstruction, screening, or visualization.

    ## Validation Rules

    For wavelet filtering, all particle image dimensions must be powers of two.

    If the particle dimensions are not powers of two, the protocol reports a
    validation error.

    The protocol also assumes that CTF filtering is used only when appropriate CTF
    information is available, either from a provided CTF object or from the first
    particle.

    ## Practical Recommendations

    Use low-pass filtering to suppress high-frequency noise or prepare particles
    for coarse processing.

    Use high-pass filtering only when low-frequency background is problematic.

    Use band-pass filtering when both low-frequency background and high-frequency
    noise should be reduced.

    Use angstrom-based frequency parameters unless you specifically need normalized
    digital frequencies.

    Use CTF filtering mainly for simulated data; do not confuse it with CTF
    correction.

    Use wavelet filtering only when the image dimensions are powers of two.

    Inspect representative filtered particles before using the output in expensive
    downstream processing.

    ## Final Perspective

    Filter Particles is a general particle-image filtering protocol.

    For biological users, its value is that it allows controlled suppression or
    selection of image information at different frequency scales. It can improve
    visualization, reduce noise, or prepare particles for downstream processing,
    but excessive filtering can remove useful signal.

    The filtered output should therefore be interpreted as a processed version of
    the original particle set, not as new experimental information.
    """
    _label = 'filter particles'
    tmpCTF = "ctf.xmd"

    def __init__(self, **kwargs):
        ProtFilterParticles.__init__(self, **kwargs)
        XmippProcessParticles.__init__(self, **kwargs)
        self._program = "xmipp_transform_filter"
        self.allowMpi = False
        self.allowThreads = False

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        XmippFilterHelper._defineProcessParams(form)
        
    def _insertProcessStep(self):
        XmippFilterHelper._insertProcessStep(self)

    #--------------------------- RETURN digital or analog freq
    def getLowFreq(self):
        return XmippFilterHelper.getLowFreq(self)

    def getHighFreq(self):
        return XmippFilterHelper.getHighFreq(self)

    def getFreqDecay(self):
        return XmippFilterHelper.getFreqDecay(self)

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertCTFXmippStep(self, ctfModel):
        #defocus comes from here
        inputSet = self.inputParticles.get()

        if self.inputCTF.hasValue():
            ctf = self.inputCTF.get()
            mic = ctf.getMicrograph()
            acquisition = mic.getAcquisition()
            sampling    = mic.getSamplingRate()
        else:
            ctf = inputSet.getFirstItem().getCTF()
            acquisition = inputSet.getAcquisition()
            sampling = inputSet.getSamplingRate()
        # Spherical aberration in mm
        ctf._xmipp_ctfVoltage = Float(acquisition.getVoltage())
        ctf._xmipp_ctfSphericalAberration = Float(acquisition.getSphericalAberration())
        ctf._xmipp_ctfQ0 = Float(acquisition.getAmplitudeContrast())
        ctf._xmipp_ctfSamplingRate = Float(sampling)
        writeCTFModel(ctf, ctfModel)
        ##ROB
        writeCTFModel(ctf, '/tmp/ctf.xmd')

    #--------------------------- STEPS functions ---------------------------------------------------
    def filterStep(self, args):
        args += " -o %s --save_metadata_stack %s --keep_input_columns" % (self.outputStk, self.outputMd)
        self.runJob("xmipp_transform_filter", args)

    def getInputSampling(self):
        return self.inputParticles.get().getSamplingRate()

    def _validate(self):
        if self.filterSpace != FILTER_SPACE_WAVELET:
            return []  # nothing to check

        for n in self.inputParticles.get().getDim():
            while n > 1:  # check if math.log(n, 2) is int, more rudimentary
                if n % 2 != 0:
                    return ["To use the wavelet filter, the input particles",
                            "must have dimensions which are a power of 2,",
                            "but their dimensions are %s." %
                            ImageDim(*self.inputParticles.get().getDim())]
                n /= 2
        return []

    def _summary(self):
        if not hasattr(self, "outputParticles"):
            return ["Protocol has not finished yet."]

        return (["Filtered %d particles using:" % self.outputParticles.getSize()] +
                XmippFilterHelper._summary(self))

    def _methods(self):
        return XmippFilterHelper._methods(self)



class XmippProtFilterVolumes(ProtFilterVolumes, XmippProcessVolumes):
    """ Applies Fourier filters to 3D volumes, adjusting their frequency
    content generating a filter volume as output. Filtering can emphasize
    structural features, reduce noise, or prepare volumes for comparison or
    refinement. The filter can be applied in Fourier, wavelet or real space,
    and can be set as band, high or low pass filter. Also can be set the
    resolution range and the decay length.

    AI Generated

    ## Overview

    The Filter Volumes protocol applies Fourier, real-space, or wavelet filters to
    one volume or to a set of volumes.

    Filtering volumes is useful for reducing noise, limiting the resolution of a
    map, removing low-frequency background, comparing maps at the same resolution,
    or preparing volumes for downstream workflows such as alignment, subtraction,
    visualization, or validation.

    The protocol supports Fourier low-pass, high-pass, and band-pass filters, a
    real-space median filter, and several wavelet filters.

    The main output is a filtered volume or a filtered set of volumes.

    ## Inputs and General Workflow

    The input can be a single volume or a set of volumes.

    The protocol applies the selected filter using `xmipp_transform_filter`. If
    the input is a single volume, one filtered map is produced. If the input is a
    set of volumes, a new filtered volume set is produced while preserving the
    input metadata.

    The user selects the filtering space and then defines the parameters for the
    corresponding filter type.

    ## Input Volumes

    The **Input volumes** parameter defines the map or maps to be filtered.

    The input may be a single 3D volume or a set of 3D volumes. The protocol does
    not align, resize, mask, sharpen, or validate the maps. It only changes their
    frequency or spatial content according to the selected filter.

    The output maps keep the input metadata as far as possible while pointing to
    the filtered files.

    ## Filter Space

    The **Filter space** parameter defines where the filter is applied.

    There are three options:

    **Fourier** applies the filter in frequency space.

    **Real** applies a real-space median filter.

    **Wavelets** applies a wavelet-based filter.

    Fourier filtering is the most common option for cryo-EM maps.

    ## Fourier Filtering

    For volumes, Fourier filtering supports three modes:

    - low pass;
    - high pass;
    - band pass.

    Unlike the particle-filter protocol, the volume-filter protocol does not expose
    the CTF filtering mode.

    Fourier filtering is useful when the user wants to control the resolution range
    of a map.

    ## Low-Pass Filter

    The **low pass** mode keeps low-frequency information and suppresses
    high-frequency components.

    In cryo-EM terms, it removes features finer than the selected resolution. This
    is commonly used to compare maps at the same resolution, reduce noise, or
    prepare a map for initial alignment.

    ## High-Pass Filter

    The **high pass** mode keeps high-frequency components and suppresses
    low-frequency information.

    This can reduce broad background variations, but it should be used cautiously.
    Strong high-pass filtering can damage the overall shape and low-resolution
    contrast of a map.

    ## Band-Pass Filter

    The **band pass** mode keeps frequencies between a lower and an upper cutoff.

    This is useful for focusing on a selected structural scale. It can suppress
    both slow background variations and high-frequency noise.

    Band-pass filtering is the default Fourier mode.

    ## Resolution or Normalized Frequency

    The **Provide resolution in Angstroms?** option controls how frequency limits
    are entered.

    If enabled, the user provides cutoffs as resolution values in angstroms.

    If disabled, the user provides normalized digital frequencies between 0 and
    0.5, where 0.5 is Nyquist.

    Angstrom-based parameters are usually more intuitive for biological
    interpretation. Normalized frequencies are useful for precise technical
    control.

    ## Resolution Parameters

    When angstrom values are used, the filter is defined by:

    - **Lowest** resolution;
    - **Highest** resolution;
    - **Decay length**.

    For low-pass filtering, the highest-resolution cutoff is used. For high-pass
    filtering, the lowest-resolution cutoff is used. For band-pass filtering, the
    preserved range lies between the highest and lowest resolution limits.

    The decay length controls the smooth transition of the filter.

    ## Normalized Frequency Parameters

    When normalized frequency values are used, the filter is defined by:

    - **Lowest** frequency;
    - **Highest** frequency;
    - **Frequency decay**.

    Normalized frequencies range from 0 to 0.5.

    These parameters are internally passed directly to the Fourier filter.

    ## Filter Decay

    The decay parameter controls how gradually the filter transitions between
    preserved and suppressed frequencies.

    A smooth transition helps reduce ringing artifacts that can appear when
    frequency cutoffs are too abrupt.

    The decay is expressed either in angstroms or normalized frequency units,
    depending on the selected input convention.

    ## Real-Space Median Filter

    When **real** filter space is selected, the available mode is **median**.

    The median filter replaces each voxel with the median of its local
    neighborhood. It can reduce isolated spikes or local outliers while preserving
    some edges better than a simple averaging filter.

    Median filtering changes the map directly in real space and should be used
    carefully when quantitative density interpretation is important.

    ## Wavelet Filtering

    When **wavelets** filter space is selected, the protocol performs wavelet-based
    filtering.

    The available wavelet bases are:

    - DAUB4;
    - DAUB12;
    - DAUB20.

    The available wavelet modes are:

    - remove scale;
    - soft thresholding;
    - adaptive soft;
    - central.

    The Bayesian wavelet mode is not implemented.

    Wavelet filtering requires the input volume dimensions to be powers of two.
    The protocol validates this requirement.

    ## Output Volume or Volume Set

    The main output is **outputVol**.

    If the input is a single volume, this output is the filtered volume.

    If the input is a set of volumes, this output is a filtered set of volumes.

    The output can be used in downstream protocols such as map comparison,
    alignment, subtraction, local analysis, or visualization.

    ## Validation Rules

    For wavelet filtering, all volume dimensions must be powers of two.

    If the dimensions are not powers of two, the protocol reports a validation
    error.

    Users should also ensure that the selected frequency cutoffs are meaningful
    for the sampling rate of the input volume.

    ## Interpreting the Filtered Map

    The filtered output is a processed version of the input map.

    Low-pass filtering makes the map smoother and removes finer details. High-pass
    filtering removes broad low-frequency components. Band-pass filtering isolates
    a selected frequency range. Median and wavelet filters modify the map according
    to their own spatial or multiscale criteria.

    Filtering can improve interpretability for some purposes, but it can also
    remove real signal or introduce artifacts if used too aggressively.

    ## Practical Recommendations

    Use low-pass filtering when comparing maps at the same resolution or when
    reducing high-frequency noise.

    Use band-pass filtering when focusing on a specific structural scale.

    Use high-pass filtering only when low-frequency background is clearly
    problematic.

    Use smooth decay values to avoid ringing artifacts.

    Use angstrom-based cutoffs unless normalized frequencies are required.

    Use wavelet filtering only when volume dimensions are powers of two.

    Always inspect the filtered map together with the original map.

    Avoid making biological claims from features that appear only after aggressive
    filtering.

    ## Final Perspective

    Filter Volumes is a general map-filtering protocol.

    For biological users, its value is that it provides controlled manipulation of
    the spatial-frequency content of cryo-EM maps. This can help with
    visualization, comparison, preprocessing, and noise reduction.

    The protocol should be used as a preprocessing or exploratory tool. Its output
    is filtered density, not newly validated structural information.
    """
    _label = 'filter volumes'

    #--------------------------- UTILS functions ---------------------------------------------------

    def __init__(self, **kwargs):
        ProtFilterVolumes.__init__(self, **kwargs)
        XmippProcessVolumes.__init__(self, **kwargs)
        self._program = "xmipp_transform_filter"
        self.allowThreads = False
        self.allowMpi = False

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        XmippFilterHelper._defineProcessParams(form, fourierChoices=['low pass', 'high pass', 'band pass'])
        
    def _insertProcessStep(self):
        XmippFilterHelper._insertProcessStep(self)
        
    #--------------------------- STEPS functions ---------------------------------------------------
    def filterStep(self, args):
        if self._isSingleInput():
            args += " -o %s" % self.outputStk
        else:
            args += " -o %s --save_metadata_stack %s --keep_input_columns" % (self.outputStk, self.outputMd)

        self.runJob("xmipp_transform_filter", args)

    def getInputSampling(self):
        return self.inputVolumes.get().getSamplingRate()

    def _validate(self):
        if self.filterSpace != FILTER_SPACE_WAVELET:
            return []  # nothing to check

        for n in self.inputVolumes.get().getDim():
            while n > 1:  # check if math.log(n, 2) is int, more rudimentary
                if n % 2 != 0:
                    return ["To use the wavelet filter, the input volumes",
                            "must have dimensions which are a power of 2,"
                            "but their dimensions are %s." %
                            ImageDim(*self.inputVolumes.get().getDim())]
                n /= 2
        return []

    def _summary(self):
        if not hasattr(self, "outputVol"):
            return ["Protocol has not finished yet."]

        if hasattr(self.outputVol, "getSize"):
            summary = ["Filtered %d volumes using:" % self.outputVol.getSize()]
        else:
            summary = ["Filtered one volume using:"]

        return summary + XmippFilterHelper._summary(self)

    def _methods(self):
        return XmippFilterHelper._methods(self)

    #--------------------------- RETURN digital or analog freq
    def getLowFreq(self):
        return XmippFilterHelper.getLowFreq(self)

    def getHighFreq(self):
        return XmippFilterHelper.getHighFreq(self)

    def getFreqDecay(self):
        return XmippFilterHelper.getFreqDecay(self)
