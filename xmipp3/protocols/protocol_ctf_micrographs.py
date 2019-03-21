# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              Amaya Jimenez (ajimenez@cnb.csic.es)
# *              Javier Mota Garcia (jmota@cnb.csic.es)
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

import sys
import os

from pyworkflow.object import Set, String
import pyworkflow.em as em
import pyworkflow.em.metadata as md
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as pwconst
import pyworkflow.utils as pwutils

from xmipp3.utils import isMdEmpty
from xmipp3.convert import mdToCTFModel, readCTFModel
from xmippLib import Image


class XmippProtCTFMicrographs(em.ProtCTFMicrographs):
    """ Protocol to estimate CTF on a set of micrographs using Xmipp. """
    _label = 'ctf estimation'

    _criterion = ("ctfCritFirstZero<5 OR ctfCritMaxFreq>20 OR "
                  "ctfCritfirstZeroRatio<0.9 OR ctfCritfirstZeroRatio>1.1 OR "
                  "ctfCritFirstMinFirstZeroRatio>10 OR ctfCritCorr13<0 OR "
                  "ctfCritCtfMargin<0 OR ctfCritNonAstigmaticValidty<0 OR "
                  "ctfCritNonAstigmaticValidty>25 "
                  "OR ctfBgGaussianSigmaU<1000 OR "
                  "ctfCritIceness>1")

    _criterion_phaseplate = ("ctfCritFirstZero<5 OR ctfCritMaxFreq>20 OR "
                  "ctfCritFirstMinFirstZeroRatio>50 AND "
                             "ctfCritFirstMinFirstZeroRatio!=1000 "
                  "OR ctfCritfirstZeroRatio<0.9 OR ctfCritfirstZeroRatio>1.1 OR "
                  "ctfCritNonAstigmaticValidty<=0 OR ctfVPPphaseshift>140 OR " 
                  "ctfCritNonAstigmaticValidty>25 "
                  "OR ctfCritIceness>1.03") #ctfCritCorr13==0 OR "ctfCritFirstMinFirstZeroRatio>50 AND "

    def __init__(self, **args):

        em.ProtCTFMicrographs.__init__(self, **args)

    def _createFilenameTemplates(self):
        prefix = '%(root)s/%(micBase)s_xmipp_ctf'
        _templateDict = {
                        # This templates are relative to a micDir
                        'micrographs': 'micrographs.xmd',
                        'prefix': prefix,
                        'ctfParam': prefix + '.ctfparam',
                        'ctfErrorParam': prefix + '_error.xmd',
                        'psd': prefix + '.psd',
                        'enhanced_psd': prefix + '_enhanced_psd.xmp',
                        'ctfmodel_quadrant': prefix + '_ctfmodel_quadrant.xmp',
                        'ctfmodel_halfplane': prefix + '_ctfmodel_halfplane.xmp',
                        'ctf': prefix + '.xmd',
                        'rejected': prefix + '_rejected.xmd'
                        }
        self._updateFilenamesDict(_templateDict)

    def _defineProcessParams(self, form):
        # Change default value for Automatic downsampling
        param = form.getParam("AutoDownsampling")
        param.setDefault(True)

        form.addParam('doInitialCTF', params.BooleanParam, default=False,
                      label="Use defoci from a previous CTF estimation")
        form.addParam('ctfRelations',params.RelationParam, allowsNull=True,
                      condition='doInitialCTF',
                      relationName=em.RELATION_CTF,
                      attributeName='inputMicrographs',
                      label='Previous CTF estimation',
                      help='Choose some CTF estimation related to input '
                           'micrographs, in case you want to use the defocus '
                           'values found previously')
        form.addParam('findPhaseShift', params.BooleanParam, default=False,
                      label="Find additional phase shift?",
                      help='If the data was collected with phase plate, this '
                           'will find additional phase shift due to phase '
                           'plate',
                      expertLevel=params.LEVEL_ADVANCED)

        form.addParam('doCTFAutoDownsampling', params.BooleanParam,
                      default=True,
                      label="Automatic CTF downsampling detection",
                      expertLevel=pwconst.LEVEL_ADVANCED,
                      help='If this option is chosen, the algorithm '
                           'automatically tries by default the suggested '
                           'Downsample factor; and if it fails, +1; '
                           'and if it fails, -1.')
        form.addParam('refineAmplitudeContrast', params.BooleanParam, default=False,
                      label='Allow amplitude constrast refinement')

    def getInputMicrographs(self):
        return self.inputMicrographs.get()

    # --------------------------- STEPS functions ------------------------------
    def _loadSet(self, inputSet, SetClass, getKeyFunc):
        """ method overrided in order to check if the previous CTF estimation
            is ready when doInitialCTF=True and streaming is activated
        """
        setFn = inputSet.getFileName()
        self.debug("Loading input db: %s" % setFn)
        updatedSet = SetClass(filename=setFn)
        updatedSet.loadAllProperties()
        streamClosed = updatedSet.isStreamClosed()
        initCtfCheck = lambda idItem: True
        if self.doInitialCTF.get():
            ctfSet = em.SetOfCTF(filename=self.ctfRelations.get().getFileName())
            ctfSet.loadAllProperties()
            streamClosed = streamClosed and ctfSet.isStreamClosed()
            if not streamClosed:
                initCtfCheck = lambda idItem: idItem in ctfSet

        newItemDict = em.OrderedDict()
        for item in updatedSet:
            micKey = item.getObjId()  # getKeyFunc(item)
            if micKey not in self.micDict and initCtfCheck(micKey):
                newItemDict[micKey] = item.clone()
        updatedSet.close()
        self.debug("Closed db.")
        return newItemDict, streamClosed


    def calculateAutodownsampling(self,samplingRate, coeff=1.5):
        ctfDownFactor = coeff / samplingRate
        if ctfDownFactor < 1.0:
            ctfDownFactor = 1.0
        return ctfDownFactor

    def _calculateDownsampleList(self, samplingRate):
        
        if self.AutoDownsampling:
            if self.findPhaseShift:
                ctfDownFactor = self.calculateAutodownsampling(samplingRate, 1.1)
            else:
                ctfDownFactor = self.calculateAutodownsampling(samplingRate)
        else:
            ctfDownFactor = self.ctfDownFactor.get()
        downsampleList = [ctfDownFactor]

        if self.doCTFAutoDownsampling:
            downsampleList.append(ctfDownFactor + 1)
            if ctfDownFactor >= 2:
                downsampleList.append(ctfDownFactor - 1)
            else:
                if ctfDownFactor > 1:
                    downsampleList.append((ctfDownFactor + 1) / 2)
        return downsampleList

    def _estimateCTF(self, mic, *args):
        """ Run the estimate CTF program """
        micFn = mic.getFileName()
        micName = mic.getMicName()
        micBase = self._getMicBase(mic)
        micDir = self._getMicrographDir(mic)

        localParams = self.__params.copy()

        localParams['pieceDim'] = self.windowSize.get()
        localParams['ctfmodelSize'] = self.windowSize.get()

        if self.doInitialCTF:
            # getting prevValues (in streaming couldn't be defined yet)
            prevValues = (self.ctfDict[micName] if micName in self.ctfDict
                          else self.getSinglePreviousParameters(mic.getObjId()))

            localParams['defocusU'], localParams['phaseShift0'] = prevValues
            localParams['defocus_range'] = 0.1 * localParams['defocusU']
        else:
            ma = self._params['maxDefocus']
            mi = self._params['minDefocus']
            localParams['defocusU'] = (ma + mi) / 2
            localParams['defocus_range'] = (ma - mi) / 2

            if self.findPhaseShift:
                localParams['phaseShift0'] = self._params['phaseShift0']

        # Create micrograph dir under extra directory
        pwutils.path.makePath(micDir)
        if not os.path.exists(micDir):
            raise Exception("No created dir: %s " % micDir)

        finalName = micFn
        def _getFn(key):
            return self._getFileName(key, micBase=micBase, root=micDir)
        localParams['root'] = _getFn('prefix')
        downsampleList = self._calculateDownsampleList(mic.getSamplingRate())

        try:
            for i, downFactor in enumerate(downsampleList):
                # Downsample if necessary
                if downFactor != 1:
                    # Replace extension by 'mrc' cause there are some formats that
                    # cannot be written (such as dm3)
                    baseFn = pwutils.replaceBaseExt(micFn, 'mrc')
                    finalName = os.path.join(micDir, baseFn)
                    self.runJob("xmipp_transform_downsample",
                                "-i %s -o %s --step %f --method fourier"
                                % (micFn, finalName, downFactor))
                    psd = Image(finalName)
                    psd = psd.getData()
                    if psd.shape[0] < self.windowSize.get():
                        localParams['pieceDim'] = self.windowSize.get()/2
                        localParams['ctfmodelSize'] = self.windowSize.get()/2


                # Update _params dictionary with mic and micDir
                localParams['micFn'] = finalName
                localParams['samplingRate'] = mic.getSamplingRate() * downFactor

                # CTF estimation with Xmipp

                params = self._args % localParams
                params += " --downSamplingPerformed %f" % downFactor
                if not self.doInitialCTF:
                    params += " --selfEstimation "
                self.runJob(self._program, params)


                # Check the quality of the estimation and reject it necessary
                good = self.evaluateSingleMicrograph(mic)
                if good:
                    break

            for key in ['ctfParam', 'psd', 'enhanced_psd', 'ctfmodel_halfplane',
                        'ctfmodel_quadrant', 'ctf']:
                pwutils.moveFile(_getFn(key), self._getExtraPath())

        except Exception as ex:
            print >> sys.stderr, "xmipp_ctf_estimate_from_micrograph has " \
                     "failed with micrograph %s" % finalName

    def _reEstimateCTF(self, mic, ctfModel):
        """ Run the estimate CTF program """
        self._prepareRecalCommand(ctfModel)
        # CTF estimation with Xmipp
        self.runJob(self._program, self._args % self._params)
        mic = ctfModel.getMicrograph()
        self.evaluateSingleMicrograph(mic)

    def _createOutputStep(self):
        pass

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        validateMsgs = []
        # downsampling factor must be greater than 1
        if self.ctfDownFactor.get() < 1:
            validateMsgs.append('Downsampling factor must be >=1.')
        if self.doInitialCTF:
            if not self.ctfRelations.hasValue():
                validateMsgs.append('If you want to use a previous estimation '
                                    'of the CTF, the corresponding set of CTFs '
                                    'is needed')

    def _summary(self):
        summary = em.ProtCTFMicrographs._summary(self)
        if self.methodsVar.hasValue():
            summary.append(self.methodsVar.get())
        return summary

    def _methods(self):
        strMsg = "We calculated the CTF of micrographs %s using Xmipp " \
                 "[Sorzano2007a]" % self.getObjectTag('inputMicrographs')
        strMsg += "."

        if self.methodsVar.hasValue():
            strMsg += " " + self.methodsVar.get()

        if self.hasAttribute('outputCTF'):
            strMsg += '\nOutput set is %s.' % self.getObjectTag('outputCTF')

        return [strMsg]

    def _citations(self):
        papers = ['Sorzano2007a']
        return papers

    # --------------------------- UTILS functions ------------------------------
    def _prepareArgs(self, params):
        self._args = ("--micrograph %(micFn)s --oroot %(root)s "
                      "--sampling_rate %(samplingRate)s --defocusU %("
                      "defocusU)f --defocus_range %(defocus_range)f "
                      "--overlap 0.7 --pieceDim %(pieceDim)s "
                      "--ctfmodelSize %(ctfmodelSize)s --acceleration1D ")

        if self.findPhaseShift:
            self._args += "--phase_shift %(phaseShift0)f --VPP_radius 0.005"
        if self.refineAmplitudeContrast:
            self._args += "--refine_amplitude_contrast"
        for par, val in params.iteritems():
            self._args += " --%s %s" % (par, str(val))

    def getPreviousValues(self, ctf):
        phaseShift0 = 0.0
        if self.findPhaseShift:
            if ctf.hasPhaseShift():
                phaseShift0 = ctf.getPhaseShift()
            else:
                phaseShift0 = 1.57079  # pi/2
            ctfValues = (ctf.getDefocusU(), phaseShift0)
        else:
            ctfValues = (ctf.getDefocusU(), phaseShift0)

        return ctfValues

    def getSinglePreviousParameters(self, micId):
        if self.ctfRelations.hasValue():
            ctf = self.ctfRelations.get()[micId]
            return self.getPreviousValues(ctf)

    def getPreviousParameters(self):
        if self.ctfRelations.hasValue():
            self.ctfDict = {}
            for ctf in self.ctfRelations.get():
                ctfName = ctf.getMicrograph().getMicName()
                self.ctfDict[ctfName] = self.getPreviousValues(ctf)

        if self.findPhaseShift and not self.ctfRelations.hasValue():
            self._params['phaseShift0'] = 1.57079

    def _defineCtfParamsDict(self):
        em.ProtCTFMicrographs._defineCtfParamsDict(self)

        if not hasattr(self, "ctfDict"):
            self.getPreviousParameters()

        self._createFilenameTemplates()
        self._program = 'xmipp_ctf_estimate_from_micrograph'

        # Mapping between base protocol parameters and the package specific
        # command options
        params = self.getCtfParamsDict()
        self.__params = {'kV': params['voltage'],
                         'Cs': params['sphericalAberration'],
                         #'ctfmodelSize': params['windowSize'],
                         'Q0': params['ampContrast'],
                         'min_freq': params['lowRes'],
                         'max_freq': params['highRes'],
                         #'pieceDim': params['windowSize']
                         }

        self._prepareArgs(self.__params)

    def _prepareRecalCommand(self, ctfModel):
        if self.recalculate:
            self._defineRecalValues(ctfModel)
            self._createFilenameTemplates()
            self._program = 'xmipp_ctf_estimate_from_psd_fast'
            self._args = "--psd %(psdFn)s "
            line = ctfModel.getObjComment().split()

            # get the size and the image of psd
            imgPsd = ctfModel.getPsdFile()
            psdFile = pwutils.path.basename(imgPsd)
            imgh = em.ImageHandler()
            size, _, _, _ = imgh.getDimensions(imgPsd)

            mic = ctfModel.getMicrograph()
            micDir = self._getMicrographDir(mic)
            downFactor = self._calculateDownsampleList(mic.getSamplingRate())[0]

            params = dict(self.getCtfParamsDict())
            params.update(self.getRecalCtfParamsDict())
            params.update({'psdFn': os.path.join(micDir, psdFile),
                           'defocusU': float(line[0])
                           })
            # Mapping between base protocol parameters and the package specific
            # command options
            self.__params = {'sampling_rate': params['samplingRate'],
                             'downSamplingPerformed': downFactor,
                             'kV': params['voltage'],
                             'Cs': params['sphericalAberration'],
                             'min_freq': line[3],
                             'max_freq': line[4],
                             'defocusU': params['defocusU'],
                             'Q0': params['ampContrast'],
                             'defocus_range': 5000,
                             'ctfmodelSize': size
                             }

            if self.findPhaseShift:
                fnCTFparam = self._getFileName('ctfParam',
                                               micBase=self._getMicBase(mic))
                mdCTFParam = md.MetaData(fnCTFparam)
                phase_shift = mdCTFParam.getValue(md.MDL_CTF_PHASE_SHIFT,
                                                  mdCTFParam.firstObject())
                self.__params['VPP_radius'] = 0.005
                self.__params['phase_shift'] = phase_shift

            for par, val in self.__params.iteritems():
                self._args += " --%s %s" % (par, str(val))

    def _setPsdFiles(self, ctfModel):
        micBase = self._getMicBase(ctfModel.getMicrograph())
        extra = self._getExtraPath()

        def _getString(key):
            return String(self._getFileName(key, micBase=micBase, root=extra))

        ctfModel._psdFile = _getString('psd')
        ctfModel._xmipp_enhanced_psd = _getString('enhanced_psd')
        ctfModel._xmipp_ctfmodel_quadrant = _getString('ctfmodel_quadrant')
        ctfModel._xmipp_ctfmodel_halfplane = _getString('ctfmodel_halfplane')

    def evaluateSingleMicrograph(self, mic):
        micFn = mic.getFileName()
        micBase = self._getMicBase(mic)
        micDir = self._getMicrographDir(mic)

        def _getStr(key):
            return str(self._getFileName(key, micBase=micBase, root=micDir))

        fnCTF = _getStr('ctfParam')
        mdCTFparam = md.MetaData(fnCTF)
        objId = mdCTFparam.firstObject()
        mdCTFparam.setValue(md.MDL_MICROGRAPH, micFn, objId)
        mdCTFparam.setValue(md.MDL_PSD, _getStr('psd'), objId)
        mdCTFparam.setValue(md.MDL_PSD_ENHANCED, _getStr('enhanced_psd'), objId)
        mdCTFparam.setValue(md.MDL_CTF_MODEL, _getStr('ctfParam'), objId)
        mdCTFparam.setValue(md.MDL_IMAGE1, _getStr('ctfmodel_quadrant'), objId)
        mdCTFparam.setValue(md.MDL_IMAGE2, _getStr('ctfmodel_halfplane'), objId)

        fnEval = _getStr('ctf')
        mdCTFparam.write(fnEval)

        # Evaluate if estimated ctf is good enough
        try:
            self.runJob("xmipp_ctf_sort_psds", "-i %s" % fnEval)
        except Exception:
            pass

        # Check if it is a good micrograph
        fnRejected = _getStr('rejected')
        if self.findPhaseShift:
            criterion = self._criterion_phaseplate
        else:
            criterion = self._criterion
        self.runJob("xmipp_metadata_utilities",
                    '-i %s --query select "%s" -o %s'
                    % (fnEval, criterion, fnRejected))

        retval = True
        if not isMdEmpty(fnRejected):
            retval = False
            mdCTFparam = md.MetaData(fnEval)
            Iceness = mdCTFparam.getValue(md.MDL_CTF_CRIT_ICENESS, 1)
            mdCTFparam.setValue(md.MDL_ENABLED, -1, mdCTFparam.firstObject())
            mdCTFparam.write(fnEval)
            if Iceness > 1.03:
                retval = True

        return retval

    def _createCtfModel(self, mic, updateSampling=True):
        if updateSampling:
            newSampling = mic.getSamplingRate() * self.ctfDownFactor.get()
            mic.setSamplingRate(newSampling)
        ctfParam = self._getFileName('ctf', micBase=self._getMicBase(mic),
                                     root=self._getExtraPath())
        ctfModel2 = readCTFModel(ctfParam, mic)
        ctfModel2.setMicrograph(mic)
        self._setPsdFiles(ctfModel2)
        return ctfModel2

    def _createErrorCtfParam(self, mic):
        ctfParam = self._getFileName('ctfErrorParam',
                                     micBase=self._getMicBase(mic),
                                     root=self._getExtraPath())
        f = open(ctfParam, 'w+')
        lines = """# XMIPP_STAR_1 *
#
data_fullMicrograph
 _ctfSamplingRate -999
 _ctfVoltage -999
 _ctfDefocusU -999
 _ctfDefocusV -999
 _ctfDefocusAngle -999
 _ctfSphericalAberration -999
 _ctfChromaticAberration -999
 _ctfEnergyLoss -999
 _ctfLensStability -999
 _ctfConvergenceCone -999
 _ctfLongitudinalDisplacement -999
 _ctfTransversalDisplacement -999
 _ctfQ0 -999
 _ctfK -999
 _ctfEnvR0 -999
 _ctfEnvR1 -999
 _ctfEnvR2 -999
 _ctfBgGaussianK -999
 _ctfBgGaussianSigmaU -999
 _ctfBgGaussianSigmaV -999
 _ctfBgGaussianCU -999
 _ctfBgGaussianCV -999
 _ctfBgGaussianAngle -999
 _ctfBgSqrtK -999
 _ctfBgSqrtU -999
 _ctfBgSqrtV -999
 _ctfBgSqrtAngle -999
 _ctfBgBaseline -999
 _ctfBgGaussian2K -999
 _ctfBgGaussian2SigmaU -999
 _ctfBgGaussian2SigmaV -999
 _ctfBgGaussian2CU -999
 _ctfBgGaussian2CV -999
 _ctfBgGaussian2Angle -999
 _ctfBgR1 -999
 _ctfBgR2 -999
 _ctfBgR3 -999
 _ctfX0 -999
 _ctfXF -999
 _ctfY0 -999
 _ctfYF -999
 _ctfCritFitting -999
 _ctfCritCorr13 -999
 _ctfVPPphaseshift -999
 _ctfVPPRadius -999
 _ctfCritIceness -999
 _CtfDownsampleFactor -999
 _ctfCritPsdStdQ -999
 _ctfCritPsdPCA1 -999
 _ctfCritPsdPCARuns -999
 _micrograph NULL
 _psd NULL
 _psdEnhanced NULL
 _ctfModel NULL
 _image1 NULL
 _image2 NULL
 _enabled -1
 _ctfCritFirstZero -999
 _ctfCritMaxFreq -999
 _ctfCritDamping -999
 _ctfCritfirstZeroRatio -999
 _ctfEnvelopePlot -999
 _ctfCritFirstMinFirstZeroRatio -999
 _ctfCritCtfMargin -999
 _ctfCritNonAstigmaticValidty -999
 _ctfCritPsdCorr90 -999
 _ctfCritPsdInt -999
 _ctfCritNormality -999
"""
        f.write(lines)
        f.close()
        return ctfParam

    def _getMicBase(self, mic):
        return pwutils.removeBaseExt(mic.getFileName())
