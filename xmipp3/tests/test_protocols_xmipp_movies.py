# **************************************************************************
# *
# * Authors:    Laura del Cano (ldelcano@cnb.csic.es)
# *             Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
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

from os.path import abspath
from pyworkflow.tests import *

from xmipp3.convert import *
from xmipp3.protocols import *
from pwem.objects import SetOfMovies, MovieAlignment
from pwem.protocols import ProtImportMovies, ProtImportCoordinates
import pyworkflow.utils as pwutils
from contextlib import redirect_stdout

# Some utility functions to import movies that are used in several tests.
class TestXmippBase(BaseTest):
    @classmethod
    def setData(cls):
        cls.dataset = DataSet.getDataSet('movies')
        cls.movie1 = cls.dataset.getFile('qbeta/qbeta.mrc')
        cls.movie2 = cls.dataset.getFile('cct/cct_1.em')
    
    @classmethod
    def runImportMovie(cls, pattern, samplingRate, voltage, scannedPixelSize,
                       magnification, sphericalAberration, dosePerFrame=None):
        """ Run an Import micrograph protocol. """

        kwargs = {
                 'filesPath': pattern,
                 'magnification': magnification,
                 'voltage': voltage,
                 'sphericalAberration': sphericalAberration,
                 'dosePerFrame' : dosePerFrame
                  }

        # We have two options: pass the SamplingRate or
        # the ScannedPixelSize + microscope magnification
        if samplingRate is not None:
            kwargs.update({'samplingRateMode': 0,
                           'samplingRate': samplingRate})
        else:
            kwargs.update({'samplingRateMode': 1,
                           'scannedPixelSize': scannedPixelSize})

        cls.protImport = cls.newProtocol(ProtImportMovies, **kwargs)
        cls.proj.launchProtocol(cls.protImport, wait=True)

        if cls.protImport.isFailed():
            raise Exception("Protocol has failed. Error: ",
                            cls.protImport.getErrorMessage())

        # Check that input movies have been imported (a better way to do this?)
        if cls.protImport.outputMovies is None:
            raise Exception('Import of movies: %s, failed, '
                            'outputMovies is None.' % pattern)

        return cls.protImport
    
    @classmethod
    def runImportMovie1(cls, pattern):
        """ Run an Import movie protocol. """
        return cls.runImportMovie(pattern, samplingRate=1.14, voltage=300,
                                  sphericalAberration=2.26, dosePerFrame=1.5,
                                  scannedPixelSize=None, magnification=50000)
    
    @classmethod
    def runImportMovie2(cls, pattern):
        """ Run an Import movie protocol. """
        return cls.runImportMovie(pattern, samplingRate=1.4, voltage=300,
                                  sphericalAberration=2.7, dosePerFrame=1.5,
                                  scannedPixelSize=None,
                                  magnification=61000)


class TestCorrelationAlignment(BaseTest):
    @classmethod
    def setData(cls):
        cls.ds = DataSet.getDataSet('movies')

    @classmethod
    def runImportMovies(cls, pattern, **kwargs):
        """ Run an Import micrograph protocol. """
        # We have two options: passe the SamplingRate or
        # the ScannedPixelSize + microscope magnification
        params = {'samplingRate': 1.14,
                  'voltage': 300,
                  'sphericalAberration': 2.7,
                  'magnification': 50000,
                  'scannedPixelSize': None,
                  'filesPattern': pattern
                  }
        if 'samplingRate' not in kwargs:
            del params['samplingRate']
            params['samplingRateMode'] = 0
        else:
            params['samplingRateMode'] = 1

        params.update(kwargs)

        protImport = cls.newProtocol(ProtImportMovies, **params)
        cls.launchProtocol(protImport)
        return protImport

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.setData()
        cls.protImport1 = cls.runImportMovies(cls.ds.getFile('qbeta/qbeta.mrc'),
                                              magnification=50000)
        cls.protImport2 = cls.runImportMovies(cls.ds.getFile('cct/cct_1.em'),
                                              magnification=61000)

    def _checkMicrographs(self, protocol):
        self.assertIsNotNone(getattr(protocol, 'outputMicrographs', None),
                             "Output SetOfMicrographs were not created.")

    def _checkAlignment(self, movie, goldRange, goldRoi):
        alignment = movie.getAlignment()
        range = alignment.getRange()
        msgRange = "Alignment range must be %s (%s) and it is %s (%s)"
        self.assertEqual(goldRange, range, msgRange
                         % (goldRange, range, type(goldRange), type(range)))
        roi = alignment.getRoi()
        msgRoi = "Alignment ROI must be %s (%s) and it is %s (%s)"
        self.assertEqual(goldRoi, roi,
                         msgRoi % (goldRoi, roi, type(goldRoi), type(roi)))

    def test_qbeta_cpu(self):
        prot = self.newProtocol(XmippProtFlexAlign,doPSD=True, useGpu=False, doLocalAlignment=False)
        prot.inputMovies.set(self.protImport1.outputMovies)
        self.launchProtocol(prot)

        self._checkMicrographs(prot)
        self._checkAlignment(prot.outputMovies[1],
                             (1,7), [0, 0, 0, 0])

    def test_qbeta(self):
        prot = self.newProtocol(XmippProtFlexAlign,doPSD=True)
        prot.inputMovies.set(self.protImport1.outputMovies)
        self.launchProtocol(prot)

        self._checkMicrographs(prot)
        self._checkAlignment(prot.outputMovies[1],
                             (1,7), [0, 0, 0, 0])

    def test_qbeta_patches(self):
        prot = self.newProtocol(XmippProtFlexAlign,doPSD=True, patchX=7, patchY=7)
        prot.inputMovies.set(self.protImport1.outputMovies)
        self.launchProtocol(prot)

        self._checkMicrographs(prot)
        self._checkAlignment(prot.outputMovies[1],
                             (1,7), [0, 0, 0, 0])

    def test_qbeta_corrDownscale(self):
        prot = self.newProtocol(XmippProtFlexAlign,doPSD=True, corrDownscale=3)
        prot.inputMovies.set(self.protImport1.outputMovies)
        self.launchProtocol(prot)

        self._checkMicrographs(prot)
        self._checkAlignment(prot.outputMovies[1],
                             (1,7), [0, 0, 0, 0])

    def test_cct(self):
        prot = self.newProtocol(XmippProtFlexAlign,
                                doSaveMovie=True,
                                doPSD=True)
        prot.inputMovies.set(self.protImport2.outputMovies)
        self.launchProtocol(prot)

        self._checkMicrographs(prot)
        self._checkAlignment(prot.outputMovies[1],
                             (1,7), [0, 0, 0, 0])

    def test_controlPoints(self):
        prot = self.newProtocol(XmippProtFlexAlign,
                                doSaveMovie=False,
                                doPSD=False,
                                autoControlPoints=False,
                                skipAutotuning=True,
                                controlPointY=9,
                                objLabel="TestControlPoints(ShouldFail)")
        prot.inputMovies.set(self.protImport2.outputMovies)
        with self.assertRaises(Exception,
                               msg=("Protocol should fail because number of control points is higher "
                                    "than number of patches for local alignment")):
            with redirect_stdout(None):
                self.launchProtocol(prot)


class TestEstimateGain(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        ds = DataSet.getDataSet('movies')

        # Reduce input movie size to speed-up gain computation
        ih = ImageHandler()
        inputFn = ds.getFile('ribo/Falcon_2012_06_12-14_33_35_0_movie.mrcs')
        outputFn = cls.proj.getTmpPath(abspath(basename(inputFn)))

        frameImg = ih.createImage()
        xdim, ydim, zdim, ndim = ih.getDimensions(inputFn)
        n = int(max(zdim, ndim) / 2)  # also half of the frames
        print("Scaling movie: %s -> %s" % (inputFn, outputFn))
        pwutils.cleanPath(outputFn)
        for i in range(1, n+1):
            frameImg.read((i, inputFn))
            frameImg.scale(int(xdim/2), int(ydim/2))
            frameImg.write((i, outputFn))

        args = cls.getArgs(outputFn)
        cls.protImport = cls.newProtocol(ProtImportMovies, **args)
        cls.launchProtocol(cls.protImport)

    @classmethod
    def getArgs(self, filesPath, pattern=''):
        return {'importFrom': ProtImportMovies.IMPORT_FROM_FILES,
                'filesPath': filesPath,
                'filesPattern': pattern,
                'amplitudConstrast': 0.1,
                'sphericalAberration': 2.,
                'voltage': 300,
                'samplingRate': 3.54 * 2
                }

    def test_estimate(self):
        protGain = self.newProtocol(XmippProtMovieGain,
                                    objLabel='estimate gain',
                                    estimateGain=True,
                                    estimateResidualGain=True,
                                    estimateOrientation=False,
                                    normalizeGain=False)
        protGain.inputMovies.set(self.protImport.outputMovies)
        self.launchProtocol(protGain)


class TestMaxShift(BaseTest):
    @classmethod
    def setData(cls):
        cls.ds = DataSet.getDataSet('movies')

    @classmethod
    def runImportMovies(cls, pattern, **kwargs):
        """ Run an Import movies protocol. """
        # We have two options: passe the SamplingRate or
        # the ScannedPixelSize + microscope magnification
        params = {'samplingRate': 1.14,
                  'voltage': 300,
                  'sphericalAberration': 2.7,
                  'magnification': 50000,
                  'scannedPixelSize': None,
                  'filesPattern': pattern,
                  'dosePerFrame': 123
                  }
        if 'samplingRate' not in kwargs:
            del params['samplingRate']
            params['samplingRateMode'] = 0
        else:
            params['samplingRateMode'] = 1

        params.update(kwargs)

        protImport = cls.newProtocol(ProtImportMovies, **params)
        cls.launchProtocol(protImport)
        return protImport

    @classmethod
    def runAlignMovMics(cls):  # do SAVE averaged mics and dose weighted mics
        protAlign = cls.newProtocol(XmippProtFlexAlign,
                                    alignFrame0=1, alignFrameN=0,
                                    doLocalAlignment=False, useGpu=True,
                                    objLabel='Movie alignment (SAVE mic)',
                                    doSaveAveMic=True,
                                    binFactor=1)
        protAlign.inputMovies.set(cls.protImport.outputMovies)
        cls.launchProtocol(protAlign)

        return protAlign.outputMovies

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.setData()
        fn = 'Falcon_2012_06_12-*0_movie.mrcs'
        cls.protImport = cls.runImportMovies(cls.ds.getFile(fn))
        cls.alignedMovMics = cls.runAlignMovMics()

    def _checkMaxShiftFiltering(self, protocol, label, hasMic, hasDw=False, noBin=False, results=[]):
        """ Check if outputSets are right.
              If hasMic=True then it's checked that micrographs are generated.
              results = [True, False]  # means first movie should pass whereas second not
        """

        def assertOutput(outputName, ids=[1, 2]):
            """ Check if outputName exists and if so, if it's right. (for each id)
            """
            print("checking '%s' in movies %s" % (outputName, ids))
            targetSamplingRate = protocol.inputMovies.get().getSamplingRate()
            if 'Micrographs' in outputName:
                # The Mics has n=1
                if hasDw or noBin:
                    inputDim = (1950, 1950, 1)
                else:
                    # The corr. align. prot. crops the micrographs and is binned
                    inputDim = (975, 975, 1)
                    targetSamplingRate *= 2
            else:
                inputDim = protocol.inputMovies.get().getDim()

            DWstr = ' (DoseWeighted)' if hasDw else ''
            for itemId in ids:
                output = getattr(protocol, outputName, None)
                self.assertIsNotNone(output, "%s (accepted) were not created. "
                                             "Bad filtering in %s test%s."
                                     % (outputName, label, DWstr))
                self.assertIsNotNone(output[itemId], "%s (accepted) were not "
                                            "created. Bad filtering in %s test%s."
                                     % (outputName, label, DWstr))
                self.assertEqual(output[itemId].getDim(), inputDim,
                                 "The size of the movies/mics has changed "
                                 "for %s test%s." % (label, DWstr))
                self.assertEqual(output[itemId].getSamplingRate(),
                                 targetSamplingRate,
                                 "The samplig rate is incorrect for %s test%s."
                                 % (label, DWstr))

        if all(results):
            #  Checking if only the accepted set is created and
            #    its items have the good size and sampling rate
            assertOutput('outputMovies')
            if hasMic:
                assertOutput('outputMicrographs')
            if hasDw:
                assertOutput('outputMicrographsDoseWeighted')

            #  Checking if the Movies/MicsDiscarded set are not created
            self.assertIsNone(getattr(protocol, 'outputMoviesDiscarded', None),
                              "outputMoviesDiscarded were created. "
                              "Bad filtering in %s test." % label)
            if hasMic:
                outMics = getattr(protocol, 'outputMicrographsDiscarded', None)
                self.assertIsNone(outMics, "outputMicrographsDiscarded were "
                                   "created. Bad filtering in %s test." % label)
        elif not any(results):
            #  Checking if only the discarded set is crated and
            #    its items have the good size and sampling rate
            assertOutput('outputMoviesDiscarded')
            if hasMic:
                assertOutput('outputMicrographsDiscarded')
            if hasDw:
                assertOutput('outputMicrographsDoseWeightedDiscarded')

            #  Checking if the Movie (accepted) set is not created
            self.assertIsNone(getattr(protocol, 'outputMovies', None),
                              "outputMovies (accepted)t were created. "
                              "Bad filtering")
            if hasMic:
                self.assertIsNone(getattr(protocol, 'outputMicrographs', None),
                                  "outputMicrographs (accepted)t were created. "
                                  "Bad filtering")
        else:
            # Check if the passed and rejected movies corresponds to the goods.
            assertOutput('outputMovies', ids=[results.index(True)+1])
            assertOutput('outputMoviesDiscarded', ids=[results.index(False)+1])
            if hasMic:
                assertOutput('outputMicrographs', ids=[results.index(True)+1])
                assertOutput('outputMicrographsDiscarded', ids=[results.index(False)+1])
    
    def doFilter(self, inputMovies, rejType, label, mxFm=0.12, mxMo=1.01):
        """ Template for the movieMaxShift protocol.
            Default thresholds here should discard one movie and let pass the other
        """
        protMaxShift = self.newProtocol(XmippProtMovieMaxShift,
                                        inputMovies=inputMovies,
                                        maxFrameShift=mxFm,
                                        maxMovieShift=mxMo,
                                        rejType=rejType,
                                        objLabel=label)
        self.launchProtocol(protMaxShift)
        return protMaxShift

    # ------- the Tests ---------------------------------------
    # Note: the shift in these movies is very small. Especially in combination with binning,
    # we're getting to the limit of shift we can detect
    # Movie 1: maxGlobalShift 1.0017 maxFrameShift 0.124
    # Movie 2: maxGlobalShift 1.026 maxFrameShift 0.112

    def testFilterFrame(self):
        """ This must discard the second movie for a Frame shift.
        """
        label = 'maxShift by Frame'
        rejType = XmippProtMovieMaxShift.REJ_FRAME

        protDoMic = self.doFilter(self.alignedMovMics, rejType, label, 0.12) # roughly half the precision of the non-binned version
        self._checkMaxShiftFiltering(protDoMic, label, noBin=True, hasMic=True, results=[False, True])

    def testFilterMovie(self): 
        """ This must discard the second movie for a Global shift.
        """
        label = 'maxShift by Movie'
        rejType = XmippProtMovieMaxShift.REJ_MOVIE

        protDoMic = self.doFilter(self.alignedMovMics, rejType, label, 0.13, 1.01) # roughly half the precision of the non-binned version
        self._checkMaxShiftFiltering(protDoMic, label, noBin=True, hasMic=True, results=[True, False])

    def testFilterAnd(self): 
        """ This must discard the second movie for AND.
        """
        label = 'maxShift AND'
        rejType = XmippProtMovieMaxShift.REJ_AND

        protDoMic = self.doFilter(self.alignedMovMics, rejType, label, 0.11, 1.01) # roughly half the precision of the non-binned version
        self._checkMaxShiftFiltering(protDoMic, label, noBin=True, hasMic=True, results=[True, False])

    def testFilterOrFrame(self):
        """ This must discard the second movie for OR (Frame).
        """
        label = 'maxShift OR (by frame)'
        rejType = XmippProtMovieMaxShift.REJ_OR

        protDoMic = self.doFilter(self.alignedMovMics, rejType, label, 0.12, 1.03) # roughly half the precision of the non-binned version
        self._checkMaxShiftFiltering(protDoMic, label, noBin=True,  hasMic=True, results=[False, True])

    def testFilterOrMovie(self): 
        """ This must discard the second movie for OR (Movie).
        """
        label = 'maxShift OR (by movie)'
        rejType = XmippProtMovieMaxShift.REJ_OR

        protDoMic = self.doFilter(self.alignedMovMics, rejType, label, 0.13, 1.01) # roughly half the precision of the non-binned version
        self._checkMaxShiftFiltering(protDoMic, label, noBin=True, hasMic=True, results=[True, False])

    def testFilterRejectBoth(self):
        """ This must discard both movies.
        """
        label = 'maxShift REJECT both'
        rejType = XmippProtMovieMaxShift.REJ_OR

        protDoMic = self.doFilter(self.alignedMovMics, rejType, label, mxMo=1.0)
        self._checkMaxShiftFiltering(protDoMic, label,  noBin=True, hasMic=True, results=[False, False])

    def testFilterAcceptBoth(self):
        """ This must accept both movies.
        """
        label = 'maxShift ACCEPT both'
        rejType = XmippProtMovieMaxShift.REJ_AND

        protDoMic = self.doFilter(self.alignedMovMics, rejType, label, mxMo=5)
        self._checkMaxShiftFiltering(protDoMic, label,  noBin=True, hasMic=True, results=[True, True])


class TestMovieDoseAnalysis(BaseTest):

    @classmethod
    def setData(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('relion30_tutorial')

    @classmethod
    def runImportMovies(cls):
        protImport = cls.newProtocol(
            ProtImportMovies,
            filesPath=cls.ds.getFile('Movies/'),
            filesPattern='*.tiff',
            samplingRateMode=0,
            samplingRate=0.885,
            magnification=50000,
            scannedPixelSize=7.0,
            voltage=200,
            sphericalAberration=1.4,
            doseInitial=0.0,
            dosePerFrame=1.277,
            gainFile=cls.ds.getFile("Movies/gain.mrc")
        )
        protImport.setObjLabel('import 24 movies')
        protImport.setObjComment('Relion 3 tutorial movies:\n\n'
                                 'Microscope Jeol Cryo-ARM 200\n'
                                 'Data courtesy of Takyuki Kato in the Namba '
                                 'group\n(Osaka University, Japan)')
        return cls.launchProtocol(protImport)


    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.setData()
        cls.protImport = cls.runImportMovies()

    # ------- Tests ---------------------------------------
    def testDoseAnalysisAssert(self):
        """ This must create two sets of movies.
        """
        label = 'Dose Analysis Assert'
        protPoisson = self.newProtocol(XmippProtMovieDoseAnalysis,
                                       objLabel=label,
                                       n_samples=10,
                                       movieStep=4
                                       )
        protPoisson.inputMovies.set(self.protImport.outputMovies)
        self.launchProtocol(protPoisson)

        self.assertIsNotNone(getattr(protPoisson, 'outputMoviesDiscarded', None),
                          "outputMoviesDiscarded were not created. "
                          "Bad filtering in test.")

        self.assertIsNotNone(getattr(protPoisson, 'outputMovies', None),
                             "outputMovies were not created. "
                             "Bad filtering in test.")

    def testDoseAnalysisFiltering(self):
        """ This must discard movies by dose analysis.
        """
        label = 'Dose Analysis Filter'
        protPoisson = self.newProtocol(XmippProtMovieDoseAnalysis,
                                       objLabel=label,
                                       n_samples=24,
                                       movieStep=4
                                       )
        protPoisson.inputMovies.set(self.protImport.outputMovies)
        self.launchProtocol(protPoisson)

        sizeAccepted = protPoisson.outputMovies.getSize()
        self.assertEqual(sizeAccepted, 19, 'Number of accepted movies must be 19 and its '
                                           '%d' % sizeAccepted)

        sizeDiscarded = protPoisson.outputMoviesDiscarded.getSize()
        self.assertEqual(sizeDiscarded, 5, 'Number of accepted movies must be 5 and its '
                                            '%d' % sizeDiscarded)


class TestMovieAlignmentConsensus(BaseTest):
    @classmethod
    def setData(cls):
        cls.ds = DataSet.getDataSet('movies')

    @classmethod
    def runImportMovies(cls, pattern, **kwargs):
        """ Run an Import movies protocol. """
        # We have two options: passes the SamplingRate or
        # the ScannedPixelSize + microscope magnification
        params = {'samplingRate': 1.14,
                  'voltage': 300,
                  'sphericalAberration': 2.7,
                  'magnification': 50000,
                  'scannedPixelSize': None,
                  'filesPattern': pattern,
                  'dosePerFrame': 123
                  }
        if 'samplingRate' not in kwargs:
            del params['samplingRate']
            params['samplingRateMode'] = 0
        else:
            params['samplingRateMode'] = 1

        params.update(kwargs)

        protImport = cls.newProtocol(ProtImportMovies, **params)
        cls.launchProtocol(protImport)
        return protImport

    @classmethod
    def runAlignMovies1(cls):
        protAlign = cls.newProtocol(XmippProtFlexAlign,
                                    alignFrame0=1, alignFrameN=0,
                                    doLocalAlignment=False, useGpu=False,
                                    objLabel='Reference movie alignment',
                                    doSaveAveMic=True)

        protAlign.inputMovies.set(cls.protImport.outputMovies)
        cls.launchProtocol(protAlign)
        return protAlign

    @classmethod
    def runAlignMovies2(cls):
        protAlign2 = cls.newProtocol(XmippProtFlexAlign,
                                    alignFrame0=1, alignFrameN=0,
                                    maxResForCorrelation=20,
                                    doLocalAlignment=False, useGpu=False,
                                    objLabel='Target movie alignment',
                                    doSaveAveMic=True)

        protAlign2.inputMovies.set(cls.protImport.outputMovies)
        cls.launchProtocol(protAlign2)
        return protAlign2

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.setData()
        fn = 'Falcon_2012_06_12-*0_movie.mrcs'
        cls.protImport = cls.runImportMovies(cls.ds.getFile(fn))
        cls.align1 = cls.runAlignMovies1()
        cls.align2 = cls.runAlignMovies2()

    def testMovieAlignmentConsensusFiltering1(self):
        """ This must discard movies by movie alignment consensus.
        """
        label = 'Alignment consensus 0.5 correlation limit'
        protConsensus1 = self.newProtocol(XmippProtConsensusMovieAlignment,
                                          objLabel=label,
                                          minConsCorrelation=0.5,
                                          trajectoryPlot=True
                                          )

        protConsensus1.inputMovies1.set(self.align1)
        protConsensus1.inputMovies1.setExtended("outputMovies")

        protConsensus1.inputMovies2.set(self.align2)
        protConsensus1.inputMovies2.setExtended("outputMovies")

        self.launchProtocol(protConsensus1)

        sizeAccepted = protConsensus1.outputMovies.getSize()
        self.assertEqual(sizeAccepted, 2, 'Number of accepted movies must be 2 and its %d' % sizeAccepted)

    def testMovieAlignmentConsensusFiltering2(self):
        """ This must discard movies by movie alignment consensus.
        """
        label = 'Alignment consensus 0.9 correlation limit'
        protConsensus2 = self.newProtocol(XmippProtConsensusMovieAlignment,
                                          objLabel=label,
                                          minConsCorrelation=0.9,
                                          minRangeShift=0.01,
                                          trajectoryPlot=True
                                          )

        protConsensus2.inputMovies1.set(self.align1)
        protConsensus2.inputMovies1.setExtended("outputMovies")

        protConsensus2.inputMovies2.set(self.align2)
        protConsensus2.inputMovies2.setExtended("outputMovies")

        self.launchProtocol(protConsensus2)

        sizeDiscarded = protConsensus2.outputMoviesDiscarded.getSize()
        self.assertEqual(sizeDiscarded, 2, 'Number of discarded movies must be 0 and its %d' % sizeDiscarded)


class TestMovieDoseAnalysisState(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testRuntimeStateIsNotSharedBetweenInstances(self):
        prot1 = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot2 = self.newProtocol(XmippProtMovieDoseAnalysis)

        prot1.stats[1] = {'mean': 1.0}
        prot1.meanDoseList.append(1.0)
        prot1.medianDoseTemporal.append(1.0)
        prot1.medianDifferences.append(0.0)

        self.assertEqual(prot2.stats, {})
        self.assertEqual(prot2.meanDoseList, [])
        self.assertEqual(prot2.medianDoseTemporal, [])
        self.assertEqual(prot2.medianDifferences, [])

    def testRuntimeStateIsRestoredFromOutputs(self):
        class OutputMovie:
            def __init__(self, movieId, mean, diff):
                self.movieId = movieId
                self.mean = mean
                self.diff = diff

            def getObjId(self):
                return self.movieId

            def getAttributeValue(self, name, defaultValue=None):
                values = {'_MEAN_DOSE_PER_ANGSTROM2': self.mean, '_DIFF_TO_DOSE_PER_ANGSTROM2': self.diff}
                return values.get(name, defaultValue)

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.outputMovies = [OutputMovie(3, 1.3, 3.0), OutputMovie(1, 1.1, 1.0)]
        prot.outputMoviesDiscarded = [OutputMovie(2, 1.2, 2.0)]

        prot._restoreRuntimeStateFromOutputs()

        self.assertEqual(prot.meanDoseList, [1.1, 1.2, 1.3])
        self.assertEqual(prot.medianDoseTemporal, [1.1, 1.2, 1.3])
        self.assertEqual(prot.medianDifferences, [1.0, 2.0, 3.0])

    def testContinuePreservesPartialDoseWindow(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId, mean=None, diff=None, globalMedian=None, usingExperimental=None):
                self.movieId = movieId
                self.mean = mean
                self.diff = diff
                self.globalMedian = globalMedian
                self.usingExperimental = usingExperimental

            def getObjId(self):
                return self.movieId

            def getAttributeValue(self, name, defaultValue=None):
                values = {
                    '_MEAN_DOSE_PER_ANGSTROM2': self.mean,
                    '_DIFF_TO_DOSE_PER_ANGSTROM2': self.diff,
                    '_GLOBAL_DOSE_PER_ANGSTROM2': self.globalMedian,
                    '_USING_EXPERIMENTAL_DOSE': self.usingExperimental
                }
                return values.get(name, defaultValue)

            def setFramesRange(self, framesRange):
                pass

        class OutputSet(list):
            def getIdSet(self):
                return {movie.getObjId() for movie in self}

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, window=3, percentage_window=101)
        prot.outputMovies = OutputSet([Movie(1, 1.0, 0.0, 1.0, True)])
        prot.outputMoviesDiscarded = OutputSet([Movie(2, 3.0, 200.0, 1.0, True)])
        prot._restoreRuntimeStateFromOutputs()

        prot.stats = {3: {'mean': 5.0, 'std': 0.0, 'min': 5.0, 'max': 5.0}}
        prot.meanDoseById[3] = 5.0
        prot.insertedIds = [1, 2, 3]
        prot.processedIds = [3]
        prot.framesRange = (1, 1, 1)
        prot.isStreamClosed = False
        prot._getInputSize = lambda: 3
        prot._loadMoviesByIds = lambda movieIds: {movieId: Movie(movieId) for movieId in movieIds}
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._updateDosePlots = lambda *args, **kwargs: None
        prot._store = lambda: None
        prot._loadOutputSet = lambda setClass, baseName: prot.outputMovies if baseName == 'movies.sqlite' else prot.outputMoviesDiscarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute'):
            prot._checkNewOutput()

        self.assertEqual(prot.medianDoseTemporal, [1.0, 3.0, 5.0])
        self.assertEqual(prot.medianDifferenceIds, [1, 2, 3])
        self.assertEqual(prot.mu, 3.0)

    def testClosedStreamUsesAvailableDoseSamples(self):
        class InputSet:
            def getSize(self):
                return 3

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, n_samples=5)
        prot.meanDoseList = [1.0, 1.1, 0.9]
        prot.processedIds = [1, 2]
        prot.isStreamClosed = True
        prot.movsFn = 'movies.sqlite'
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._loadInputSet = lambda _: InputSet()

        self.assertFalse(prot._hasEnoughDoseSamples())

        prot.processedIds.append(3)

        self.assertTrue(prot._hasEnoughDoseSamples())

    def testInputWalChangeTriggersNewInputCheck(self):
        import tempfile

        class InputSet:
            def getIdSet(self):
                return {1}

            def isStreamClosed(self):
                return False

            def close(self):
                pass

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        loadCalls = []
        prot.insertedIds = []
        prot._getFirstJoinStep = lambda: None
        prot._insertNewMoviesSteps = lambda newIds: []
        prot.updateSteps = lambda: None
        prot.isContinued = lambda: False

        with tempfile.TemporaryDirectory() as tmpDir:
            prot.movsFn = os.path.join(tmpDir, 'movies.sqlite')
            with open(prot.movsFn, 'wb') as fh:
                fh.write(b'sqlite')

            prot._loadInputSet = lambda _: loadCalls.append(1) or InputSet()
            prot._checkNewInput()
            prot.insertedIds = [1]
            prot._checkNewInput()
            self.assertEqual(len(loadCalls), 1)

            with open(prot.movsFn + '-wal', 'wb') as fh:
                fh.write(b'wal')

            prot._checkNewInput()
            self.assertEqual(len(loadCalls), 2)

    def testParallelCompletionKeepsAcquisitionOrder(self):
        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.insertedIds = [1, 2, 3, 4]
        prot.processedIds = [3, 4]
        prot.meanDoseById = {3: 1.3, 4: 1.4}

        self.assertEqual(prot._getNewDoneIds([]), [])

        prot.processedIds.extend([1, 2])
        prot.meanDoseById.update({1: 1.1, 2: 1.2})
        prot._syncMeanDoseList()

        self.assertEqual(prot._getNewDoneIds([]), [1, 2, 3, 4])
        self.assertEqual(prot.meanDoseList, [1.1, 1.2, 1.3, 1.4])

    def testOutOfOrderCompletionDoesNotTriggerInitialDoseSampling(self):
        prot = self.newProtocol(XmippProtMovieDoseAnalysis, n_samples=2)
        prot.insertedIds = [1, 2, 3, 4]
        prot.processedIds = [3, 4]
        prot.meanDoseById = {3: 1.3, 4: 1.4}
        prot.isStreamClosed = False
        prot._getAllDoneIds = lambda: ([], 0, [], [])

        self.assertFalse(prot._hasEnoughDoseSamples())

        prot.processedIds.extend([1, 2])
        prot.meanDoseById.update({1: 1.1, 2: 1.2})

        self.assertTrue(prot._hasEnoughDoseSamples())

    def testInitialMedianUsesOnlyConfiguredOrderedSamples(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

            def setFramesRange(self, framesRange):
                pass

        class OutputSet:
            def __init__(self):
                self.ids = []

            def append(self, movie):
                self.ids.append(movie.getObjId())

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, n_samples=2)
        prot.usingExperimental = True
        prot.meanDoseById = {1: 1.0, 2: 1.0, 3: 100.0, 4: 100.0}
        prot.stats = {movieId: {'mean': mean, 'std': 0.0, 'min': mean, 'max': mean} for movieId, mean in prot.meanDoseById.items()}
        prot.insertedIds = [1, 2, 3, 4]
        prot.processedIds = [3, 4, 1, 2]
        prot.medianDifferences = []
        prot.medianDifferenceIds = []
        prot.medianDoseTemporal = []
        prot.framesRange = (1, 1, 1)
        prot.isStreamClosed = False
        prot._doneIds = set()
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getInputSize = lambda: 4
        prot._loadMoviesByIds = lambda movieIds: {movieId: Movie(movieId) for movieId in movieIds}
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._registerDoneIds = lambda movieIds, accepted: prot._doneIds.update(movieIds)
        prot._updateDosePlots = lambda *args, **kwargs: None
        prot._store = lambda: None

        accepted = OutputSet()
        discarded = OutputSet()
        prot._loadOutputSet = lambda setClass, baseName: accepted if baseName == 'movies.sqlite' else discarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute'):
            prot._checkNewOutput()

        self.assertEqual(prot.mu, 1.0)
        self.assertEqual(accepted.ids, [1, 2])
        self.assertEqual(discarded.ids, [3, 4])

    def testDoneIdsAreCached(self):
        class OutputSet:
            def __init__(self, ids):
                self.ids = set(ids)
                self.scans = 0

            def getIdSet(self):
                self.scans += 1
                return set(self.ids)

            def getSize(self):
                return len(self.ids)

        accepted = OutputSet([1, 3])
        discarded = OutputSet([2])
        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.outputMovies = accepted
        prot.outputMoviesDiscarded = discarded

        doneIds, sizeOutput, acceptedIds, discardedIds = prot._getAllDoneIds()
        prot._getAllDoneIds()

        self.assertEqual(set(doneIds), {1, 2, 3})
        self.assertEqual(sizeOutput, 3)
        self.assertEqual(set(acceptedIds), {1, 3})
        self.assertEqual(set(discardedIds), {2})
        self.assertEqual(accepted.scans, 1)
        self.assertEqual(discarded.scans, 1)

        prot._registerDoneIds([4], accepted=True)
        doneIds, sizeOutput, acceptedIds, _ = prot._getAllDoneIds()

        self.assertEqual(set(doneIds), {1, 2, 3, 4})
        self.assertEqual(sizeOutput, 4)
        self.assertEqual(set(acceptedIds), {1, 3, 4})
        self.assertEqual(accepted.scans, 1)
        self.assertEqual(discarded.scans, 1)

    def testInputSizeIsCachedFromInputScan(self):
        class InputSet:
            def getIdSet(self):
                return {1, 2, 3}

            def isStreamClosed(self):
                return False

            def close(self):
                pass

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.movsFn = 'movies.sqlite'
        prot.insertedIds = []
        prot._getInputSetSignature = lambda _: ('signature', None)
        prot._loadInputSet = lambda _: InputSet()
        prot._getFirstJoinStep = lambda: None
        prot._insertNewMoviesSteps = lambda newIds: []
        prot.updateSteps = lambda: None
        prot.isContinued = lambda: False

        prot._checkNewInput()
        prot._loadInputSet = lambda _: self.fail('Input set should not be reopened for its size')

        self.assertEqual(prot._getInputSize(), 3)

    def testDoseAnalysisValidatesPositiveSampleAndWindowSizes(self):
        invalid = self.newProtocol(XmippProtMovieDoseAnalysis, n_samples=0, window=0)
        errors = invalid._validate()

        self.assertIn('Samples to estimate the median dose must be greater than zero.', errors)
        self.assertIn('Window step must be greater than zero.', errors)

        valid = self.newProtocol(XmippProtMovieDoseAnalysis, n_samples=1, window=1)
        self.assertEqual(valid._validate(), [])

    def testDosePlotsDoNotLeakFigures(self):
        import tempfile
        import matplotlib.pyplot as plt
        from xmipp3.protocols.protocol_movie_dose_analysis import plotDoseAnalysis, plotDoseAnalysisDiff

        plt.close('all')

        with tempfile.TemporaryDirectory() as tmpDir:
            plotDoseAnalysis(os.path.join(tmpDir, 'dose.png'), [1.0, 1.1, 0.9], 1.0, 0.95, 1.05)
            plotDoseAnalysisDiff(os.path.join(tmpDir, 'diff.png'), [0.0, 1.0, -1.0])

        self.assertEqual(plt.get_fignums(), [])

    def testDosePlotsAreThrottled(self):
        from unittest.mock import patch

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.meanDoseList = [1.0] * 108
        prot.medianDifferences = [0.0] * 108
        prot.mu = 1.0
        prot.finished = False
        prot.getDosePlot = lambda: 'dose.png'
        prot.getDoseDiffPlot = lambda: 'diff.png'

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.plotDoseAnalysis') as dosePlot:
            with patch('xmipp3.protocols.protocol_movie_dose_analysis.plotDoseAnalysisDiff') as diffPlot:
                prot._updateDosePlots(8, 0.95, 1.05)
                self.assertEqual(dosePlot.call_count, 0)
                self.assertEqual(diffPlot.call_count, 0)

                prot._updateDosePlots(50, 0.95, 1.05)
                self.assertEqual(dosePlot.call_count, 1)
                self.assertEqual(diffPlot.call_count, 1)

                prot._updateDosePlots(80, 0.95, 1.05)
                self.assertEqual(dosePlot.call_count, 1)
                self.assertEqual(diffPlot.call_count, 1)

                prot._updateDosePlots(100, 0.95, 1.05)
                self.assertEqual(dosePlot.call_count, 2)
                self.assertEqual(diffPlot.call_count, 2)

                prot.finished = True
                prot._updateDosePlots(108, 0.95, 1.05)
                self.assertEqual(dosePlot.call_count, 3)
                self.assertEqual(diffPlot.call_count, 3)

                prot._updateDosePlots(108, 0.95, 1.05)
                self.assertEqual(dosePlot.call_count, 3)
                self.assertEqual(diffPlot.call_count, 3)

    def testUpdatedMedianRefreshesAcceptanceLimitsWithinBatch(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def clone(self):
                return Movie(self.movieId)

            def getObjId(self):
                return self.movieId

            def setFramesRange(self, framesRange):
                pass

        class InputSet:
            def getItem(self, field, movieId):
                return Movie(movieId)

        class OutputSet:
            def __init__(self):
                self.ids = []

            def append(self, movie):
                self.ids.append(movie.getObjId())

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, window=1, percentage_window=101)
        prot.mu = 1.0
        prot.usingExperimental = True
        prot.meanDoseList = [2.0, 2.0]
        prot.stats = {
            1: {'mean': 2.0, 'std': 0.0, 'min': 2.0, 'max': 2.0},
            2: {'mean': 2.0, 'std': 0.0, 'min': 2.0, 'max': 2.0}
        }
        prot.insertedIds = [1, 2]
        prot.processedIds = [1, 2]
        prot.medianDifferences = []
        prot.medianDoseTemporal = []
        prot.framesRange = (1, 1, 1)
        prot.movsFn = 'movies.sqlite'
        prot.isStreamClosed = False
        prot._doneIds = set()
        prot._hasEnoughDoseSamples = lambda: False
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getInputSize = lambda: 2
        prot._loadInputSet = lambda _: InputSet()
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._registerDoneIds = lambda movieIds, accepted: prot._doneIds.update(movieIds)
        prot._updateDosePlots = lambda *args, **kwargs: None
        prot._store = lambda: None

        accepted = OutputSet()
        discarded = OutputSet()
        prot._loadOutputSet = lambda setClass, baseName: accepted if baseName == 'movies.sqlite' else discarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute'):
            prot._checkNewOutput()

        self.assertEqual(prot.mu, 2.0)
        self.assertEqual(accepted.ids, [2])
        self.assertEqual(discarded.ids, [1])

    def testWindowMedianIgnoresFutureOutOfOrderMovies(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

            def setFramesRange(self, framesRange):
                pass

        class OutputSet:
            def __init__(self):
                self.ids = []

            def append(self, movie):
                self.ids.append(movie.getObjId())

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, window=2, percentage_window=101)
        prot.mu = 1.0
        prot.usingExperimental = True
        prot.meanDoseById = {1: 1.0, 2: 1.0, 4: 100.0, 5: 100.0}
        prot.meanDoseList = []
        prot.stats = {movieId: {'mean': mean, 'std': 0.0, 'min': mean, 'max': mean} for movieId, mean in prot.meanDoseById.items()}
        prot.insertedIds = [1, 2, 3, 4, 5]
        prot.processedIds = [4, 5, 1, 2]
        prot.medianDifferences = []
        prot.medianDifferenceIds = []
        prot.medianDoseTemporal = []
        prot.framesRange = (1, 1, 1)
        prot.isStreamClosed = False
        prot._doneIds = set()
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getInputSize = lambda: 5
        prot._loadMoviesByIds = lambda movieIds: {movieId: Movie(movieId) for movieId in movieIds}
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._registerDoneIds = lambda movieIds, accepted: prot._doneIds.update(movieIds)
        prot._updateDosePlots = lambda *args, **kwargs: None
        prot._store = lambda: None

        accepted = OutputSet()
        discarded = OutputSet()
        prot._loadOutputSet = lambda setClass, baseName: accepted if baseName == 'movies.sqlite' else discarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute'):
            prot._checkNewOutput()

        self.assertEqual(prot.mu, 1.0)
        self.assertEqual(accepted.ids, [1, 2])
        self.assertEqual(discarded.ids, [])

    def testWindowBoundaryDoseIsNotCountedAsFaulty(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

            def setFramesRange(self, framesRange):
                pass

        class OutputSet:
            def __init__(self):
                self.ids = []

            def append(self, movie):
                self.ids.append(movie.getObjId())

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, window=1, percentage_threshold=10, percentage_window=0)
        prot.mu = 100.0
        prot.usingExperimental = False
        prot.meanDoseList = [110.0]
        prot.stats = {1: {'mean': 110.0, 'std': 0.0, 'min': 110.0, 'max': 110.0}}
        prot.insertedIds = [1]
        prot.processedIds = [1]
        prot.medianDifferences = []
        prot.medianDoseTemporal = []
        prot.framesRange = (1, 1, 1)
        prot.isStreamClosed = False
        prot._doneIds = set()
        prot._hasEnoughDoseSamples = lambda: False
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getInputSize = lambda: 1
        prot._loadMoviesByIds = lambda movieIds: {1: Movie(1)}
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._registerDoneIds = lambda movieIds, accepted: prot._doneIds.update(movieIds)
        prot._updateDosePlots = lambda *args, **kwargs: None
        prot._store = lambda: None
        prot._getExtraPath = lambda *args: 'WARNING.TXT'

        accepted = OutputSet()
        discarded = OutputSet()
        prot._loadOutputSet = lambda setClass, baseName: accepted if baseName == 'movies.sqlite' else discarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute'):
            with patch('builtins.open') as openFile:
                prot._checkNewOutput()

        self.assertEqual(accepted.ids, [1])
        self.assertEqual(discarded.ids, [])
        openFile.assert_not_called()

    def testFailedDoseMovieIsPersistedAsDiscarded(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def clone(self):
                return Movie(self.movieId)

            def getObjId(self):
                return self.movieId

            def setFramesRange(self, framesRange):
                pass

        class InputSet:
            def getItem(self, field, movieId):
                return Movie(movieId)

        class OutputSet:
            def __init__(self):
                self.ids = []

            def append(self, movie):
                self.ids.append(movie.getObjId())

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.mu = 1.0
        prot.stats = {1: {'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0}}
        prot.meanDoseList = [1.0]
        prot.insertedIds = [1, 2]
        prot.processedIds = [1, 2]
        prot.medianDifferences = []
        prot.medianDoseTemporal = []
        prot.framesRange = (1, 1, 1)
        prot.movsFn = 'movies.sqlite'
        prot.isStreamClosed = False
        prot._doneIds = set()
        prot._hasEnoughDoseSamples = lambda: False
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getInputSize = lambda: 2
        prot._loadInputSet = lambda _: InputSet()
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._registerDoneIds = lambda movieIds, accepted: prot._doneIds.update(movieIds)
        prot._updateDosePlots = lambda *args, **kwargs: None
        prot._store = lambda: None

        accepted = OutputSet()
        discarded = OutputSet()
        prot._loadOutputSet = lambda setClass, baseName: accepted if baseName == 'movies.sqlite' else discarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute') as setAttr:
            prot._checkNewOutput()

        self.assertEqual(accepted.ids, [1])
        self.assertEqual(discarded.ids, [2])
        self.assertTrue(any(call.args[0].getObjId() == 2 and call.args[1] == '_DOSE_ANALYSIS_FAILED' and call.args[2] is True for call in setAttr.call_args_list))

    def testAllFailedDoseMoviesFinishClosedStream(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def clone(self):
                return Movie(self.movieId)

            def getObjId(self):
                return self.movieId

            def setFramesRange(self, framesRange):
                pass

        class InputSet:
            def getItem(self, field, movieId):
                return Movie(movieId)

        class OutputSet:
            def __init__(self):
                self.ids = []

            def append(self, movie):
                self.ids.append(movie.getObjId())

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.stats = {}
        prot.meanDoseById = {}
        prot.meanDoseList = []
        prot.insertedIds = [1, 2]
        prot.processedIds = [1, 2]
        prot.framesRange = (1, 1, 1)
        prot.movsFn = 'movies.sqlite'
        prot.isStreamClosed = True
        prot._doneIds = set()
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getInputSize = lambda: 2
        prot._loadInputSet = lambda _: InputSet()
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._registerDoneIds = lambda movieIds, accepted: prot._doneIds.update(movieIds)
        prot._getFirstJoinStep = lambda: None
        prot._store = lambda: None

        discarded = OutputSet()
        prot._loadOutputSet = lambda setClass, baseName: discarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute'):
            prot._checkNewOutput()

        self.assertTrue(prot.finished)
        self.assertFalse(hasattr(prot, 'mu'))
        self.assertEqual(discarded.ids, [1, 2])
        self.assertEqual(prot._doneIds, {1, 2})

    def testLoadInputSetReturnsOpenSet(self):
        from unittest.mock import patch

        class InputSet:
            def __init__(self, filename=None):
                self.closed = False

            def loadAllProperties(self):
                pass

            def isStreamClosed(self):
                return False

            def close(self):
                self.closed = True

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.SetOfMovies', InputSet):
            inputSet = prot._loadInputSet('movies.sqlite')

        self.assertFalse(inputSet.closed)
        inputSet.close()

    def testLoadMoviesByIdsClosesInputSetOnce(self):
        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def clone(self):
                return Movie(self.movieId)

            def getObjId(self):
                return self.movieId

        class InputSet:
            def __init__(self):
                self.closeCalls = 0
                self.getCalls = []

            def getItem(self, field, movieId):
                self.getCalls.append(movieId)
                return Movie(movieId)

            def close(self):
                self.closeCalls += 1

        inputSet = InputSet()
        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.movsFn = 'movies.sqlite'
        prot._loadInputSet = lambda _: inputSet

        movies = prot._loadMoviesByIds([3, 1, 2])

        self.assertEqual(inputSet.getCalls, [3, 1, 2])
        self.assertEqual(inputSet.closeCalls, 1)
        self.assertEqual([movies[movieId].getObjId() for movieId in [3, 1, 2]], [3, 1, 2])

    def testNewInputDoesNotUseLinearInsertedIdLookups(self):
        class CountingList(list):
            def __init__(self, values):
                super().__init__(values)
                self.containsCalls = 0

            def __contains__(self, value):
                self.containsCalls += 1
                return super().__contains__(value)

        class InputSet:
            def getIdSet(self):
                return {1, 2, 3, 4}

            def isStreamClosed(self):
                return False

            def close(self):
                pass

        insertedIds = CountingList([1, 2, 3])
        scheduled = []
        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.movsFn = 'movies.sqlite'
        prot.insertedIds = insertedIds
        prot._getInputSetSignature = lambda _: ('signature', None)
        prot._loadInputSet = lambda _: InputSet()
        prot._getFirstJoinStep = lambda: None
        prot._insertNewMoviesSteps = lambda newIds: scheduled.append(list(newIds)) or []
        prot.updateSteps = lambda: None
        prot.isContinued = lambda: False

        prot._checkNewInput()

        self.assertEqual(scheduled, [[4]])
        self.assertEqual(insertedIds.containsCalls, 0)

    def testRuntimeDoseDecisionIsRestoredFromOutputs(self):
        class OutputMovie:
            def __init__(self, movieId, mean, diff, globalMedian, usingExperimental):
                self.movieId = movieId
                self.mean = mean
                self.diff = diff
                self.globalMedian = globalMedian
                self.usingExperimental = usingExperimental

            def getObjId(self):
                return self.movieId

            def getAttributeValue(self, name, defaultValue=None):
                values = {
                    '_MEAN_DOSE_PER_ANGSTROM2': self.mean,
                    '_DIFF_TO_DOSE_PER_ANGSTROM2': self.diff,
                    '_GLOBAL_DOSE_PER_ANGSTROM2': self.globalMedian,
                    '_USING_EXPERIMENTAL_DOSE': self.usingExperimental
                }
                return values.get(name, defaultValue)

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.outputMovies = [OutputMovie(1, 1.1, 1.0, 1.08, True), OutputMovie(3, 1.3, 2.0, 1.25, True)]
        prot.outputMoviesDiscarded = [OutputMovie(2, 1.2, 3.0, 1.15, True)]

        prot._restoreRuntimeStateFromOutputs()

        self.assertEqual(prot.mu, 1.25)
        self.assertTrue(prot.usingExperimental)
        self.assertEqual(prot.medianDifferenceIds, [1, 2, 3])

    def testDoseRuntimeStateIsPersistedAfterMedianUpdate(self):
        from unittest.mock import patch

        class Movie:
            def __init__(self, movieId):
                self.movieId = movieId

            def clone(self):
                return Movie(self.movieId)

            def getObjId(self):
                return self.movieId

            def setFramesRange(self, framesRange):
                pass

        class OutputSet:
            def __init__(self):
                self.ids = []

            def append(self, movie):
                self.ids.append(movie.getObjId())

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, window=1, percentage_window=101)
        prot.mu = 1.0
        prot.usingExperimental = True
        prot.meanDoseList = [2.0]
        prot.stats = {1: {'mean': 2.0, 'std': 0.0, 'min': 2.0, 'max': 2.0}}
        prot.insertedIds = [1]
        prot.processedIds = [1]
        prot.medianDifferences = []
        prot.medianDoseTemporal = []
        prot.framesRange = (1, 1, 1)
        prot.movsFn = 'movies.sqlite'
        prot.isStreamClosed = False
        prot._doneIds = set()
        prot._hasEnoughDoseSamples = lambda: False
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getInputSize = lambda: 1
        prot._loadMoviesByIds = lambda movieIds: {1: Movie(1)}
        prot._updateOutputSet = lambda *args, **kwargs: None
        prot._registerDoneIds = lambda movieIds, accepted: prot._doneIds.update(movieIds)
        prot._updateDosePlots = lambda *args, **kwargs: None
        prot._store = lambda: None

        accepted = OutputSet()
        discarded = OutputSet()
        prot._loadOutputSet = lambda setClass, baseName: accepted if baseName == 'movies.sqlite' else discarded

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.setAttribute') as setAttr:
            prot._checkNewOutput()

        self.assertEqual(prot.mu, 2.0)
        self.assertTrue(any(call.args[1] == '_GLOBAL_DOSE_PER_ANGSTROM2' and call.args[2] == 2.0 for call in setAttr.call_args_list))
        self.assertTrue(any(call.args[1] == '_USING_EXPERIMENTAL_DOSE' and call.args[2] is True for call in setAttr.call_args_list))

    def testDoseDiffPlotUsesConfiguredThreshold(self):
        import tempfile
        from unittest.mock import patch
        from xmipp3.protocols.protocol_movie_dose_analysis import plotDoseAnalysisDiff

        with tempfile.TemporaryDirectory() as tmpDir:
            with patch('xmipp3.protocols.protocol_movie_dose_analysis.plt.axhline') as axhline:
                plotDoseAnalysisDiff(os.path.join(tmpDir, 'diff.png'), [-1.0, 0.0, 1.0], 10.0)

        lineValues = [call.kwargs['y'] for call in axhline.call_args_list]
        self.assertEqual(lineValues, [10.0, 0.0, -10.0])

    def testDosePlotUpdatePassesConfiguredThreshold(self):
        from unittest.mock import patch

        prot = self.newProtocol(XmippProtMovieDoseAnalysis, percentage_threshold=10)
        prot.meanDoseList = [1.0] * 50
        prot.meanDoseById = {movieId: 1.0 for movieId in range(1, 51)}
        prot.medianDifferences = [0.0] * 50
        prot.medianDifferenceIds = list(range(1, 51))
        prot.mu = 1.0
        prot.finished = False
        prot.getDosePlot = lambda: 'dose.png'
        prot.getDoseDiffPlot = lambda: 'diff.png'

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.plotDoseAnalysis') as dosePlot:
            with patch('xmipp3.protocols.protocol_movie_dose_analysis.plotDoseAnalysisDiff') as diffPlot:
                prot._updateDosePlots(50, 0.9, 1.1)

        dosePlot.assert_called_once_with('dose.png', prot.meanDoseList, 1.0, 0.9, 1.1, list(range(1, 51)))
        diffPlot.assert_called_once_with('diff.png', prot.medianDifferences, 10, prot.medianDifferenceIds)

    def testDosePlotUsesActualMovieIds(self):
        import tempfile
        from unittest.mock import patch
        from xmipp3.protocols.protocol_movie_dose_analysis import plotDoseAnalysis

        movieIds = [1, 4, 9]
        with tempfile.TemporaryDirectory() as tmpDir:
            with patch('xmipp3.protocols.protocol_movie_dose_analysis.plt.scatter') as scatter:
                plotDoseAnalysis(os.path.join(tmpDir, 'dose.png'), [1.0, 1.1, 0.9], 1.0, 0.95, 1.05, movieIds)

        self.assertEqual(list(scatter.call_args.args[0]), movieIds)

    def testDoseDiffPlotUsesActualMovieIds(self):
        import tempfile
        from unittest.mock import patch
        from xmipp3.protocols.protocol_movie_dose_analysis import plotDoseAnalysisDiff

        movieIds = [1, 4, 9]
        with tempfile.TemporaryDirectory() as tmpDir:
            with patch('xmipp3.protocols.protocol_movie_dose_analysis.plt.scatter') as scatter:
                plotDoseAnalysisDiff(os.path.join(tmpDir, 'diff.png'), [0.0, 2.0, -3.0], 5.0, movieIds)

        self.assertEqual(list(scatter.call_args.args[0]), movieIds)

    def testDoseAnalysisSamplesActualMiddleFrame(self):
        from unittest.mock import MagicMock, patch

        class Movie:
            def getNumberOfFrames(self):
                return 5

            def getFileName(self):
                return 'movie.mrcs'

            def getObjId(self):
                return 1

        image = MagicMock()
        image.getData.return_value = np.ones((2, 2))
        imageHandler = MagicMock()
        imageHandler.read.return_value = image

        prot = self.newProtocol(XmippProtMovieDoseAnalysis)
        prot.samplingRate = 1.0

        with patch('xmipp3.protocols.protocol_movie_dose_analysis.ImageHandler', return_value=imageHandler):
            stats = prot.estimatePoissonCount(Movie())

        self.assertIsNotNone(stats)
        self.assertEqual([call.args[0] for call in imageHandler.read.call_args_list], ['1@movie.mrcs', '3@movie.mrcs', '5@movie.mrcs'])

