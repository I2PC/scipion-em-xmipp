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