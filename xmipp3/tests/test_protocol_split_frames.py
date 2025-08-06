# **************************************************************************
# *
# * Authors:    Eduardo García Delgado (eduardo.garcia@cnb.csic.es)
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

from pwem.protocols import ProtImportMovies, exists
from pyworkflow.tests import BaseTest, DataSet, setupTestProject

from xmipp3.protocols import XmippProtSplitFrames


class TestSplitFramesMoviesBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='movies'):
        cls.dataset = DataSet.getDataSet(dataProject)

    @classmethod
    def runImportMovies(cls, pattern, samplingRate, voltage, scannedPixelSize,
                       magnification, sphericalAberration, dosePerFrame=None):
        """ Run an Import micrograph protocol. """

        kwargs = {
            'filesPath': pattern,
            'magnification': magnification,
            'voltage': voltage,
            'sphericalAberration': sphericalAberration,
            'dosePerFrame': dosePerFrame
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
        cls.launchProtocol(cls.protImport, wait=True)

        if cls.protImport.isFailed():
            raise Exception("Protocol has failed. Error: ",
                            cls.protImport.getErrorMessage())

        # Check that input movies have been imported (a better way to do this?)
        if cls.protImport.outputMovies is None:
            raise Exception('Import of movies: %s, failed, '
                            'outputMovies is None.' % pattern)

        return cls.protImport

class TestSplitFramesMovies(TestSplitFramesMoviesBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestSplitFramesMoviesBase.setData()
        cls.protImportMov = cls.runImportMovies(cls.dataset.getFile('qbeta/qbeta.mrc'),1.14,300,
                                                None,50000,2.7)

    def testSplitFrames1(self):
        SplitFrames = self.newProtocol(XmippProtSplitFrames,
                                   objLabel='split frames',
                                   inputMovies=self.protImportMov.outputMovies,
                                   set=0)
        self.launchProtocol(SplitFrames)

        oddMovie = getattr(SplitFrames, 'oddMovie', None)
        evenMovie = getattr(SplitFrames, 'evenMovie', None)

        self.assertIsNotNone(oddMovie, "OddMovie didn't generate correctly")
        self.assertIsNotNone(evenMovie, "EvenMovie didn't generate correctly")

        oddSR = oddMovie.getSamplingRate()
        evenSR = evenMovie.getSamplingRate()

        self.assertAlmostEqual(oddSR, 1.14)
        self.assertAlmostEqual(evenSR, 1.14)

        oddImage = oddMovie.getFirstItem().getFileName()
        evenImage = evenMovie.getFirstItem().getFileName()

        self.assertTrue(exists(oddImage), "OddImage has failed")
        self.assertTrue(exists(evenImage), "EvenImage has failed")

        oddFrames = oddMovie.getFirstItem().getNumberOfFrames()
        evenFrames = evenMovie.getFirstItem().getNumberOfFrames()

        self.assertEqual(oddFrames, 4, "That's not OddMovie's length")
        self.assertEqual(evenFrames, 3, "That's not EvenMovie's length")

    def testSplitFrames2(self):
        SplitFrames2 = self.newProtocol(XmippProtSplitFrames,
                                       objLabel='split frames',
                                       inputMovies=self.protImportMov.outputMovies,
                                       set=1)
        self.launchProtocol(SplitFrames2)

        oddMicrograph = getattr(SplitFrames2, 'oddMicrograph', None)
        evenMicrograph = getattr(SplitFrames2, 'evenMicrograph', None)

        self.assertIsNotNone(oddMicrograph, "OddMicrograph didn't generate correctly")
        self.assertIsNotNone(evenMicrograph, "EvenMicrograph didn't generate correctly")

        oddSR = oddMicrograph.getSamplingRate()
        evenSR = evenMicrograph.getSamplingRate()

        self.assertAlmostEqual(oddSR, 1.14)
        self.assertAlmostEqual(evenSR, 1.14)

        oddImage = oddMicrograph.getFirstItem().getFileName()
        evenImage = evenMicrograph.getFirstItem().getFileName()

        self.assertTrue(exists(oddImage), "OddImage has failed")
        self.assertTrue(exists(evenImage), "EvenImage has failed")
