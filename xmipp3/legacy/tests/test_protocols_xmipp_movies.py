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


class TestOFAlignment(TestXmippBase):
    """This class check if the preprocessing micrographs protocol
    in Xmipp works properly."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport1 = cls.runImportMovie1(cls.movie1)
        cls.protImport2 = cls.runImportMovie2(cls.movie2)
    
    def runOFProtocol(self, movies, label="Default", saveMic=True,
                      saveMovie=False, useAlign=False):
        protOF = XmippProtOFAlignment(doSaveAveMic=saveMic,
                                      doSaveMovie=saveMovie,
                                      useAlignment=useAlign)
        protOF.setObjLabel(label)
        protOF.inputMovies.set(movies)
        self.launchProtocol(protOF)
        return protOF
    
    def testAlignOF1(self):
        protOF1 = self.runOFProtocol(self.protImport1.outputMovies,
                                     label="Movie MRC")
        self.assertIsNotNone(protOF1.outputMicrographs,
                             "SetOfMicrographs has not been created.")
    
    def testAlignOF2(self):
        protOF2 = self.runOFProtocol(self.protImport2.outputMovies,
                                     label="Movie EM")
        self.assertIsNotNone(protOF2.outputMicrographs,
                             "SetOfMicrographs has not been created.")
    
    def testAlignOFSaveMovieAndMic(self):
        protOF3 = self.runOFProtocol(self.protImport1.outputMovies,
                                     label="Save Movie", saveMovie=True)
        self.assertIsNotNone(protOF3.outputMovies,
                             "SetOfMovies has not been created.")
    
    def testAlignOFSaveMovieNoMic(self):
        protOF4 = self.runOFProtocol(self.protImport1.outputMovies,
                                     label="Save Movie", saveMic=False,
                                     saveMovie=True)
        self.assertIsNotNone(protOF4.outputMovies,
                             "SetOfMovies has not been created.")
    
    def testAlignOFWAlignment(self):
        prot = XmippProtFlexAlign(doSaveAveMic=False)
        prot.inputMovies.set(self.protImport1.outputMovies)
        self.launchProtocol(prot)
        
        protOF5 = self.runOFProtocol(prot.outputMovies,
                                     label="Movie w Alignment",
                                     saveMic=False, 
                                     saveMovie=True)
        self.assertIsNotNone(protOF5.outputMovies,
                             "SetOfMovies has not been created.")


class TestOFAlignment2(TestXmippBase):
    """This class check if the optical flow protocol in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsMovies = DataSet.getDataSet('movies')

    def getArgs(self, filesPath, pattern=''):
        return {'importFrom': ProtImportMovies.IMPORT_FROM_FILES,
                'filesPath': self.dsMovies.getFile(filesPath),
                'filesPattern': pattern,
                'amplitudConstrast': 0.1,
                'sphericalAberration': 2.,
                'voltage': 300,
                'samplingRate': 3.54,
                'dosePerFrame' : 2.0,
                }

    def _checkOutput(self, prot, args, moviesId=[], size=None, dim=None):
        movies = getattr(prot, 'outputMovies', None)
        self.assertIsNotNone(movies)
        self.assertEqual(movies.getSize(), size)

        for i, m in enumerate(movies):
            if moviesId:
                self.assertEqual(m.getObjId(), moviesId[i])
            self.assertAlmostEqual(m.getSamplingRate(),
                                   args['samplingRate'])
            a = m.getAcquisition()
            self.assertAlmostEqual(a.getVoltage(), args['voltage'])

            if dim is not None: # Check if dimensions are the expected ones
                x, y, n = m.getDim()
                self.assertEqual(dim, (x, y, n))

    def _importMovies(self):
        args = self.getArgs('ribo/', pattern='*movie.mrcs')

        # Id's should be set increasing from 1 if ### is not in the pattern
        protMovieImport = self.newProtocol(ProtImportMovies, **args)
        protMovieImport.setObjLabel('from files')
        self.launchProtocol(protMovieImport)

        self._checkOutput(protMovieImport, args, [1, 2, 3], size=3,
                          dim=(1950, 1950, 16))
        return protMovieImport

    def test_OpticalFlow(self):
        protMovieImport = self._importMovies()

        mc1 = self.newProtocol(XmippProtFlexAlign,
                               objLabel='CC (no-write)',
                               alignFrame0=2, alignFrameN=10,
                               useAlignToSum=True,
                               numberOfThreads=1)
        mc1.inputMovies.set(protMovieImport.outputMovies)
        self.launchProtocol(mc1)

        of1 = self.newProtocol(XmippProtOFAlignment,
                               objLabel='OF DW',
                               alignFrame0=2, alignFrameN=10,
                               useAlignment=True,
                               doApplyDoseFilter=True,
                               doSaveUnweightedMic=True,
                               numberOfThreads=1)
        of1.inputMovies.set(mc1.outputMovies)
        self.launchProtocol(of1)
        self.assertIsNotNone(of1.outputMicrographs,
                             "SetOfMicrographs has not been created.")
        self.assertIsNotNone(of1.outputMicrographsDoseWeighted,
                             "SetOfMicrographs with dose correction has not "
                             "been created.")


class TestExtractMovieParticles(BaseTest):
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
		cls.protImport1 = cls.runImportMovies(
			cls.ds.getFile('qbeta/qbeta.mrc'),
			magnification=50000)
		cls.protImport2 = cls.runImportMovies(cls.ds.getFile('cct/cct_1.em'),
		                                      magnification=61000)

	# def _checkMicrographs(self, protocol, goldDimensions):
	#     self.assertIsNotNone(getattr(protocol, 'outputMicrographs', None),
	#                          "Output SetOfMicrographs were not created.")
	#     mic = protocol.outputMicrographs[1]
	#     x, y, _ = mic.getDim()
	#     dims = (x, y)
	#     msgError = "The dimensions must be %s and it is %s"
	#     self.assertEqual(goldDimensions, dims,
	#                      msgError % (goldDimensions, dims))
	#
	def _checkAlignment(self, movie, goldRange, goldRoi):
		alignment = movie.getAlignment()
		range = alignment.getRange()
		msgRange = "Alignment range must be %s %s and it is %s (%s)"
		self.assertEqual(goldRange, range,
		                 msgRange % (
			                 goldRange, range, type(goldRange), type(range)))
		roi = alignment.getRoi()
		msgRoi = "Alignment ROI must be %s (%s) and it is %s (%s)"
		self.assertEqual(goldRoi, roi,
		                 msgRoi % (goldRoi, roi, type(goldRoi), type(roi)))

	def test_qbeta(self):
		movAliProt = self.newProtocol(XmippProtFlexAlign,
		                              alignFrame0=2, alignFrameN=6,
		                              doSaveAveMic=True)
		movAliProt.inputMovies.set(self.protImport1.outputMovies)
		self.launchProtocol(movAliProt)

		self._checkAlignment(movAliProt.outputMovies[1],
		                     (2, 6), [0, 0, 0, 0])

		importPick = self.newProtocol(ProtImportCoordinates,
		                              importFrom=ProtImportCoordinates.IMPORT_FROM_XMIPP,
		                              filesPath=self.ds.getFile('qbeta/'),
		                              filesPattern='*.pos', boxSize=320,
		                              invertX=False,
		                              invertY=False
		                              )
		importPick.inputMicrographs.set(movAliProt.outputMicrographs)
		importPick.setObjLabel('import coords from xmipp ')
		self.launchProtocol(importPick)

		protExtract = self.newProtocol(XmippProtExtractMovieParticles,
		                               boxSize=320, frame0=2, frameN=6,
		                               applyAlignment=True, doInvert=True)
		protExtract.inputMovies.set(movAliProt.outputMovies)
		protExtract.inputCoordinates.set(importPick.outputCoordinates)
		protExtract.setObjLabel('extract with alignment')
		self.launchProtocol(protExtract)

		self.assertIsNotNone(getattr(protExtract, 'outputParticles', None),
		                     "Output SetOfMovieParticles were not created.")

		size = protExtract.outputParticles.getSize()
		self.assertEqual(size, 135, 'Number of particles must be 135 and its '
		                            '%d' % size)

	def test_cct(self):
		movAliProt = self.newProtocol(XmippProtFlexAlign,
		                              alignFrame0=2, alignFrameN=6,
		                              doSaveAveMic=True)
		movAliProt.inputMovies.set(self.protImport2.outputMovies)
		self.launchProtocol(movAliProt)

		self._checkAlignment(movAliProt.outputMovies[1],
		                     (2, 6), [0, 0, 0, 0])

		importPick = self.newProtocol(ProtImportCoordinates,
		                              importFrom=ProtImportCoordinates.IMPORT_FROM_XMIPP,
		                              filesPath=self.ds.getFile('cct/'),
		                              filesPattern='*.pos', boxSize=320,
		                              invertX=False,
		                              invertY=False
		                              )
		importPick.inputMicrographs.set(movAliProt.outputMicrographs)
		importPick.setObjLabel('import coords from xmipp ')
		self.launchProtocol(importPick)

		protExtract = self.newProtocol(XmippProtExtractMovieParticles,
		                               boxSize=320, frame0=3, frameN=6,
		                               applyAlignment=False, doInvert=True)
		protExtract.inputMovies.set(movAliProt.outputMovies)
		protExtract.inputCoordinates.set(importPick.outputCoordinates)
		protExtract.setObjLabel('extract without alignment')
		self.launchProtocol(protExtract)

		self.assertIsNotNone(getattr(protExtract, 'outputParticles', None),
		                     "Output SetOfMovieParticles were not created.")

		size = protExtract.outputParticles.getSize()
		self.assertEqual(size, 88, 'Number of particles must be 135 and its '
		                           '%d' % size)


