# **************************************************************************
# *
# * Authors:    Amaya Jimenez (ajimenez@cnb.csic.es)
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

from pyworkflow.tests import *
from pyworkflow.protocol import getProtocolFromDb
from pyworkflow.protocol.constants import STATUS_FINISHED
from pwem.protocols import ProtCreateStreamData, ProtImportMovies
from pwem.protocols.protocol_create_stream_data import SET_OF_MOVIES

from xmipp3.protocols import XmippProtMovieResize


RESIZE_SAMPLINGRATE = 0
RESIZE_DIMENSIONS = 1
RESIZE_FACTOR = 2

NUM_MOVIES = 5

# Some utility functions to import movies that are used in several tests.
class TestMovieResize(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('movies')
        cls.dataFile = cls.dataset.getFile('ribo/')
        cls.movie1 = cls.dataset.getFile('qbeta/qbeta.mrc')
        cls.movie2 = cls.dataset.getFile('cct/cct_1.em')

    def runImportMoviesRibo(self):
        args = {'filesPath': self.dataFile,
                'filesPattern': '*.mrcs',
                'amplitudConstrast': 0.1,
                'sphericalAberration': 2.,
                'voltage': 300,
                'samplingRate': 3.54
                }
        protMovieImport = self.newProtocol(ProtImportMovies, **args)
        self.proj.launchProtocol(protMovieImport, wait=False)
        return protMovieImport

    def importMoviesStr(self, fnMovies):
        kwargs = {'inputMovies': fnMovies,
                  'nDim': NUM_MOVIES,
                  'creationInterval': 30,
                  'delay': 10,
                  'setof': SET_OF_MOVIES  # SetOfMovies
                  }
        protStream = self.newProtocol(ProtCreateStreamData, **kwargs)
        protStream.setObjLabel('create Stream Movies')
        self.proj.launchProtocol(protStream, wait=False)

        return protStream

    def runMovieResize(self, fnMovies, option, paramVal):
        kwargs = {'inputMovies': fnMovies,
                  'resizeOption': option
                  }
        if option==RESIZE_SAMPLINGRATE:
            kwargs['resizeSamplingRate']=paramVal
        if option==RESIZE_DIMENSIONS:
            kwargs['resizeDim']=paramVal
        if option==RESIZE_FACTOR:
            kwargs['resizeFactor']=paramVal

        protResize = self.newProtocol(XmippProtMovieResize, **kwargs)
        self.proj.launchProtocol(protResize, wait=False)

        return protResize

    def _updateProtocol(self, prot):
        prot2 = getProtocolFromDb(prot.getProject().path,
                                  prot.getDbPath(),
                                  prot.getObjId())
        # Close DB connections
        prot2.getProject().closeMapper()
        prot2.closeMappers()
        return prot2


    def test_pattern(self):

        protImport = self.runImportMoviesRibo()
        self._waitOutput(protImport, 'outputMovies')

        protImportMovsStr = self.importMoviesStr(protImport.outputMovies)
        self._waitOutput(protImportMovsStr, 'outputMovies')
        if protImportMovsStr.isFailed():
            self.assertTrue(False)

        newSamplingRate=4.0
        protMovieResize = self.runMovieResize(protImportMovsStr.outputMovies,
                                              RESIZE_SAMPLINGRATE,
                                              newSamplingRate)
        self._waitOutput(protMovieResize, 'outputMovies')

        newDimensions = 1700
        protMovieResize2 = self.runMovieResize(protMovieResize.outputMovies,
                                              RESIZE_DIMENSIONS,
                                              newDimensions)
        self._waitOutput(protMovieResize2, 'outputMovies')

        newFactor = 2.15
        protMovieResize3 = self.runMovieResize(protMovieResize2.outputMovies,
                                              RESIZE_FACTOR, newFactor)
        self._waitOutput(protMovieResize3, 'outputMovies')

        if not protMovieResize3.hasAttribute('outputMovies'):
            self.assertTrue(False)

        while protMovieResize3.getStatus() != STATUS_FINISHED:
            protMovieResize3 = self._updateProtocol(protMovieResize3)
            if protMovieResize3.isFailed():
                self.assertTrue(False)

        protImportMovsStr = self._updateProtocol(protImportMovsStr)
        protMovieResize = self._updateProtocol(protMovieResize)
        protMovieResize2 = self._updateProtocol(protMovieResize2)
        protMovieResize3 = self._updateProtocol(protMovieResize3)
        if protMovieResize3.outputMovies.getSize() != \
                protMovieResize3.outputMovies.getSize():
            self.assertTrue(False)

        xOrig, yOrig, zOrig = protImportMovsStr.outputMovies.getDim()
        samplingRateOrig = protImportMovsStr.outputMovies.getSamplingRate()
        x1, y1, z1 = protMovieResize.outputMovies.getDim()
        factor1 = newSamplingRate / samplingRateOrig
        if x1 != int(xOrig/factor1) or y1 != int(yOrig/factor1) \
                or z1 != zOrig:
            self.assertTrue(False)

        x2, y2, z2 = protMovieResize2.outputMovies.getDim()
        factor2 = float(x1) / float(newDimensions)
        if x2 != int(x1 / factor2) or y2 != int(y1 / factor2) \
                or z2 != z1:
            self.assertTrue(False)

        x3, y3, z3 = protMovieResize3.outputMovies.getDim()
        if x3 != int(x2 / newFactor) or y3 != int(y2 / newFactor) \
                or z3 != z2:
            self.assertTrue(False)

