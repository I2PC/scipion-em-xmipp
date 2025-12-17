# **************************************************************************
# *
# * Authors:     Alberto Garcia Mena (alberto.garcia@cnb.csic.es)
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
from xmipp3.base import XmippProtocol
from pyworkflow import BETA, UPDATED, NEW, PROD
import pyworkflow.protocol.params as params
from pyworkflow.protocol import getUpdatedProtocol

SCREENING_PROTOCOL = ['smartscopeConnection']
FRAME_PROTOCOL = ['ProtImportMovies']
MICROGRAPH_PROTOCOL = ['ProtMotionCorr', 'XmippProtFlexAlign' ]
RESIZE_SIZE = 3680
DOSE = 1

class XmippProtPushSPALake(XmippProtocol):
    """
    Protocol to push and populate the SPA_lake dataBase with data from several protocols of the SPA workflow.

    Project: Title, Date, Sample, Molecular weight, microscope, detector
    Screening: Grid material, meshSize, meshMaterial, holeType, holeSpacing, mesh:squareSize, mes_basSize, mesh_pitch
    Square image
    Hole image
    Frame image
    Micrograph image (1024x1024, uint8, normalized, padding and compressed tif
    Coordenates (x,y)
    2DClasses reference and images
    """

    _devStatus = BETA
    _label = 'pushSPALake'

    def __init__(self, **args):
        XmippProtPushSPALake.__init__(self, **args)

    #--------------------------- DEFINE param functions ------------------------

    def _defineAlignmentParams(self, form):
        form.addParam('inputProtocols', params.MultiPointerParam,
	                  label="Input protocols", important=True,
	                  pointerClass='EMProtocol',
	                  help="this protocol/s will be used to push data to the SPA-lake database")



    def initialize(self):
        self.prots = [getUpdatedProtocol(p) for p in self.getInputProtocols()]


    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):

        #DYNAMIC TEMPLATE STARTS
        import os
        fname = "/home/agarcia/Documents/test_DEBUGALBERTO.txt"
        if os.path.exists(fname):
            os.remove(fname)
        fjj = open(fname, "a+")
        fjj.write('ALBERTO--------->onDebugMode PID {}'.format(os.getpid()))
        fjj.close()
        print('ALBERTO--------->onDebugMode PID {}'.format(os.getpid()))
        import time
        time.sleep(15)
        #DYNAMIC TEMPLATE ENDS

        self.initialize()
        #self.prepareScreening()
        self.prepareFrames()
        self.prepareMicrograph()


    # --------------------------- prepare functions --------------------------------------------
    def prepareScreening(self):
        pass



    def prepareFrames(self, movie):
        '''
        Take the movie
        1A- If mrc or tif, extract the central frame; adjust the frames to the dose desire:
            Assign the frames to have a dose/frame in [1 - 1.5] e-/A2
            dose 1e-/A2: 100 frames dose/frame = 0.5 => 2 frames
            dose 1e-/A2: 50 frames dose/frame = 0.75 => 2 frames
            dose 1e-/A2: 25 frames dose/frame = 1.5 => 1 frame
        1B- if eer, get a frame based on the dose choosen
        2- Crop to square getting the upper left corner as zero
        3- Resize to 3680x3680 usando box/area method sinc(x), indirectly it applies a low pass filter
        4- Change the pixel deep to uint8 (8bits is more than enought)
        5- Normalze to 0-255 (10->255) or maive 5
        5- Apply jpeg compression and save it as jpeg

        Returns
        -> jpeg frame image
        ------

        '''
        print(self.prots)



    def prepareMicrograph(self):
        pass

    # ------------------------------------------ Utils ---------------------------------------------------
    def getInputProtocols(self):
        protocols = []
        for protPointer in self.inputProtocols:
            prot = protPointer.get()
            prot.setProject(self.getProject())
            protocols.append(prot)
        return protocols

