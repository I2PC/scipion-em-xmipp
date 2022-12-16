# **************************************************************************
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *              Daniel del Hoyo Gomez (daniel.delhoyo.gomez@alumnos.upm.es)
# *              Jorge Garcia Condado (jgcondado@cnb.csic.es)
# *              Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

from unittest import result
from pwem import emlib
from pwem.protocols import EMProtocol
from pwem.objects import Volume

from pyworkflow.protocol.params import MultiPointerParam, EnumParam, IntParam
from pyworkflow.protocol.params import Range, GE, GT, LE, LT
from pyworkflow.object import Float
from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import STEPS_PARALLEL
from pyworkflow.utils.path import makePath

from xmipp3.convert import setXmippAttribute

import math
import csv
import collections
import itertools
import os.path
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

class XmippProtConsensusClasses3D(EMProtocol):
    """ Compare several SetOfClasses3D.
        Return the consensus clustering based on a objective function
        that uses the similarity between clusters intersections and
        the entropy of the clustering formed.
    """
    _label = 'consensus clustering 3D'
    _devStatus = BETA

    def __init__(self, *args, **kwargs):
        EMProtocol.__init__(self, *args, **kwargs)
        
    def _defineParams(self, form):
        # Input data
        form.addSection(label='Input')
        form.addParam('inputClassifications', MultiPointerParam, pointerClass='SetOfClasses3D', label="Input classes", 
                      important=True,
                      help='Select several sets of classes where to evaluate the '
                           'intersections.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        pass

    def createOutputStep(self):
        pass
    
    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        pass
