# **************************************************************************
# *
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

from pwem import emlib
from pwem.protocols import ProtClassify3D
from pwem.objects import Particle, SetOfParticles

from pyworkflow import BETA
from pyworkflow.object import ObjectWrap
from pyworkflow.protocol.params import (Form, PointerParam, EnumParam, 
                                        IntParam, StringParam, LEVEL_ADVANCED)
import pyworkflow.utils as pwutils

import xmipp3
from xmipp3.convert import setXmippAttribute, getXmippAttribute

import numpy as np
import sklearn.manifold
import umap


class XmippProtHetDimred(ProtClassify3D, xmipp3.XmippProtocol):
    OUTPUT_PARTICLES_NAME = 'Particles'
    
    _label = 'heterogeneity dimred'
    _devStatus = BETA
    _possibleOutputs = {
        OUTPUT_PARTICLES_NAME: SetOfParticles
    }
    DIMRED_METHODS = [
        'isomap',
        'spectral',
        'tsne',
        'mds',
        'umap'
    ]

    def __init__(self, *args, **kwargs):
        ProtClassify3D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form: Form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label='Particles', important=True,
                      pointerClass=SetOfParticles, help='Input particle set')
        form.addParam('method', EnumParam, label='Dimensionality reduction method',
                      choices=self.DIMRED_METHODS, default=0 )
        form.addParam('outputDim', IntParam, label='Output dimensions',
                      default=2)
        form.addParam('neighbors', IntParam, label='Neighbor count',
                      default=12, expertLevel=LEVEL_ADVANCED)
        form.addParam('components', StringParam, label='Components',
                      expertLevel=LEVEL_ADVANCED,
                      help='Components used in evaluation')
        form.addParallelSection(threads=8)
        
    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('dimredStep')

    # --------------------------- STEPS functions -------------------------------
    def dimredStep(self):
        particles = self._getInputParticles()
        components = self._getComponents()
        transformer = self._getTransformer()
        result: SetOfParticles = self._createSetOfParticles()
        
        values = self._getValues(particles)
        if len(components) > 0:
            values = values[:,components]

        projections = transformer.fit_transform(values)

        def updateItem(item: Particle, coord: np.ndarray):
            setXmippAttribute(item, emlib.MDL_DIMRED, ObjectWrap(coord.tolist()))

        result.copyInfo(particles)
        result.copyItems(particles, updateItemCallback=updateItem, itemDataIterator=iter(projections))
        
        self._defineOutputs(**{self.OUTPUT_PARTICLES_NAME: result})

    # --------------------------- UTILS functions -------------------------------
    def _getInputParticles(self) -> SetOfParticles:
        return self.inputParticles.get()

    def _getComponents(self):
        result = []
        
        if self.components.get():
            result = pwutils.getListFromRangeString(self.components.get())
            
        for i in range(len(result)):
            result[i] -= 1
            
        return result

    def _getTransformer(self) -> sklearn.base.TransformerMixin:
        method = self.DIMRED_METHODS[self.method.get()]
        d = self.outputDim.get()
        jobs = self.numberOfThreads.get()
        neighbors = self.neighbors.get()
        
        result = None
        if method == 'isomap':
            result = sklearn.manifold.Isomap(n_components=d, n_neighbors=neighbors, n_jobs=jobs)
        elif method == 'spectral':
            result = sklearn.manifold.SpectralEmbedding(n_components=d, n_neighbors=neighbors, n_jobs=jobs)
        elif method == 'tsne':
            result = sklearn.manifold.TSNE(n_components=d, n_jobs=jobs)
        elif method == 'mds':
            result = sklearn.manifold.MDS(n_components=d, n_jobs=jobs)
        elif method == 'umap':
            result = umap.UMAP(n_components=d, n_neighbors=neighbors, n_jobs=jobs)
            
        return result
        
    def _getValues(self, particles: SetOfParticles) -> np.ndarray:
        result = []
        
        for particle in particles.iterItems():
            row = np.array(getXmippAttribute(particle, emlib.MDL_DIMRED), dtype=float)
            result.append(row)
            
        return np.row_stack(result)
    