# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
# *  BCU, Centro Nacional de Biotecnologia, CSIC
# *
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
from pwem.objects.data import Volume
from pwem.protocols import EMProtocol
from pyworkflow.protocol.params import FloatParam, TextParam


class XmippProtPhantom(EMProtocol):
    """ Creates a phantom volume based on a feature description file by using the xmipp_phantom_create tool. This synthetic volume can be used to test and validate algorithms or processing pipelines under controlled conditions. """

    _label = 'phantom volume'

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('desc', TextParam, label='Create phantom',
                      default='40 40 40 0\ncyl + 1 0 0 0 15 15 2 0 0 0\nsph + 1 0 0 5 2\ncyl + 1 0 0 -5 2 2 10 0 90 0\n'
                              'sph + 1 0 -5 5 2',
                      help="create a phantom description: x y z backgroundValue geometry(cyl, sph...) +(superimpose) "
                           "desnsityValue origin radius height rot tilt psi. See more information in "
                           "https://web.archive.org/web/20180813105422/http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/FileFormats#Phantom_metadata_file")
        form.addParam('sampling', FloatParam, label='Sampling rate', default=4)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('createPhantomsStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def createPhantomsStep(self):
        desc = self.desc.get()
        fnDescr = self._getExtraPath("phantom.descr")
        fhDescr = open(fnDescr, 'w')
        fhDescr.write(desc)
        fhDescr.close()
        self.runJob("xmipp_phantom_create", " -i %s -o %s" % (fnDescr, self._getExtraPath("phantom.vol")))

    def createOutputStep(self):
        outputVol = Volume()
        outputVol.setLocation(self._getExtraPath("phantom.vol"))
        outputVol.setSamplingRate(self.sampling.get())
        self._defineOutputs(outputVolume=outputVol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output phantom not ready yet.")
        else:
            summary.append("Phantoms created")
        return summary
