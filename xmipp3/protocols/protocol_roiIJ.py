# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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

import os
from zipfile import ZipFile

from pyworkflow.em import *
from pyworkflow.em.viewers.showj import *

from xmipp_base import getXmippPath


class XmippProtRoiIJ(ProtAnalysis2D):
    """ Tomogram ROI selection in IJ """
    _label = 'imagej roi'

    def __init__(self, **kwargs):
        ProtAnalysis2D.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('inputTomo', PointerParam, label="Input Tomogram",
                      pointerClass='Tomogram',
                      help='Select a Tomogram.')

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        # Launch Boxing GUI
        self._insertFunctionStep('launchIJGUIStep', interactive=True)

    def _createOutput(self, outputDir):
        pass

    # --------------------------- STEPS functions -----------------------------
    def launchIJGUIStep(self):

        imagej_home = getXmippPath(os.path.join('bindings','java'), 'imagej')
        macroPath = os.path.join(imagej_home, "macros", "AutoSave_ROI.ijm")
        macro = r"""path = "%s";
if (File.exists(path + "ROI.zip")){
roiManager("Open", path + "ROI.zip");
}
else{
roiManager("Draw");
}

setTool("polygon");
waitForUser("Draw the desired ROIs\n\nThen click Ok");
wait(50);

while(roiManager("count")==0)
{
waitForUser("Draw the desired ROIs\n\nThen click Ok");
wait(50);
}

roiManager("save", path + "ROI.zip");
run("Quit");""" % (os.path.join(self._getExtraPath(), ''))
        macroFid = open(macroPath, 'w')
        macroFid.write(macro)
        macroFid.close()

        imagej_home = getXmippPath(os.path.join('bindings','java'), 'imagej')
        macroPath = os.path.join(imagej_home, "macros", "AutoSave_ROI.ijm")
        args = "-i %s -macro %s" %(self.inputTomo.get().getFileName(), macroPath)
        app = "xmipp.ij.commons.XmippImageJ"

        runJavaIJapp(None, app, args).wait()

        #self._createOutput(self.getWorkingDir())

    def _summary(self):
        summary = []
        roiPath = os.path.join(self._getExtraPath(),'ROI.zip')
        if not os.path.isfile(roiPath):
            summary.append("Output ROIs not ready yet.")

        else:
            with ZipFile(roiPath, 'r') as zipObj:
                listOfiles = zipObj.namelist()
            summary.append("%s ouput ROIs have been saved in Scipion." % len(listOfiles))
        return summary