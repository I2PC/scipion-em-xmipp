# **************************************************************************
# *
# * Authors:     Daniel Marchán Torres (da.marchan@cnb.csic.es)
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

from pyworkflow.viewer import Viewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import LabelParam
from pwem.viewers import EmProtocolViewer, ObjectView, ClassesView
from xmipp3.protocols.protocol_cl2d_clustering import XmippProtCL2DClustering
import matplotlib.pyplot as plt
import os



class XmippCL2DClusteringViewer(EmProtocolViewer):
    """ This viewer is intended to visualize the selection made by the Xmipp - clustering 2d classes protocol.
    """
    _label = 'viewer Clustering 2D Classes'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtCL2DClustering]


    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('visualizeOutput', LabelParam,
                      label="Visualize output",
                      help="Visualize the aggregated 2D classes, 2D averages or both.")
        form.addParam('visualizeCluster', LabelParam,
                      label="Visualize a 2D representation of the clustering",
                      help="This will show a 2D representation of the clustering operation. "
                              "The clustering operation is done on a multidimensional space, meaning that this"
                              "2D representation (using TSNE) might not capture the real difference between some clusters.")
        form.addParam('visualizeClusterImages', LabelParam,
                      label="Visualize the clusters distribution",
                      help="Visualize the clusters images.")

    def _getVisualizeDict(self):
        return {
                 'visualizeOutput': self._visualizeOutputs,
                 'visualizeCluster': self._visualizeCluster,
                 'visualizeClusterImages': self._visualizeClusterImages
                }

    def _visualizeOutputs(self, e=None):
        outputList = []

        for objName in ["outputClasses", "outputAverages"]:
            if self.protocol.hasAttribute(objName):
                outputList.append(objName)

        return self._visualizeMultipleOutputs(outputList)

    def _visualizeCluster(self, e=None):
        if os.path.exists(self.protocol.getClusterPlot()):# Load the image
            image = plt.imread(self.protocol.getClusterPlot())
            # Get the image dimensions (height, width)
            height, width, _ = image.shape
            # Convert pixels to inches for the figure size (assuming 100 DPI)
            dpi = 100
            figsize = (width / dpi, height / dpi)
            # Create the figure with the calculated size
            plt.figure(figsize=figsize)
            # Display the image without axes
            fig = plt.imshow(image)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            # Show the image
            plt.show()

    def _visualizeClusterImages(self, e=None):
        if os.path.exists(self.protocol.getClusterImagesPlot()):
            # Load the image
            image = plt.imread(self.protocol.getClusterImagesPlot())
            # Get the image dimensions (height, width)
            height, width, _ = image.shape
            # Convert pixels to inches for the figure size (assuming 100 DPI)
            dpi = 100
            figsize = (width / dpi, height / dpi)
            # Create the figure with the calculated size
            plt.figure(figsize=figsize)
            # Display the image without axes
            fig = plt.imshow(image)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            # Show the image
            plt.show()

    def _visualizeMultipleOutputs(self, objList):
        views = []
        classView = "outputClasses"

        if objList:
            for objName in objList:
                if self.protocol.hasAttribute(objName):
                    outputSet = getattr(self.protocol, objName)
                    outputId = outputSet.strId()
                    outputFn = outputSet.getFileName()
                    if objName == classView:
                        views.append(ClassesView(self._project, outputId, outputFn))
                    else:
                        views.append(ObjectView(self._project, outputId, outputFn))
        else:
            self.infoMessage('%s does not have output %s'
                             % (self.protocol.getObjLabel(),
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

def getStringIfActive(prot):
    return 'yet.' if prot.isActive() else '.'