# **************************************************************************
# *
# * Authors:  Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pyworkflow.protocol.params as params
import pwem.emlib.metadata as md
import numpy as np
import math
import matplotlib.pyplot as plt
from pyworkflow.utils.path import cleanPath
from pwem.objects import SetOfParticles
from xmipp3.protocols.protocol_angular_alignment_sph import XmippProtAngularAlignmentSPH


class XmippAngularAlignmentSphViewer(ProtocolViewer):
    """ Visualize the output of protocol volume strain """
    _label = 'viewer angular align sph'
    _targets = [XmippProtAngularAlignmentSPH]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data


    def _defineParams(self, form):
        form.addSection(label='Show info for angular alignment')
        form.addParam('doShowHist1D', params.LabelParam,
                      label="Display the coefficients histogram 1D")
        form.addParam('doShowHist2D', params.LabelParam,
                      label="Display the coefficients histogram 2D")
        # form.addParam('displayClustering', params.LabelParam,
        #               label='Open clustering tool?',
        #               help='Open a GUI to visualize the images as points'
        #                    'and select some of them to create new clusters.')

    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        return {'doShowHist1D': self._doShowHist1D,
                'doShowHist2D': self._doShowHist2D
                # 'displayClustering': self._displayClustering
                }

    def _doShowHist1D(self, param=None):
        tsne1D = self.loadData1D()
        plt.hist(tsne1D, bins=50)
        plt.title("SPH coefficients histogram in 1D")
        plt.show()

    def _doShowHist2D(self, param=None):
        tsne2D = self.loadData2D()
        # H, xedges, yedges = np.histogram2d(tsne2D[:,0], tsne2D[:,1], bins=(50,50))
        # H = H.T
        # fig = plt.figure()
        # ax = fig.add_subplot(111, title='SPH coefficients histogram in 2D',
        #                      aspect = 'equal', xlim = xedges[[0, -1]],
        #                      ylim = yedges[[0, -1]])
        # im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
        # xcenters = (xedges[:-1] + xedges[1:]) / 2
        # ycenters = (yedges[:-1] + yedges[1:]) / 2
        # im.set_data(xcenters, ycenters, H)
        # ax.images.append(im)
        # plt.show()

        #AJ trying something new...
        x = tsne2D[:,0]
        y = tsne2D[:,1]

        rangeX = np.max(x)-np.min(x)
        rangeY = np.max(y)-np.min(y)
        if rangeX>rangeY:
            sigma = rangeX/50
        else:
            sigma = rangeY/50
        print("sigma",sigma)

        # define grid.
        xi = np.linspace(min(x)-0.1, max(x)+0.1, 100)
        yi = np.linspace(min(x)-0.1, max(x)+0.1, 100)

        print(x.shape)
        print(y.shape)
        print(xi.shape)
        print(yi.shape)

        z = np.zeros((100, 100), float)
        zSize = z.shape
        N = len(x)
        for c in range(zSize[1]):
            for r in range(zSize[0]):
                for d in range(N):
                    z[r,c] = z[r,c] + (1.0/N) * (1.0/((2*math.pi)*sigma**2)) * math.exp(-((xi[c]-x[d])**2 + (yi[r]-y[d])**2)/(2*sigma**2))

        # grid the data
        #zi = griddata((x, y), z, (xi, yi), method='cubic')
        zMax = np.max(z)
        z = z/zMax

        # contour the gridded data, plotting dots at the randomly spaced data points.
        CS = plt.contour(xi, yi, z, 15, linewidths=0.5, colors='k')
        CS = plt.contourf(xi, yi, z, 15, cmap=plt.cm.jet)
        plt.colorbar()  # draw colorbar
        # plot data points.
        # plt.scatter(x, y, marker='o', c='b', s=0.5)
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        plt.title('griddata test')
        plt.show()


    # def _displayClustering(self, paramName):
    #     self.clusterWindow = self.tkWindow(ClusteringWindow,
    #                                        title='Clustering Tool',
    #                                        dim=2,
    #                                        data=self.getData(),
    #                                        callback=self._createCluster
    #                                        )
    #     return [self.clusterWindow]

    # def _createCluster(self):
    #     """ Create the cluster with the selected particles
    #     from the cluster. This method will be called when
    #     the button 'Create Cluster' is pressed.
    #     """
    #     # Write the particles
    #     prot = self.protocol
    #     project = prot.getProject()
    #     inputSet = prot.getInputParticles()
    #     fnSqlite = prot._getExtraPath('cluster_particles.sqlite')
    #     cleanPath(fnSqlite)
    #     partSet = SetOfParticles(filename=fnSqlite)
    #     partSet.copyInfo(inputSet)
    #     for point in self.getData():
    #         if point.getState() == Point.SELECTED:
    #             particle = inputSet[point.getId()]
    #             partSet.append(particle)
    #     partSet.write()
    #     partSet.close()
    #
    #     from xmipp3.protocols.nma.protocol_batch_cluster import BatchProtNMACluster
    #     newProt = project.newProtocol(BatchProtNMACluster)
    #     clusterName = self.clusterWindow.getClusterName()
    #     if clusterName:
    #         newProt.setObjLabel(clusterName)
    #     newProt.inputNmaDimred.set(prot)
    #     newProt.sqliteFile.set(fnSqlite)
    #
    #     project.launchProtocol(newProt)

    def loadData1D(self):
        fnOut = self.protocol._getFileName('fnOut')
        mdOut = md.MetaData(fnOut)
        i=0
        for row in md.iterRows(mdOut):
            coeff1D = mdOut.getValue(md.MDL_SPH_TSNE_COEFF1D, row.getObjId())
            if i==0:
                tsne1D = coeff1D
            else:
                tsne1D = np.vstack((tsne1D, coeff1D))
            i+=1
        return tsne1D

    def loadData2D(self):
        fnOut = self.protocol._getFileName('fnOut')
        mdOut = md.MetaData(fnOut)
        i=0
        for row in md.iterRows(mdOut):
            coeff2D = mdOut.getValue(md.MDL_SPH_TSNE_COEFF2D, row.getObjId())
            if i==0:
                tsne2D = coeff2D
            else:
                tsne2D = np.vstack((tsne2D, coeff2D))
            i+=1
        return tsne2D

    # def loadData(self):
    #     """ Iterate over the images and the output matrix txt file
    #     and create a Data object with theirs Points.
    #     """
    #     matrix = self.loadData2D()
    #     particles = self.protocol.getInputParticles()
    #     data = Data()
    #     for i, particle in enumerate(particles):
    #         data.addPoint(Point(pointId=particle.getObjId(),
    #                             data=matrix[i, :],
    #                             weight=1))
    #     return data

