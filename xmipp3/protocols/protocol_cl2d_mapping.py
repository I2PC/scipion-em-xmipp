# ******************************************************************************
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
# ******************************************************************************


import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider, RectangleSelector, Button
from matplotlib.cm import get_cmap, ScalarMappable

from pyworkflow.utils.properties import Message
from pyworkflow.gui.dialog import askYesNo
import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as param

from pwem.emlib.image import ImageHandler
import pwem.emlib.metadata as md
from pwem.protocols import ProtAnalysis2D
from pwem.objects import SetOfClasses2D


class XmippProtCL2DMap(ProtAnalysis2D):
    """ Create a low dimensional mapping from a SetOfClasses2D with interactive selection of classes.
    Use mouse left-click to select/deselect classes individually or mouse right-click to select/deselect
    several classes."""
    
    _label = '2D classes mapping'
    red_methods = ['PCA', 'LTSA', 'DM', 'LLTSA', 'LPP', 'kPCA', 'pPCA', 'LE', 'HLLE', 'SPE', 'NPE']
    distances = ['Euclidean', 'Correlation']
    
    def __init__(self, **args):
        ProtAnalysis2D.__init__(self, **args)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', param.PointerParam,
                      label="Input 2D classes",
                      important=True, pointerClass='SetOfClasses2D',
                      help='Select the input classes to be mapped.')

        form.addSection(label='Mapping')
        form.addParam('method', param.EnumParam,
                      choices=self.red_methods, default=0,
                      label='Dimension reduction method')
        form.addParam('distance', param.EnumParam,
                      choices=self.distances, default=1,
                      label='Distance metric to compare images')

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('computeMappingStep')
        self._insertFunctionStep('interactiveSelStep', interactive=True)

    #--------------------------- STEPS functions -------------------------------
    def computeMappingStep(self):
        self.classes = self.inputClasses.get()
        self.metadata = pwutils.replaceExt(self.classes.getFirstItem().getRepresentative().getFileName(), 'xmd')

        params = dict(metadata=self.metadata, method=self.red_methods[self.method.get()],
                      metric=self.distances[self.distance.get()])
        args = '-i {metadata} -m {method} --distance {metric}'.format(**params)

        self.runJob("xmipp_transform_dimred", args)

    def interactiveSelStep(self):
        self.classes = self.inputClasses.get()
        self.metadata = pwutils.replaceExt(self.classes.getFirstItem().getRepresentative().getFileName(), 'xmd')
        img_paths = np.unique(np.asarray([rep.getFileName() for rep in self.classes.iterRepresentatives()]))
        img_ids = []
        occupancy = []
        pos = []

        mdOut = md.MetaData(self.metadata)
        MDL_DIMRED_COEFFS = 155
        for row in md.iterRows(mdOut):
            pos.append(mdOut.getValue(MDL_DIMRED_COEFFS, row.getObjId()))
            img_ids.append(mdOut.getValue(md.MDL_REF, row.getObjId()))
            occupancy.append(mdOut.getValue(md.MDL_CLASS_COUNT, row.getObjId()))
        occupancy = np.asarray(occupancy)
        pos = np.vstack(pos)

        # Read selected ids from previous runs (if they exist)
        if os.path.isfile(self._getExtraPath('selected_ids,txt')):
            self.selection = np.loadtxt(self._getExtraPath('selected_ids,txt'))
            self.selection = [int(self.selection)] if self.selection.size == 1 else \
                              self.selection.astype(int).tolist()
        else:
            self.selection = None

        view = ScatterImageMarker(pos=pos, img_paths=img_paths, ids=img_ids, occupancy=occupancy,
                                  prevsel=self.selection)
        view.initializePlot()

        self.selection = view.selected_ids

        # Save selected ids for interactive mode
        np.savetxt(self._getExtraPath('selected_ids,txt'), np.asarray(self.selection))

        if askYesNo(Message.TITLE_SAVE_OUTPUT, Message.LABEL_SAVE_OUTPUT, None):
            self._createOutputStep()

    def _createOutputStep(self):
        suffix = self.__getOutputSuffix()
        selected_classes = self._createSetOfClasses2D(self.classes.getImages(), suffix=suffix)

        for class_id in self.selection:
            selected_classes.append(self.classes[class_id].clone())

        result = {'selectedClasses2D_' + suffix: selected_classes}
        self._defineOutputs(**result)
        self._defineSourceRelation(self.inputClasses, selected_classes)


    #--------------------------- INFO functions --------------------------------
    def _summary(self):
        summary = []
        if self.getOutputsSize() >= 1:
            totalClasses = self.inputClasses.get().getSize()
            for key, outClasses in self.iterOutputAttributes():
                summary.append("*Output %s*" % key.split('_')[-1])
                summary.append("A total of %d classes out of %d were selected"
                               % (outClasses.getSize(), totalClasses))
        else:
            summary.append("No classes selected yet.")
        return summary

    def _methods(self):
        methods = []
        methods.append("*Dimensionality reduction method:* %s" % self.red_methods[self.method.get()])
        methods.append("*Classes comparison metric:* %s" % self.distances[self.distance.get()])
        return methods
    
    #--------------------------- UTILS functions -------------------------------
    def __getOutputSuffix(self):
        maxCounter = -1
        for attrName, _ in self.iterOutputAttributes(SetOfClasses2D):
            suffix = attrName.replace('selectedClasses2D_', '')
            try:
                counter = int(suffix)
            except:
                counter = 1  # when there is not number assume 1
            maxCounter = max(counter, maxCounter)

        return str(maxCounter+1) if maxCounter > 0 else '1'  # empty if not outputs


class ScatterImageMarker(object):

    def __init__(self, img_paths, pos, ids, occupancy=None, prevsel=None):
        self.running = True
        self.ids = ids
        self.occupancy = occupancy / np.sum(occupancy) if occupancy is not None else None
        self.selected_ids = prevsel if prevsel is not None else []
        self.pos = pos
        self.readImages(img_paths)
        self.zoom = 30 / np.amax(self.images[0].shape)
        self.pad = 0.6
        self.artists = []
        with plt.style.context('seaborn-darkgrid'):
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('whitesmoke')
        lim_x_low, lim_x_high = -1.5 * np.amax(np.abs(self.pos[:, 0])), 1.5 * np.amax(np.abs(self.pos[:, 0]))
        lim_y_low, lim_y_high = -1.5 * np.amax(np.abs(self.pos[:, 1])), 1.5 * np.amax(np.abs(self.pos[:, 1]))
        self.ax.set_xlim(lim_x_low, lim_x_high)
        self.ax.set_ylim(lim_y_low, lim_y_high)
        plt.title('Interactive Class Selector', fontweight="bold", fontsize=15)
        plt.setp(self.ax.get_yticklabels(), fontweight="bold")
        plt.setp(self.ax.get_xticklabels(), fontweight="bold")
        plt.rcParams["font.weight"] = "bold"

    def readImages(self, img_paths):
        if len(img_paths) == 1:
            self.images = np.squeeze(ImageHandler().read(img_paths[0]).getData())
        else:
            self.images = [np.squeeze(ImageHandler().read(img_path).getData()) for img_path in img_paths]

    def imScatter(self, image, x, y, imid, edge_color=None):
        image = OffsetImage(image, zoom=self.zoom, cmap=plt.cm.gray)
        x, y = np.atleast_1d(x, y)
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=True)
            ab.patch.set_boxstyle("Round, pad={}".format(self.pad))
            if imid in self.selected_ids:
                ab.patch.set_facecolor('palegreen')
            else:
                ab.patch.set_facecolor("lightgray")
            ab.patch.set_alpha(0.5)
            ab.patch.set_linewidth(2)
            if edge_color is not None:
                ab.patch.set_edgecolor(edge_color)
            elif edge_color is None and imid in self.selected_ids:
                ab.patch.set_edgecolor('darkgreen')
            self.artists.append(self.ax.add_artist(ab))

    def is_window_closed(self, event):
        self.running = False

    def initializePlot(self):
        if self.occupancy is not None:
            cmap = get_cmap('cool')
            _ = [self.imScatter(image, x, y, imid, cmap(occupancy))
                 for x, y, image, occupancy, imid in zip(self.pos[:, 0], self.pos[:, 1],
                                                         self.images, self.occupancy, self.ids)]
            cb = self.fig.colorbar(ScalarMappable(cmap=cmap), ax=self.ax, extend='both')
            cb.set_label("Class Occupancy", fontweight="bold", labelpad=15)
        else:
            _ = [self.imScatter(image, x, y, imid)
                 for x, y, image, imid in zip(self.pos[:, 0], self.pos[:, 1], self.images, self.ids)]
        self.ax.scatter(self.pos[:, 0], self.pos[:, 1], alpha=0, picker=True)  # Probably set pickradius param?

        # Picking callback
        def onPickImage(event):
            ind = event.ind[0]
            selected_artist = self.artists[ind]
            selected_id = self.ids[ind]
            patch = selected_artist.patch
            if selected_id in self.selected_ids:
                self.selected_ids.remove(selected_id)
                patch.set_facecolor("lightgray")
                if self.occupancy is None:
                    patch.set_edgecolor('black')
            else:
                self.selected_ids.append(selected_id)
                patch = self.artists[ind].patch
                patch.set_facecolor('palegreen')
                if self.occupancy is None:
                    patch.set_edgecolor('darkgreen')
            self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('pick_event', onPickImage)

        # Slider callback
        slider = plt.axes([0.28, 0.01, 0.3, 0.03], facecolor='lavender') if self.occupancy is not None else \
            plt.axes([0.355, 0.01, 0.3, 0.03], facecolor='lavender')
        self.size_slider = Slider(ax=slider,
                                  label='Image Zoom',
                                  valstep=0.1,
                                  valmin=0,
                                  valmax=2,
                                  valinit=self.zoom,
                                  color='springgreen')

        def updateSize(val):
            self.pad = val * self.pad / self.zoom
            self.zoom = val
            for image, artist in zip(self.images, self.artists):
                image = OffsetImage(image, zoom=self.zoom, cmap=plt.cm.gray)
                artist.offsetbox = image
                artist.patch.set_boxstyle("Round, pad={}".format(self.pad))
            self.fig.canvas.draw_idle()

        self.size_slider.on_changed(updateSize)

        # Rectangle Selector
        def images_select(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            if x1 < x2:
                x_inrange = np.logical_and(self.pos[:, 0] >= x1, self.pos[:, 0] <= x2)
            else:
                x_inrange = np.logical_and(self.pos[:, 0] >= x2, self.pos[:, 0] <= x1)
            if y1 < y2:
                y_inrange = np.logical_and(self.pos[:, 1] >= y1, self.pos[:, 1] <= y2)
            else:
                y_inrange = np.logical_and(self.pos[:, 1] >= y2, self.pos[:, 1] <= y1)
            ids_inrange = np.argwhere(np.logical_and(x_inrange, y_inrange)).flatten()
            for ind in ids_inrange:
                selected_artist = self.artists[ind]
                selected_id = self.ids[ind]
                patch = selected_artist.patch
                if selected_id in self.selected_ids:
                    self.selected_ids.remove(selected_id)
                    patch.set_facecolor("lightgray")
                    if self.occupancy is None:
                        patch.set_edgecolor('black')
                else:
                    self.selected_ids.append(selected_id)
                    patch = self.artists[ind].patch
                    patch.set_facecolor('palegreen')
                    if self.occupancy is None:
                        patch.set_edgecolor('darkgreen')
            self.fig.canvas.draw_idle()

        def toggle_selector(event):
            pass

        rectprops = dict(facecolor='cyan', edgecolor='gray',
                         alpha=0.2, fill=True)
        toggle_selector.RS = RectangleSelector(self.ax, images_select,
                                               drawtype='box', useblit=True,
                                               button=3,  # use only right click
                                               minspanx=5, minspany=5,
                                               spancoords='data',
                                               rectprops=rectprops,
                                               interactive=False)
        self.fig.canvas.mpl_connect('key_press_event', toggle_selector)

        # Selection Buttons
        def selectAll(event):
            self.selected_ids = self.ids.copy()
            for ind in self.ids:
                patch = self.artists[ind].patch
                patch.set_facecolor('palegreen')
                if self.occupancy is None:
                    patch.set_edgecolor('darkgreen')
            self.fig.canvas.draw_idle()

        def selectNone(event):
            self.selected_ids = []
            for ind in self.ids:
                patch = self.artists[ind].patch
                patch.set_facecolor("lightgray")
                if self.occupancy is None:
                    patch.set_edgecolor('black')
            self.fig.canvas.draw_idle()

        axprev = plt.axes([0.65, 0.015, 0.15, 0.035], facecolor='lavender')
        axnext = plt.axes([0.81, 0.015, 0.15, 0.035], facecolor='lavender')
        self.bnall = Button(axnext, 'Select All')
        self.bnall.on_clicked(selectAll)
        self.bnone = Button(axprev, 'Select None')
        self.bnone.on_clicked(selectNone)

        # Wait until interactive plot is closed
        self.fig.canvas.mpl_connect('close_event', self.is_window_closed)
        while self.running:
            plt.pause(.001)
