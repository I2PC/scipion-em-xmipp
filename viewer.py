# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
This module implement the wrappers around xmipp_showj
visualization program.
"""

import os

from pyworkflow.viewer import Viewer, DESKTOP_TKINTER, WEB_DJANGO, CommandView
from pyworkflow.em.data import *
from pyworkflow.em.protocol import *

from xmipp3 import getXmippPath, getEnviron
from pyworkflow.em.data_tiltpairs import MicrographsTiltPair, ParticlesTiltPair, CoordinatesTiltPair
from convert import *
from os.path import dirname, join
from pyworkflow.utils import makePath, runJob, copyTree
import pyworkflow as pw
import xmipp

from protocol_cl2d_align import XmippProtCL2DAlign
from protocol_cl2d import XmippProtCL2D
from protocol_ctf_discrepancy import XmippProtCTFDiscrepancy
from protocol_extract_particles import XmippProtExtractParticles
from protocol_extract_particles_pairs import XmippProtExtractParticlesPairs
from protocol_helical_parameters import XmippProtHelicalParameters
from protocol_kerdensom import XmippProtKerdensom
from protocol_particle_pick_automatic import XmippParticlePickingAutomatic
from protocol_particle_pick import XmippProtParticlePicking
from protocol_particle_pick_pairs import XmippProtParticlePickingPairs
from protocol_preprocess import XmippProtPreprocessVolumes
from protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from protocol_projection_outliers import XmippProtProjectionOutliers
from protocol_rotational_spectra import XmippProtRotSpectra
from protocol_screen_classes import XmippProtScreenClasses
from protocol_screen_particles import XmippProtScreenParticles
from protocol_ctf_micrographs import XmippProtCTFMicrographs
from pyworkflow.em.showj import *
from protocol_movie_alignment import ProtMovieAlignment


class XmippViewer(Viewer):
    """ Wrapper to visualize different type of objects
    with the Xmipp program xmipp_showj
    """
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [                  
                CoordinatesTiltPair, 
                Image, 
                MicrographsTiltPair, 
                ParticlesTiltPair, 
                ProtExtractParticles, 
                SetOfClasses2D, 
                SetOfClasses3D, 
                SetOfCoordinates, 
                SetOfCTF, 
                SetOfImages, 
                SetOfMovies, 
                SetOfNormalModes, 
                XmippParticlePickingAutomatic, 
                XmippProtExtractParticlesPairs, 
                XmippProtKerdensom, 
                XmippProtParticlePicking, 
                XmippProtParticlePickingPairs,
                XmippProtProjectionOutliers, 
                XmippProtRotSpectra, 
                XmippProtScreenClasses, 
                XmippProtScreenParticles, 
                XmippProtCTFMicrographs, 
                ProtMovieAlignment
                ]
    
    def __init__(self, **args):
        Viewer.__init__(self, **args)
        self._views = []   
        
    def visualize(self, obj, **args):
        self._visualize(obj, **args)
        
        for v in self._views:
            v.show()
            
    def _visualize(self, obj, **args):
        cls = type(obj)
        def _getMicrographDir(mic):
            """ Return an unique dir name for results of the micrograph. """
            return obj._getExtraPath(removeBaseExt(mic.getFileName()))        
    
        def iterMicrographs(mics):
            """ Iterate over micrographs and yield
            micrograph name and a directory to process.
            """
            for mic in mics:
                micFn = mic.getFileName()
                micDir = _getMicrographDir(mic) 
                yield (micFn, micDir, mic)
    
        def visualizeCTFObjs(obj, setOfMics):
            
            if exists(obj._getPath("ctfs_temporary.sqlite")):
                os.remove(obj._getPath("ctfs_temporary.sqlite"))
            self.protocol._createFilenameTemplates()
            
            ctfSet = self.protocol._createSetOfCTF("_temporary")
            
            for fn, micDir, mic in iterMicrographs(setOfMics):
                ctfparam = self.protocol._getFileName('ctfparam', micDir=micDir)
                
                if exists(ctfparam) or exists('xmipp_default_ctf.ctfparam'):
                    if not os.path.exists(ctfparam):
                        ctfparam = 'xmipp_default_ctf.ctfparam'
                    
                    ctfModel = readCTFModel(ctfparam, mic)
                    self.protocol._setPsdFiles(ctfModel, micDir)
                    ctfSet.append(ctfModel)
            
            if ctfSet.getSize() < 1:
                raise Exception("Has not been completed the CTT estimation of any micrograph")
            else:
                ctfSet.write()
                ctfSet.close()
                self._visualize(ctfSet)

        if issubclass(cls, Volume):
            fn = getImageLocation(obj)
            self._views.append(DataView(fn, viewParams={RENDER: 'image', SAMPLINGRATE: obj.getSamplingRate()}))
                 
        elif issubclass(cls, Image):
            fn = getImageLocation(obj)

            self._views.append(ObjectView(self._project, obj.strId(), fn))
            
        elif issubclass(cls, SetOfNormalModes):
            fn = obj.getFileName()
            objCommands = "'%s' '%s'" % (OBJCMD_NMA_PLOTDIST, OBJCMD_NMA_VMD)
            self._views.append(ObjectView(self._project, self.protocol.strId(), fn, viewParams={OBJCMDS: objCommands}, **args))

        elif issubclass(cls, SetOfMovies):
            fn = obj.getFileName()
            # Enabled for the future has to be available
            labels = 'id _filename _samplingRate  '
            self._views.append(ObjectView(self._project, obj.strId(), fn,
                                          viewParams={ORDER: labels, 
                                                      VISIBLE: labels, 
                                                      MODE: MODE_MD,
                                                      RENDER:'_filename'}))

        elif issubclass(cls, SetOfMicrographs):            
            fn = obj.getFileName()
            self._views.append(ObjectView(self._project, obj.strId(), fn, **args))
            

        elif issubclass(cls, MicrographsTiltPair):          
#             fnU = obj.getUntilted().getFileName()
#             fnT = obj.getTilted().getFileName()
#             self._views.append(ObjectView(self._project.getName(), obj.strId(), fnU, **args))            
#             self._views.append(ObjectView(self._project.getName(), obj.strId(), fnT, **args))
            labels = 'id enabled _untilted._filename _tilted._filename'
            self._views.append(ObjectView(self._project, obj.strId(), obj.getFileName(),
                                          viewParams={ORDER: labels, 
                                                      VISIBLE: labels, 
                                                      MODE: MODE_MD,
                                                      RENDER: '_untilted._filename _tilted._filename'}))
            
        elif issubclass(cls, ParticlesTiltPair):          
            labels = 'id enabled _untilted._filename _tilted._filename'
            self._views.append(ObjectView(self._project, obj.strId(), obj.getFileName(),
                                          viewParams={ORDER: labels, 
                                                      VISIBLE: labels, RENDER:'_untilted._filename _tilted._filename',
                                                      MODE: MODE_MD}))

                            
        elif issubclass(cls, SetOfCoordinates):
            micSet = obj.getMicrographs()  # accessing mics to provide metadata file
            if micSet is None:
                raise Exception('visualize: SetOfCoordinates has no micrographs set.')
            
            mdFn = getattr(micSet, '_xmippMd', None)
            tmpDir = self._getTmpPath(obj.getName())
            makePath(tmpDir)
            
            if mdFn:
                fn = mdFn.get()
            else:  # happens if protocol is not an xmipp one
                fn = self._getTmpPath(micSet.getName() + '_micrographs.xmd')
                writeSetOfMicrographs(micSet, fn)
            posDir = getattr(obj, '_xmippMd', None)  # extra dir istead of md file for SetOfCoordinates
            if posDir:
                copyTree(posDir.get(), tmpDir)
            else:
                writeSetOfCoordinates(tmpDir, obj)

            self._views.append(CoordinatesObjectView(self._project, fn, tmpDir))

        elif issubclass(cls, SetOfParticles):
            fn = obj.getFileName()
            labels = 'id enabled _index _filename _xmipp_zScore _xmipp_cumulativeSSNR _sampling '
            labels += '_ctfModel._defocusU _ctfModel._defocusV _ctfModel._defocusAngle _transform._matrix'
            self._views.append(ObjectView(self._project, obj.strId(), fn,
                                          viewParams={ORDER: labels, 
                                                      VISIBLE: labels, 
                                                      'sortby': '_xmipp_zScore asc', RENDER:'_filename'}))
               
        elif issubclass(cls, SetOfVolumes):
            fn = obj.getFileName()
            labels = 'id enabled comment _filename '
            self._views.append(ObjectView(self._project, obj.strId(), fn,
                                          viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels, RENDER: '_filename'}))
        
        elif issubclass(cls, SetOfClasses2D):
            fn = obj.getFileName()
            self._views.append(ClassesView(self._project, obj.strId(), fn, **args))
            
        elif issubclass(cls, SetOfClasses3D):
            fn = obj.getFileName()
            self._views.append(Classes3DView(self._project, obj.strId(), fn))
        
        if issubclass(cls, XmippProtCTFMicrographs) and not obj.hasAttribute("outputCTF"):
            mics = obj.inputMicrographs.get()
            visualizeCTFObjs(obj, mics)

        elif obj.hasAttribute("outputCTF"):
            self._visualize(obj.outputCTF)
        
        elif issubclass(cls, SetOfCTF):
            fn = obj.getFileName()
#            self._views.append(DataView(fn, viewParams={MODE: 'metadata'}))
            psdLabels = '_psdFile _xmipp_enhanced_psd _xmipp_ctfmodel_quadrant _xmipp_ctfmodel_halfplane'
            labels = 'id enabled label %s _defocusU _defocusV _defocusAngle _defocusRatio ' \
                     '_xmipp_ctfCritFirstZero _xmipp_ctfCritCorr13 _xmipp_ctfCritFitting _xmipp_ctfCritNonAstigmaticValidity ' \
                     '_xmipp_ctfCritCtfMargin _micObj._filename' % psdLabels #TODO:CHECK IF _xmipp_ctfCritNonAstigmaticValidity AND _xmipp_ctfCritCtfMargin exist sometimes. 
            self._views.append(ObjectView(self._project, obj.strId(), fn,
                                          viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels, ZOOM: 50, RENDER: psdLabels}))    

        elif issubclass(cls, CoordinatesTiltPair):
            tmpDir = self._getTmpPath(obj.getName()) 
            makePath(tmpDir)

            mdFn = join(tmpDir, 'input_micrographs.xmd')
            writeSetOfMicrographsPairs(obj.getUntilted().getMicrographs(),
                                        obj.getTilted().getMicrographs(), 
                                        mdFn) 
            parentProtId = obj.getObjParentId()
            parentProt = self.getProject().mapper.selectById(parentProtId)
            extraDir = parentProt._getExtraPath()
            
            extraDir = parentProt._getExtraPath()
            #TODO: Review this if ever a non Xmipp CoordinatesTiltPair is available
#             writeSetOfCoordinates(tmpDir, obj.getUntilted()) 
#             writeSetOfCoordinates(tmpDir, obj.getTilted()) 
            
            scipion =  "%s \"%s\" %s" % (self.getProject().port, self.getProject().getDbPath(), obj.strId())
            app = "xmipp.viewer.particlepicker.tiltpair.TiltPairPickerRunner"
            args = " --input %(mdFn)s --output %(extraDir)s --mode readonly --scipion %(scipion)s"%locals()
        
            runJavaIJapp("2g", app, args)
         
        elif issubclass(cls, XmippProtExtractParticles) or issubclass(cls, XmippProtScreenParticles):
            particles = obj.outputParticles
            self._visualize(particles)
            
            fn = obj._getPath('images.xmd')
            md = xmipp.MetaData(fn) 
            # If Zscore on output images plot Zscore particle sorting
            if md.containsLabel(xmipp.MDL_ZSCORE):
                from plotter import XmippPlotter
                xplotter = XmippPlotter(windowTitle="Zscore particles sorting")
                xplotter.createSubPlot("Particle sorting", "Particle number", "Zscore")
                xplotter.plotMd(md, False, mdLabelY=xmipp.MDL_ZSCORE)
                self._views.append(xplotter)
    
        elif issubclass(cls, XmippProtRotSpectra):
            self._visualize(obj.outputClasses, viewParams={#'mode': 'rotspectra', 
                                                           'columns': obj.SomXdim.get(),
                                                           RENDER: 'average._filename spectraPlot._filename',
                                                           VISIBLE:  'enabled id _size average._filename spectraPlot._filename',
                                                           'labels': '_size',
                                                           SORT_BY: 'id'})
        
        elif issubclass(cls, XmippProtKerdensom):
            self._visualize(obj.outputClasses, viewParams={'columns': obj.SomXdim.get(),
                                                           'render': 'average._filename _representative._filename',
                                                           'labels': '_size',
                                                           'sortby': 'id'})
        
        elif issubclass(cls, XmippProtScreenClasses):
            if isinstance(obj.inputSet.get(), SetOfClasses2D):
                fn = obj.outputClasses
                labels = 'id enabled _size _representative._filename _xmipp_imageRef _xmipp_image1 _xmipp_maxCC'
                labelRender = "_representative._filename _xmipp_imageRef _xmipp_image1"
                self._visualize(fn, viewParams={ORDER: labels, 
                                                          VISIBLE: labels, 
                                                          SORT_BY: '_xmipp_maxCC desc', RENDER:labelRender,
                                                          MODE: MODE_MD})
            else:
                fn = obj.outputAverages.getFileName()
                labels = 'id enabled _filename _xmipp_imageRef _xmipp_image1 _xmipp_maxCC'
                labelRender = "_filename _xmipp_imageRef _xmipp_image1"
                self._views.append(ObjectView(self._project, obj.outputAverages.strId(), fn,
                                              viewParams={ORDER: labels, 
                                                      VISIBLE: labels, 
                                                      SORT_BY: '_xmipp_maxCC desc', RENDER:labelRender,
                                                      MODE: MODE_MD}))
        
        elif issubclass(cls, XmippProtProjectionOutliers):
            if isinstance(obj.inputSet.get(), SetOfClasses2D):
                fn = obj.outputClasses
                labels = 'id enabled _size _representative._index _representative._filename _xmipp_maxCC _xmipp_zScoreResCov _xmipp_zScoreResMean _xmipp_zScoreResVar'
                labelRender = "_representative._filename"
                self._visualize(fn, viewParams={ORDER: labels, 
                                                          VISIBLE: labels, 
                                                          SORT_BY: '_xmipp_zScoreResCov desc', RENDER:labelRender})
            else:
                fn = obj.outputAverages.getFileName()
                labels = 'id enabled _index _filename  _xmipp_maxCC _xmipp_zScoreResCov _xmipp_zScoreResMean _xmipp_zScoreResVar _transform._matrix'
                labelRender = "_filename"
                self._views.append(ObjectView(self._project, obj.outputAverages.strId(), fn,
                                              viewParams={ORDER: labels, 
                                                      VISIBLE: labels, 
                                                      SORT_BY: '_xmipp_zScoreResCov desc', RENDER:labelRender}))
            
        elif issubclass(cls, XmippProtParticlePicking):
            if obj.getOutputsSize() >= 1:
                self._visualize(obj.getCoords())
            
        elif issubclass(cls, XmippParticlePickingAutomatic):
            micSet = obj.getInputMicrographs()
            mdFn = getattr(micSet, '_xmippMd', None)
            if mdFn:
                micsfn = mdFn.get()
            else:  # happens if protocol is not an xmipp one
                micsfn = self._getTmpPath(micSet.getName() + '_micrographs.xmd')
                writeSetOfMicrographs(micSet, micsfn)
                
            posDir = getattr(obj.getCoords(), '_xmippMd').get()  # extra dir istead of md file for SetOfCoordinates
            launchSupervisedPickerGUI(2, micsfn, posDir, 'review', self.getProject().getDbPath(), obj.strId(), self.getProject().port)

        elif issubclass(cls, XmippProtParticlePickingPairs):
            tmpDir = self._getTmpPath(obj.getName()) 
            makePath(tmpDir)

            mdFn = join(tmpDir, 'input_micrographs.xmd')
            writeSetOfMicrographsPairs(obj.outputCoordinatesTiltPair.getUntilted().getMicrographs(),
                                        obj.outputCoordinatesTiltPair.getTilted().getMicrographs(), 
                                        mdFn) 
            extraDir = obj._getExtraPath()
            launchTiltPairPickerGUI(obj.memory.get(), mdFn, extraDir, 'readonly', self.getProject().getDbPath(), obj.strId(), self.getProject().port)

        elif issubclass(cls, ProtMovieAlignment):
            outputMics = obj.outputMicrographs
            plotLabels = 'psdCorr._filename plotPolar._filename plotCart._filename'
            labels = plotLabels + ' _filename '
            objCommands = "'%s' '%s' '%s'" % (OBJCMD_MOVIE_ALIGNPOLAR, OBJCMD_MOVIE_ALIGNCARTESIAN, OBJCMD_MOVIE_ALIGNPOLARCARTESIAN)
            
            self._views.append(ObjectView(self._project, obj.strId(), outputMics.getFileName(), viewParams={MODE: MODE_MD,
                                                      ORDER: labels, VISIBLE: labels, RENDER: plotLabels, 'zoom': 50,
                                                      OBJCMDS: objCommands}))


        elif issubclass(cls, XmippProtExtractParticlesPairs):
            self._visualize(obj.outputParticlesTiltPair)

        return self._views
    
    
# class ChimeraClient(CommandView):
#     """ View for calling an external command. """
#     def __init__(self, inputFile, projectionSize=256, 
#                  angularDist=None, radius=None, sphere=None, **kwargs):
# 
#         cmd = 'xmipp_chimera_client --input "%(inputFile)s" --mode projector %(projectionSize)d' % locals()
#         if angularDist:
#             cmd += ' -a %(angularDist)s red %(radius)f' % locals() 
#             if sphere > 0:
#                 cmd += ' %f' % sphere
#         CommandView.__init__(self, cmd + ' &', env=getEnviron())
#         
#     def show(self):
#         from subprocess import call
#         call(self._cmd, shell=True, env=self._env)
        
