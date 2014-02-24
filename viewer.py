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
from pyworkflow.viewer import Viewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.em.data import *
from pyworkflow.em.protocol import *
from pyworkflow.utils.process import runJob
from xmipp3 import getXmippPath
from protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from protocol_ctf_micrographs import XmippProtCTFMicrographs
from protocol_extract_particles import XmippProtExtractParticles, ProtImportParticles
from protocol_cl2d_align import XmippProtCL2DAlign
from protocol_cl2d import XmippProtCL2D
from protocol_kerdensom import XmippProtKerdensom
from protocol_rotational_spectra import XmippProtRotSpectra
from protocol_create_mask3d import XmippProtCreateMask3D
from protocol_screen_classes import XmippProtScreenClasses
from protocol_helical_parameters import XmippProtHelicalParameters
from protocol_convert_to_pseudoatoms import XmippProtConvertToPseudoAtoms
from protocol_identify_outliers import XmippProtIdentifyOutliers

from convert import writeSetOfMicrographs, writeSetOfParticles, writeSetOfVolumes, writeSetOfClasses2D, writeSetOfCoordinates, writeSetOfCTFs, locationToXmipp, \
                    writeSetOfClasses3D
from os.path import dirname, join
from pyworkflow.utils.path import makePath
import pyworkflow as pw


import xmipp


class XmippViewer(Viewer):
    """ Wrapper to visualize different type of objects
    with the Xmipp program xmipp_showj
    """
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [Image, SetOfImages, SetOfCoordinates, SetOfClasses2D, SetOfClasses3D, 
                ProtExtractParticles,
                ProtAlign, XmippProtKerdensom, XmippProtRotSpectra,  XmippProtCreateMask3D,
                SetOfClasses2D, SetOfCTF, NormalModes, XmippProtScreenClasses,
                XmippProtConvertToPseudoAtoms, XmippProtIdentifyOutliers]
    
    def __init__(self, **args):
        Viewer.__init__(self, **args)

    def visualize(self, obj, **args):
        
        cls = type(obj)
        
        if issubclass(cls, Image):
            fn = locationToXmipp(*obj.getLocation())
            runShowJ(fn)
            
        elif issubclass(cls, NormalModes):
            runShowJ(obj.getFileName())         
              
        elif issubclass(cls, SetOfMicrographs):
            
            mdFn = getattr(obj, '_xmippMd', None)
            if mdFn:
                fn = mdFn.get()
            else:
                fn = self._getTmpPath(obj.getName() + '_micrographs.xmd')
                writeSetOfMicrographs(obj, fn)
                
            extra = ''
            if obj.hasCTF():
                extra = ' --mode metadata --render first'
            #runShowJ(fn, extraParams=extra)
            runScipionShowJ(fn, "Micrographs", self._project.getName(), obj.strId())    
        
        elif issubclass(cls, SetOfCoordinates):
            obj_mics = obj.getMicrographs()#accessing mics to provide metadata file
            mdFn = getattr(obj_mics, '_xmippMd', None)
            tmpDir = self._getTmpPath(obj.getName()) 
            makePath(tmpDir)
            
            if mdFn:
                fn = mdFn.get()
            else: # happens if protocol is not an xmipp one
                fn = self._getTmpPath(obj_mics.getName() + '_micrographs.xmd')
                writeSetOfMicrographs(obj_mics, fn)
            posDir = getattr(obj, '_xmippMd', None)#extra dir istead of md file for SetOfCoordinates
            if posDir:
                copyTree(posDir.get(), tmpDir)
            else:
                 writeSetOfCoordinates(tmpDir, obj)   
                           
            runScipionParticlePicker(fn, tmpDir, self._project.getName(), obj.strId())
        
        elif issubclass(cls, SetOfParticles) or issubclass(cls, SetOfVolumes):
            mdFn = getattr(obj, '_xmippMd', None)
            if mdFn:
                fn = mdFn.get()
            else:
                fn = self._getTmpPath(obj.getName() + '_images.xmd')
                #Set hasCTF to False to avoid problems
                if issubclass(cls, SetOfParticles):
                    writeSetOfParticles(obj, fn)
                else:
                    writeSetOfVolumes(obj, fn)
            if issubclass(cls, SetOfParticles):
                runScipionShowJ(fn, "Particles", self._project.getName(), obj.strId())  
            else:
                runShowJ(fn)
        
        elif issubclass(cls, SetOfClasses2D):
            mdFn = getattr(obj, '_xmippMd', None)
            if mdFn:
                fn = mdFn.get()
            else:
                fn = self._getTmpPath(obj.getName() + '_classes.xmd')
                writeSetOfClasses2D(obj, fn, self._getTmpPath())
            runShowJ(fn, extraParams=args.get('extraParams', ''))  
            
        elif issubclass(cls, SetOfClasses3D):
            mdFn = getattr(obj, '_xmippMd', None)
            if mdFn:
                fn = mdFn.get()
            else:
                fn = self._getTmpPath(obj.getName() + '_classes.xmd')
                writeSetOfClasses3D(obj, fn, self._getTmpPath())

            runShowJ("classes@"+fn, extraParams=args.get('extraParams', ''))
              
        elif issubclass(cls, SetOfCTF):
            mdFn = getattr(obj, '_xmippMd', None)
            if mdFn:
                fn = mdFn.get()
            else:
                fn = self._getTmpPath(obj.getName() + '_ctfs.xmd')
                writeSetOfCTFs(obj, fn)
            runShowJ(fn, extraParams=' --mode metadata --render first')  

        

        
        elif (issubclass(cls, XmippProtExtractParticles)):
            self.visualize(obj.outputParticles)
            fn = getattr(obj.outputParticles, '_xmippMd', None)
            if fn:
                md = xmipp.MetaData(fn) 
                #print "MD=%s" % obj.outputParticles.getFileName()
                if md.containsLabel(xmipp.MDL_ZSCORE):
                    from plotter import XmippPlotter
                    xplotter = XmippPlotter(windowTitle="Zscore particles sorting")
                    xplotter.createSubPlot("Particle sorting", "Particle number", "Zscore")
                    xplotter.plotMd(md, False, mdLabelY=xmipp.MDL_ZSCORE)
                    xplotter.show()
            # If Zscore on output images plot Zscore particle sorting
    
        elif issubclass(cls, ProtAlign):
            self.visualize(obj.outputAverage)
            self.visualize(obj.outputParticles) 
            
        elif issubclass(cls, XmippProtRotSpectra):
            self.visualize(obj.outputClasses, extraParams='--mode rotspectra --columns %d' % obj.SomXdim.get())

        
        elif issubclass(cls, XmippProtScreenClasses):
            runShowJ(obj.getVisualizeInfo().get(), extraParams=' --mode metadata --render first')
        
        elif issubclass(cls, XmippProtIdentifyOutliers):
            runShowJ(obj.getVisualizeInfo().get(), extraParams=' --mode metadata --render first')

        elif issubclass(cls, XmippProtCreateMask3D):
            self.visualize(obj.outputMask)
            
        elif issubclass(cls, XmippProtKerdensom):
            self.visualize(obj.outputClasses, extraParams='--columns %d' % obj.SomXdim.get())

        elif issubclass(cls, XmippProtConvertToPseudoAtoms):
            from protlib_gui_ext import chimera
            chimera(obj._getPath('chimera.cmd'))
        else:
            raise Exception('XmippViewer.visualize: can not visualize class: %s' % obj.getClassName())
        
    @classmethod
    def getView(self):
        return "viewerXmipp"
        
# ------------- Xmipp utilities function to launch Java applications ------------

def getArchitecture():
    import platform
    arch = platform.architecture()[0]
    for a in ['32', '64']:
        if a in arch:
            return a
    return 'NO_ARCH' 
    
def getJavaIJappArguments(memory, appName, appArgs):
    """ Build the command line arguments to launch 
    a Java application based on ImageJ. 
    memory: the amount of memory passed to the JVM.
    appName: the qualified name of the java application.
    appArgs: the arguments specific to the application.
    """ 
    if len(memory) == 0:
        memory = "1g"
        print "No memory size provided. Using default: " + memory
    imagej_home = getXmippPath("external", "imagej")
    lib = getXmippPath("lib")
    javaLib = getXmippPath('java', 'lib')
    plugins_dir = os.path.join(imagej_home, "plugins")
    arch = getArchitecture()
    args = "-Xmx%(memory)s -d%(arch)s -Djava.library.path=%(lib)s -Dplugins.dir=%(plugins_dir)s -cp %(imagej_home)s/*:%(javaLib)s/* %(appName)s %(appArgs)s" % locals()

    return args
    
def runJavaIJapp(memory, appName, args, batchMode=True):
    args = getJavaIJappArguments(memory, appName, args)
    runJob(None, "java", args, runInBackground=batchMode)
    
def runShowJ(inputFiles, memory="1g", extraParams=""):
    runJavaIJapp(memory, "'xmipp.viewer.Viewer'", "-i %s %s" % (inputFiles, extraParams), True)
    
def runScipionShowJ(inputFiles, type, projectid, objid, memory="1g", extraParams=""):

    script = pw.join('apps', 'pw_create_image_subset.py')
    script = "%s %s" % (pw.PYTHON, script) 

    runJavaIJapp(memory, "'xmipp.viewer.scipion.ScipionViewer'", "-i %s %s --scipion \"%s\" \"%s\" \"%s\" %s" % (inputFiles, extraParams, type, script, projectid, objid), True)

def runParticlePicker(inputMics, inputCoords, memory="1g", extraParams=""):
    runJavaIJapp(memory, "xmipp.viewer.particlepicker.training.SupervisedPickerRunner", "--input %s --output %s %s" % (inputMics, inputCoords, extraParams), True)
    
def runScipionParticlePicker(inputMics, inputCoords,  projectid, objid, memory="1g"):
    script = pw.join('apps', 'pw_create_coords_subset.py')
    script = "%s %s" % (pw.PYTHON, script) 
    runJavaIJapp(memory, "xmipp.viewer.particlepicker.training.SupervisedPickerRunner", "--input %s --output %s --mode review --scipion \"%s\" \"%s\" %s" % (inputMics, inputCoords, script, projectid, objid), True)



