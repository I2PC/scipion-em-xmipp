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
from pyworkflow.em.packages.xmipp3.protocol_kerdensom import XmippProtKerdensom
"""
This module implement the wrappers around xmipp_showj
visualization program.
"""

import os
from pyworkflow.em.viewer import Viewer, Wizard
from pyworkflow.em import SetOfImages, SetOfMicrographs, DefCTFMicrographs
from pyworkflow.utils.process import runJob
from xmipp3 import getXmippPath
from data import XmippSetOfImages, XmippSetOfMicrographs, XmippClassification2D, XmippSetOfCoordinates, XmippSetOfParticles
from pyworkflow.em.protocol import ProtImportMicrographs
from protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from protocol_ctf_micrographs import XmippProtCTFMicrographs
from protocol_particle_pick import XmippProtParticlePicking
from protocol_extract_particles import XmippProtExtractParticles, ProtImportParticles
from protocol_cl2d_align import XmippProtCL2DAlign
from protocol_ml2d import XmippProtML2D
from protocol_cl2d import XmippProtCL2D
from protocol_kerdensom import XmippProtKerdensom
from protocol_rotational_spectra import XmippProtRotSpectra
from convert import writeSetOfMicrographs


import xmipp
from plotter import XmippPlotter

class XmippViewer(Viewer):
    """ Wrapper to visualize different type of objects
    with the Xmipp program xmipp_showj
    """
    _targets = [SetOfImages, XmippSetOfCoordinates, XmippClassification2D, 
                ProtImportMicrographs, XmippProtPreprocessMicrographs, XmippProtCTFMicrographs,
                XmippProtParticlePicking, ProtImportParticles, XmippProtExtractParticles,
                XmippProtCL2DAlign, XmippProtML2D, XmippProtCL2D]
    
    def __init__(self, **args):
        Viewer.__init__(self, **args)

    def visualize(self, obj, **args):
        cls = type(obj)
        
        if issubclass(cls, SetOfMicrographs):
            mdFn = getattr(obj, '_xmippMd', None)
            if mdFn:
                fn = mdFn.get()
            else:
                fn = self._getTmpPath(obj.getName() + '_micrographs.xmd')
                writeSetOfMicrographs(obj, fn)
            extra = ''
            if obj.hasCTF():
                extra = ' --mode metadata --render first'
            runShowJ(fn, extraParams=extra)  
        
        elif issubclass(cls, XmippSetOfCoordinates):
            runParticlePicker(obj.getMicrographs().getFileName(), os.path.dirname(obj.getFileName()), extraParams='readonly')
        
        elif issubclass(cls, SetOfImages):
            fn = self._getTmpPath(obj.getName() + '_images.xmd')
            imgs = XmippSetOfImages.convert(obj, fn)
            runShowJ(imgs.getFileName())
        
        elif issubclass(cls, XmippClassification2D):
            runShowJ(obj.getClassesMdFileName(), extraParams=args.get('extraParams', ''))
        
        elif (issubclass(cls, ProtImportMicrographs) or
              issubclass(cls, XmippProtPreprocessMicrographs) or
              issubclass(cls, XmippProtCTFMicrographs)):
            self.visualize(obj.outputMicrographs)
        
        elif issubclass(cls, XmippProtParticlePicking):
            self.visualize(obj.outputCoordinates)
        
        elif (issubclass(cls, ProtImportParticles) or
              issubclass(cls, XmippProtExtractParticles)):
            self.visualize(obj.outputParticles)
            # If Zscore on output images plot Zscore particle sorting            
            md = xmipp.MetaData(obj.outputParticles.getFileName()) 
            print "MD=%s" % obj.outputParticles.getFileName()
            if md.containsLabel(xmipp.MDL_ZSCORE):
                print "MD contains ZSCORE"
                xplotter = XmippPlotter(windowTitle="Zscore particles sorting")
                xplotter.createSubPlot("Particle sorting", "Particle number", "Zscore")
                xplotter.plotMd(md, False, mdLabelY=xmipp.MDL_ZSCORE)
                xplotter.show()      
        
        elif (issubclass(cls, XmippProtCL2DAlign) or
              issubclass(cls, XmippProtML2D) or
              issubclass(cls, XmippProtCL2D)):
            self.visualize(obj.outputClassification)
        
        elif issubclass(cls, XmippProtRotSpectra):
            self.visualize(obj.outputClassification, extraParams='--mode rotspectra --columns %d' % obj.SomXdim.get())
        elif issubclass(cls, XmippProtKerdensom):
            self.visualize(obj.outputClassification, extraParams='--columns %d' % obj.SomXdim.get())
        else:
            raise Exception('XmippViewer.visualize: can not visualize class: %s' % obj.getClassName())

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

def runParticlePicker(inputMics, inputCoords, memory="1g", extraParams=""):
    runJavaIJapp(memory, "''xmipp.viewer.particlepicker.training.Main''", "%s %s %s" % (inputMics, inputCoords, extraParams), True)

