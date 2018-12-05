# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *              David Maluenda Niubo (dmaluenda@cnb.csic.es) [2]
# *
# * [1] SciLifeLab, Stockholm University
# * [2] Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

import subprocess

import pyworkflow.em
import pyworkflow.utils as pwutils

from .base import *
from .constants import XMIPP_HOME


_logo = "xmipp_logo.png"
_references = ['delaRosaTrevin2013', 'Sorzano2013']
_currentVersion = '3.18.08'

class Plugin(pyworkflow.em.Plugin):
    _homeVar = XMIPP_HOME
    _pathVars = [XMIPP_HOME]
    _supportedVersions = []

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(XMIPP_HOME, 'xmipp-%s'%_currentVersion)

    @classmethod
    def getEnviron(cls, xmippFirst=True):
        """ Create the needed environment for Xmipp programs. """
        environ = pwutils.Environ(os.environ)
        pos = pwutils.Environ.BEGIN if xmippFirst else pwutils.Environ.END
        environ.update({
            'PATH': getXmippPath('bin'),
            'LD_LIBRARY_PATH': getXmippPath('lib'),
        }, position=pos)

        if os.environ['CUDA'] != 'False':  # environ variables are strings not booleans
            environ.update({
                'LD_LIBRARY_PATH': os.environ['NVCC_LIBDIR']
            }, position=pos)

        return environ

    #TODO: Standardize to just: runProgram
    @classmethod
    def runXmippProgram(cls, program, args=""):
        """ Internal shortcut function to launch a Xmipp program. """
        pwutils.runJob(None, program, args, env=cls.getEnviron())

    @classmethod
    def getMatlabEnviron(cls, *toolPaths):
        """ Return an Environment prepared for launching Matlab
        scripts using the Xmipp binding.
        """
        env = pwutils.getEnviron()
        env.set('PATH', os.environ['MATLAB_BINDIR'], pwutils.Environ.BEGIN)
        env.set('LD_LIBRARY_PATH', os.environ['MATLAB_LIBDIR'], pwutils.Environ.BEGIN)
        for toolpath in toolPaths:
            env.set('MATLABPATH', toolpath, pwutils.Environ.BEGIN)
        env.set('MATLABPATH', os.path.join(os.environ[XMIPP_HOME], 'libraries', 'bindings', 'matlab'),
                pwutils.Environ.BEGIN)

        return env

    @classmethod
    def getHDF5model(self, modelName):
        return getXmippPath('models', modelName)

    @classmethod
    def defineBinaries(cls, env):

        ## XMIPP SOFTWARE ##

        installCmd = ("src/xmipp/xmipp config ; src/xmipp/xmipp check_config ;"
                      "src/xmipp/xmipp compile %d ; src/xmipp/xmipp install %s"
                       % (env.getProcessors(), cls.getHome()))

        target = "%s/bin/xmipp_reconstruct_significant" % cls.getHome()

        xmippSrc = env.addPackage('xmippSrc', version=_currentVersion,
                                  tar='xmippSrc-%s.tgz'%_currentVersion,
                                  commands=[(installCmd, target)])

        xmippBin = env.addPackage('xmippBin', version=_currentVersion,
                                  tar='xmipp-%s.tgz' %_currentVersion,
                                  default=True)

        # Old dependencies now are taken into account inside xmipp script:
        #   scons, fftw3, scikit, nma, tiff, sqlite, opencv, sh_alignment, hdf5



        ## EXTRA PACKAGES ##

        ## --- DEEP LEARNING TOOLKIT --- ##

        scipy = env.addPipModule('scipy', '0.14.0', default=False)#,
                                 # deps=[lapack, matplotlib])
        cython = env.addPipModule('cython', '0.22', target='Cython-0.22*', default=False)

        scikit_learn = env.addPipModule('scikit-learn', '0.19.1',
                                        target='scikit_learn*',
                                        default=False, deps=[scipy, cython])
        unittest2 = env.addPipModule('unittest2', '0.5.1', target='unittest2*', default=False)
        h5py = env.addPipModule('h5py', '2.8.0rc1', target='h5py*', default=False, deps=[unittest2])

        cv2 = env.addPipModule('opencv-python', "3.4.2.17",
                               target="cv2", default=False)
        # TensorFlow
        tensorFlowTarget = "1.10.0" #cuda 9
        nvccProgram = subprocess.Popen(["which", "nvcc"],
                                       stdout=subprocess.PIPE).stdout.read()
        pipCmdScipion = '%s %s/pip install' % (env.getBin('python'),
                                               env.getPythonPackagesFolder())
        if nvccProgram != "":
            nvccVersion = subprocess.Popen(["nvcc", '--version'],
                                           stdout=subprocess.PIPE).stdout.read()
            #TODO: check if cuda 9 or 10  or 7.5 ...
            if "release 8.0" in nvccVersion: #cuda 8
                tensorFlowTarget = "1.4.1"

            tensor = env.addPipModule('tensorflow-gpu', target='tensorflow*',
                                      default=False,
                                      pipCmd="%s https://storage.googleapis.com/"
                                             "tensorflow/linux/gpu/tensorflow_gpu-%s-cp27-none-"
                                             "linux_x86_64.whl"
                                             % (pipCmdScipion, tensorFlowTarget))
            keras=env.addPipModule('keras', '2.1.5', target='keras*', default=False, deps=[h5py])

        else:

            tensor = env.addPipModule('tensorflow', target='tensorflow*',
                                      default=False,
                                      pipCmd="%s https://storage.googleapis.com/"
                                             "tensorflow/linux/cpu/tensorflow-%s-cp27-none-"
                                             "linux_x86_64.whl"
                                             % (pipCmdScipion, tensorFlowTarget))

            keras = env.addPipModule('keras', '2.2.2', target='keras',
                                   default=False, deps=[cv2, h5py])


        deppLearnigTools = [scipy, cython, scikit_learn, unittest2, h5py,
                            keras, tensor]

        env.addPackage('deepLearnigToolkit', urlSuffix='external',
                       commands=[('echo "installed deepLearnig-Toolkit: %s"'
                                  % str([tool._name for tool in deppLearnigTools]),
                                  'deepLearnigToolkit')],
                       deps=deppLearnigTools)

        ## --- END OF DEEP LEARNING TOOLKIT --- ##


pyworkflow.em.Domain.registerPlugin(__name__)
