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
from datetime import datetime

import pyworkflow.em
import pyworkflow.utils as pwutils

from .base import *
from .constants import XMIPP_HOME


_logo = "xmipp_logo.png"
_references = ['delaRosaTrevin2013', 'Sorzano2013']
_currentVersion = '3.19.02'

class Plugin(pyworkflow.em.Plugin):
    _homeVar = XMIPP_HOME
    _pathVars = [XMIPP_HOME]
    _supportedVersions = []

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(XMIPP_HOME, 'xmipp')
        cls._defineEmVar(NMA_HOME, 'nma')

    @classmethod
    def getEnviron(cls, xmippFirst=True):
        """ Create the needed environment for Xmipp programs. """
        environ = pwutils.Environ(os.environ)
        pos = pwutils.Environ.BEGIN if xmippFirst else pwutils.Environ.END
        environ.update({
            'PATH': getXmippPath('bin'),
            'LD_LIBRARY_PATH': getXmippPath('lib'),
            'PYTHONPATH': getXmippPath('pylib')
        }, position=pos)

        # environ variables are strings not booleans
        if os.environ.get('CUDA', 'False') != 'False':
            environ.update({
                'PATH': os.environ.get('CUDA_BIN', ''),
                'LD_LIBRARY_PATH': os.environ.get('NVCC_LIBDIR', '')
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
        env.set('PATH', os.environ.get('MATLAB_BINDIR', ''), pwutils.Environ.BEGIN)
        env.set('LD_LIBRARY_PATH', os.environ.get('MATLAB_LIBDIR', ''),
                pwutils.Environ.BEGIN)
        for toolpath in toolPaths:
            env.set('MATLABPATH', toolpath, pwutils.Environ.BEGIN)
        env.set('MATLABPATH', os.path.join(os.environ[XMIPP_HOME],
                                           'libraries', 'bindings', 'matlab'),
                pwutils.Environ.BEGIN)

        return env

    @classmethod
    def getModel(self, *modelPath):
        """ Returns the path to the models folder followed by
            the given relative path.
        .../xmipp/models/myModel/myFile.h5 <= getModel('myModel', 'myFile.h5')
        """
        model = getXmippPath('models', *modelPath)
        if not os.path.exists(model):
            raise Exception("'%s' model not found. Please, run: \n"
                            " > scipion installb deepLearnigToolkit" % modelPath[0])
        return model

    @classmethod
    def defineBinaries(cls, env):
        """ Define the Xmipp binaries/source available tgz.
            In addition, define extra software needed by some Xmipp methods
            such as deepLearningToolkit.
            Scipion-defined software can be used as dependencies
            by using its name as string.
        """

        ## XMIPP SOFTWARE ##
        installCmd = ("src/xmipp/xmipp config ; src/xmipp/xmipp check_config ;"
                      "src/xmipp/xmipp compile %d ; src/xmipp/xmipp install %s"
                       % (env.getProcessors(), cls.getHome()))

        target = "%s/bin/xmipp_reconstruct_significant" % cls.getHome()

        env.addPackage('xmippSrc', version=_currentVersion,
                       tar='xmippSrc-%s.tgz' % _currentVersion,
                       commands=[("rm -rf %s 2>/dev/null; %s" 
                                  % (cls.getHome(), installCmd),
                                  [target, cls.getHome('xmipp.bashrc')])],
                       default=True)

        env.addPackage('xmippBin', version=_currentVersion,
                       tar='xmippBin-%s.tgz' % _currentVersion,
                       commands=[("rm -rf %s 2>/dev/null; cd .. ; ln -sf xmippBin-%s %s"
                                  % (cls.getHome(), _currentVersion, cls.getHome()),
                                  [target, cls.getHome("xmipp.conf")])],
                       default=False)

        ## EXTRA PACKAGES ##
        # joblib
        tryAddPipModule(env, 'joblib', '0.11', target='joblib*')

        installDeepLearningToolkit(cls, env)

        # NMA
        env.addPackage('nma', tar='nma.tgz', default=False, deps=['arpack'],
                       commands=[('cd ElNemo; make; mv nma_* ..',
                                  'nma_elnemo_pdbmat'),
                                 ('cd NMA_cart; LDFLAGS=-L%s make; mv nma_* ..'
                                  % env.getLibFolder(), 'nma_diag_arpack')])

        # sh_alignment
        # FIXME: Is this needed when we have it in xmipp/external/sh_alignment ??
        env.addLibrary(
            'sh_alignment',
            tar='sh_alignment.tgz',
            commands=[('cd software/tmp/sh_alignment; make install',
                       'software/lib/python2.7/site-packages/sh_alignment/frm.py')],
            default=False)  # FIXME: I set this to False because is not compiling...


def tryAddPipModule(env, moduleName, *args, **kwargs):
    """ To try to add certain pipModule.
        If it fails due to it is already add by other plugin or Scipion,
          just returns its name to use it as a dependency.
        Raise the exception if unknown error is gotten.
    """
    try:
        return env.addPipModule(moduleName, *args, **kwargs)._name
    except Exception as e:
        if "Duplicated target '%s'" % moduleName == str(e):
            return moduleName
        else:
            raise Exception(e)

def installDeepLearningToolkit(plugin, env):
    deepLearningTools = []

    # scikit
    scipy = tryAddPipModule(env, 'scipy', '0.14.0', default=False,
                            deps=['lapack', 'matplotlib'])
    cython = tryAddPipModule(env, 'cython', '0.22', target='Cython-0.22*',
                             default=False)
    scikit_learn = tryAddPipModule(env, 'scikit-learn', '0.19.1',
                                   target='scikit_learn*',
                                   default=False, deps=[scipy, cython])
    deepLearningTools.append(scikit_learn)

    # Keras deps
    unittest2 = tryAddPipModule(env, 'unittest2', '0.5.1', target='unittest2*',
                                default=False)
    h5py = tryAddPipModule(env, 'h5py', '2.8.0rc1', target='h5py*',
                           default=False, deps=[unittest2])
    cv2 = tryAddPipModule(env, 'opencv-python', "3.4.2.17",
                          target="cv2", default=False)

    # TensorFlow defs
    tensorFlowTarget = "1.10.0"  # cuda 9
    nvccProgram = subprocess.Popen(["which", "nvcc"], env=plugin.getEnviron(),
                                   stdout=subprocess.PIPE).stdout.read()
    pipCmdScipion = '%s %s/pip install' % (env.getBin('python'),
                                           env.getPythonPackagesFolder())


    cudNN_version = None
    if nvccProgram != "":
        nvccVersion = subprocess.Popen(["nvcc", '--version'], env=plugin.getEnviron(),
                                       stdout=subprocess.PIPE).stdout.read()

        if "release 8.0" in nvccVersion:  # cuda 8
            tensorFlowTarget = "1.4.1"
            cudNN_version = "v6-cuda8"
        elif "release 9.0" in nvccVersion:  # cuda 9
            tensorFlowTarget = "1.10.0"
            cudNN_version = "v7.0.1-cuda9"
        else:
            print("Error, tensorflow requires CUDA 8.0 or CUDA 9.0")

    if cudNN_version is not None:
        cudNN_installer = tryAddPipModule(env, 'cudnnenv', target="cudnnenv",
                                          pipCmd="%s git+https://github.com/"
                                                 "unnonouno/cudnnenv.git"
                                                 % pipCmdScipion,
                                          default=False)
        deepLearningTools.append(cudNN_installer)

        tensor = tryAddPipModule(env, 'tensorflow-gpu', target='tensorflow*',
                                 default=False,
                                 pipCmd="%s https://storage.googleapis.com/"
                                        "tensorflow/linux/gpu/"
                                        "tensorflow_gpu-%s-cp27-none-"
                                        "linux_x86_64.whl"
                                        % (pipCmdScipion, tensorFlowTarget))
        deepLearningTools.append(tensor)

        keras = tryAddPipModule(env, 'keras', '2.1.5', target='keras*',
                                default=False, deps=[cv2, h5py])
        deepLearningTools.append(keras)
        cudnnInstallCmd = ("cudnnenv install %s; "
                           "cp -r $HOME/.cudnn/active/cuda/lib64/* %s"
                            % (cudNN_version, getXmippPath('lib')),
                           getXmippPath('lib', 'libcudnn.so'))
    else:
        cudnnInstallCmd = ("", "")
        print("WARNING: Installing tensorflow without GPU support. "
              "Just CPU computations enabled (only predictions recommended).")
        tensor = tryAddPipModule(env, 'tensorflow', target='tensorflow*',
                                 default=False,
                                 pipCmd="%s https://storage.googleapis.com/"
                                        "tensorflow/linux/cpu/"
                                        "tensorflow-%s-cp27-none-"
                                        "linux_x86_64.whl"
                                        % (pipCmdScipion, tensorFlowTarget))
        deepLearningTools.append(tensor)

        keras = tryAddPipModule(env, 'keras', '2.2.2', target='keras',
                                default=False, deps=[cv2, h5py])
        deepLearningTools.append(keras)

    # pre-trained models
    url = "http://scipion.cnb.csic.es/downloads/scipion/software/em"
    modelsDownloadCmd = ("%s update %s %s DLmodels"
                         % (plugin.getHome('bin/xmipp_sync_data'),
                            plugin.getHome('models'), url))
    now = datetime.now()
    modelsPrefix = "models_UPDATED_on"
    modelsTarget = "%s_%s_%s_%s" % (modelsPrefix, now.day, now.month, now.year)
    deepLearningToolsStr = [str(tool) for tool in deepLearningTools]
    target = "installed_%s" % '_'.join(deepLearningToolsStr)
    env.addPackage('deepLearningToolkit', urlSuffix='external',
                   commands=[cudnnInstallCmd,
                             ("rm %s_* 2>/dev/null; %s ; touch %s" 
                              % (modelsPrefix, modelsDownloadCmd, modelsTarget), 
                              modelsTarget),
                             ("echo ; echo ' > DeepLearnig-Toolkit installed: %s' ; "
                              "echo ; touch %s" % (', '.join(deepLearningToolsStr),
                                                   target),
                              target)],
                   deps=deepLearningTools)

pyworkflow.em.Domain.registerPlugin(__name__)
