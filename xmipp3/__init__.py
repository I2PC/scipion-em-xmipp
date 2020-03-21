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

from pyworkflow import Config
import pwem
import pyworkflow.utils as pwutils

from .base import *
from .constants import XMIPP_HOME, XMIPP_URL
from xmipp3.condaEnvManager import CondaEnvManager

_logo = "xmipp_logo.png"
_references = ['delaRosaTrevin2013', 'Sorzano2013']
_currentVersion = '3.19.04'


class Plugin(pwem.Plugin):
    _homeVar = XMIPP_HOME
    _pathVars = [XMIPP_HOME]
    _supportedVersions = []
    _url = XMIPP_URL
    _condaRootPath = None

    @classmethod
    def _defineVariables(cls):
        cls._addVar(XMIPP_HOME, pwem.Config.XMIPP_HOME)
        # cls._defineEmVar(CONDA_DEFAULT_ENVIRON, "conda_default_env")

    @classmethod
    def getEnviron(cls, xmippFirst=True):
        """ Create the needed environment for Xmipp programs. """
        environ = pwutils.Environ(os.environ)
        pos = pwutils.Environ.BEGIN if xmippFirst else pwutils.Environ.END
        environ.update({
            'PATH': getXmippPath('bin'),
            'LD_LIBRARY_PATH': getXmippPath('lib'),
            'PYTHONPATH': getXmippPath('pylib')+":" +  # FIXME: Only pylib should be enough
                          getXmippPath(os.path.join('pylib', 'xmippPyModules'))
                             }, position=pos)
        environ['XMIPP_HOME'] = getXmippPath()

        # Add path to python lib folder
        environ.addLibrary(Config.getPythonLibFolder())

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
    def getCondaPathInEnv(cls, condaEnv, condaSubDir):
      if cls._condaRootPath is None: #This is for performance reasons.
        cls._condaRootPath= CondaEnvManager.getCoondaRoot( cls.getEnviron() )
      condaRoot = cls._condaRootPath
      return CondaEnvManager.getCondaPathInEnv(condaRoot, condaEnv, condaSubDir)

    @classmethod
    def getMatlabEnviron(cls, *toolPaths):
        """ Return an Environment prepared for launching Matlab
        scripts using the Xmipp binding.
        """
        env = pwem.Plugin.getEnviron()
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
    def defineBinaries(cls, env):
        """ Define the Xmipp binaries/source available tgz.
            In addition, define extra software needed by some Xmipp methods
            such as deepLearningToolkit.
            Scipion-defined software can be used as dependencies
            by using its name as string.
        """

        # scons = tryAddPipModule(env, 'scons', '3.0.4')
        # joblib = tryAddPipModule(env, 'joblib', '0.11', target='joblib*')
        #
        # # scikit
        # scipy = tryAddPipModule(env, 'scipy', '1.4.1', default=True)
        # cython = tryAddPipModule(env, 'cython', '0.29.14', target='Cython-0.29*',
        #                          default=True)
        # scikit_learn = tryAddPipModule(env, 'scikit-learn', '0.22',
        #                                target='scikit_learn*',
        #                                default=True)

        ## XMIPP SOFTWARE ##
        xmippDeps = []  # scons, joblib, scikit_learn
        installedToken = cls.getHome("installation_finished")
        bindingsToken = cls.getHome("bindings_linked")
        verToken = cls.getHome('v%s' % _currentVersion)
        sourceTgt = [cls.getHome('xmipp.bashrc')]

        installCmd = ("./xmipp all N=%d dir=%s && touch %s && rm %s 2> /dev/null"
                      % (env.getProcessors(), cls.getHome(),
                         installedToken, bindingsToken))
        installTgt = [cls.getHome('bin', 'xmipp_reconstruct_significant'),
                      cls.getHome("lib/libXmippJNI.so"), installedToken]

        bindingsAndLibsCmd = ("ln -fs %s %s && ln -fs %s %s && ln -fs %s %s && "
                              "touch %s && rm %s 2> /dev/null"
                              % (cls.getHome('bindings', 'python', '*'),
                                 Config.getBindingsFolder(),
                                 cls.getHome('lib', 'libXmipp.so'),
                                 Config.getLibFolder(),
                                 cls.getHome('lib', 'libXmippCore.so'),
                                 Config.getLibFolder(),
                                 bindingsToken, installedToken))
        bindingsAndLibsTgt = ([os.path.join(Config.getBindingsFolder(), 'xmipp_base.py'),
                               os.path.join(Config.getBindingsFolder(), 'xmippLib.so'),
                               os.path.join(Config.getLibFolder(), 'libXmipp.so'),
                               bindingsToken])

        # plugin  = scipion-em-xmipp  <--  xmipp3    <--     __init__.py
        pluginDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # bundle  = xmipp-bundle  <-  src  <-  scipion-em-xmipp
        bundleDir = os.path.dirname(os.path.dirname(pluginDir))

        isPypiDev = os.path.isfile(os.path.join(pluginDir, 'setup.py'))
        isXmippBu = (os.path.isdir(os.path.join(bundleDir, 'src')) and
                     os.path.isfile(os.path.join(bundleDir, 'xmipp')))
        develMode = isPypiDev and isXmippBu
        if develMode:
            cdCmd = 'cd %s && ' % bundleDir
            env.addPackage('xmippDev', tar='void.tgz',
                           commands=[(cdCmd+installCmd, installTgt+sourceTgt),
                                     (bindingsAndLibsCmd, bindingsAndLibsTgt)],
                           deps=xmippDeps, default=False)

        sourceTgt.append(verToken)
        env.addPackage('xmippSrc', version=_currentVersion,
                       # adding 'v' before version to fix a package target (post-link)
                       tar='xmippSrc-v'+_currentVersion+'.tgz',
                       commands=[(installCmd, installTgt + sourceTgt),
                                 (bindingsAndLibsCmd, bindingsAndLibsTgt)],
                       deps=xmippDeps, default=not develMode)

        confToken = cls.getHome("xmipp.conf")
        installBin = ("rm -rf "+cls.getHome()+" 2>/dev/null; cd .. ; "
                      "ln -sf xmippBin_%s-"+_currentVersion+" "+cls.getHome()+
                      " && touch "+installedToken)
        env.addPackage('xmippBin_Debian', version=_currentVersion,
                       commands=[(installBin % 'Debian',
                                  installTgt + [confToken, verToken+'_Debian']),
                                 (bindingsAndLibsCmd, bindingsAndLibsTgt)],
                       deps=xmippDeps, default=False)

        env.addPackage('xmippBin_Centos', version=_currentVersion,
                       commands=[(installBin % 'Centos',
                                  installTgt+[confToken, verToken+'_Centos']),
                                 (bindingsAndLibsCmd, bindingsAndLibsTgt)],
                       deps=xmippDeps, default=False)

        ## EXTRA PACKAGES ##
        installDeepLearningToolkit(cls, env)


# def tryAddPipModule(env, moduleName, *args, **kwargs):
#     """ To try to add certain pipModule.
#         If it fails due to it is already add by other plugin or Scipion,
#           just returns its name to use it as a dependency.
#         Raise the exception if unknown error is gotten.
#     """
#     try:
#         return env.addPipModule(moduleName, *args, **kwargs)._name
#     except Exception as e:
#         if str(e) == "Duplicated target '%s'" % moduleName:
#             return moduleName
#         else:
#             raise Exception(e)
#
# def installDeepLearningToolkit(plugin, env):
#     deepLearningTools = []
#
#     scikit_image = tryAddPipModule(env, 'scikit-image', '0.14.2',
#                                    target='scikit_image*',
#                                    default=False, deps=['scipy', 'scikit-learn'])
#     deepLearningTools.append(scikit_image)
#
#     # pandas
#     pandas = tryAddPipModule(env, 'pandas', '0.20.1',
#                                    target='pandas*',
#                                    default=False, deps=['scipy'])
#     deepLearningTools.append(pandas)
#
#     # Keras deps
#     unittest2 = tryAddPipModule(env, 'unittest2', '1.1.0', target='unittest2*',
#                                 default=False)
#     h5py = tryAddPipModule(env, 'h5py', '2.8.0rc1', target='h5py*',
#                            default=False, deps=[unittest2])
#     cv2 = tryAddPipModule(env, 'opencv-python', "3.4.2.17",
#                           target="cv2", default=False)
#
#     # TensorFlow defs
#     tensorFlowTarget = "1.10.0"  # cuda 9
#     pipCmdScipion = '%s %s/pip install' % (env.getBin('python'),
#                                            env.getPythonPackagesFolder())
#
#     cudNNwarning = []
#     cudNNversion = None
#     if os.environ.get('CUDA', 'True') == 'True':
#         try:
#             nvccVersion = subprocess.Popen(["nvcc", '--version'],
#                                            env=plugin.getEnviron(),
#                                            stdout=subprocess.PIPE).stdout.read()
#         except:
#             nvccVersion = 'None'  # string to avoid 'NoneType is not iterable' error
#
#         nvccVersion = nvccVersion.decode("utf8")
#         if "release 8.0" in nvccVersion:  # cuda 8
#             tensorFlowTarget = "1.4.1"
#             cudNNversion = "v6-cuda8"
#         elif "release 9.0" in nvccVersion:  # cuda 9
#             tensorFlowTarget = "1.10.0"
#             cudNNversion = "v7.0.1-cuda9"
#         else:
#             cudNNwarning.append("cudNN requires CUDA 8.0 or CUDA 9.0 "
#                                 "(8.0 recommended)")
#
#     if cudNNversion is not None:
#         cudNN = tryAddPipModule(env, 'cudnnenv', version='0.6.6',
#                                 target="cudnnenv", default=False)
#         deepLearningTools.append(cudNN)
#
#         tensor = tryAddPipModule(env, 'tensorflow-gpu', target='tensorflow*',
#                                  default=False,
#                                  pipCmd="%s https://storage.googleapis.com/"
#                                         "tensorflow/linux/gpu/"
#                                         "tensorflow_gpu-%s-cp27-none-"
#                                         "linux_x86_64.whl"
#                                         % (pipCmdScipion, tensorFlowTarget))
#         deepLearningTools.append(tensor)
#
#         keras = tryAddPipModule(env, 'keras', '2.2.2', target='keras*',
#                                 default=False, deps=[cv2, h5py])
#
#         deepLearningTools.append(keras)
#         cudnnInstallCmd = ("cudnnenv install %s ; "
#                            "cp -r $HOME/.cudnn/active/cuda/lib64/* %s"
#                             % (cudNNversion, getXmippPath('lib')),
#                            getXmippPath('lib', 'libcudnn.so'))
#     else:
#         cudNNwarning.append("Installing tensorflow without GPU "
#                             "support. Just CPU computations enabled "
#                             "(only predictions recommended).")
#         warnStr = ' > WARNING: '
#         warnSep = '\n'+' '*len(warnStr)
#         cudnnInstallCmd = ("echo '\n%s%s\n'" % (warnStr, warnSep.join(cudNNwarning)),
#                            "")
#         tensor = tryAddPipModule(env, 'tensorflow', target='tensorflow*',
#                                  default=False,
#                                  pipCmd="%s https://storage.googleapis.com/"
#                                         "tensorflow/linux/cpu/"
#                                         "tensorflow-%s-cp27-none-"
#                                         "linux_x86_64.whl"
#                                         % (pipCmdScipion, tensorFlowTarget))
#         deepLearningTools.append(tensor)
#
#         keras = tryAddPipModule(env, 'keras', '2.2.2', target='keras',
#                                 default=False, deps=[cv2, h5py])
#         deepLearningTools.append(keras)
#
#     # pre-trained models
#     url = "http://scipion.cnb.csic.es/downloads/scipion/software/em"
#     modelsDownloadCmd = ("echo 'Downloading pre-trained models...' ; "
#                          "%s update %s %s DLmodels"
#                          % (plugin.getHome('bin/xmipp_sync_data'),
#                             plugin.getHome('models'), url))
#     now = datetime.now()
#     modelsPrefix = "models_UPDATED_on"
#     modelsTarget = "%s_%s_%s_%s" % (modelsPrefix, now.day, now.month, now.year)
#     deepLearningToolsStr = [str(tool) for tool in deepLearningTools]
#     target = "installed_%s" % '_'.join(deepLearningToolsStr)
#     xmippInstallCheck = ("if ls %s > /dev/null ; then touch xmippLibToken;"
#                          "else echo ; echo ' > Xmipp installation not found, "
#                          "please install it first (xmippSrc or xmippBin*).';echo;"
#                          " fi" % plugin.getHome('lib'), 'xmippLibToken')
#     env.addPackage('deepLearningToolkit', version='0.1', urlSuffix='external',
#                    commands=[xmippInstallCheck, cudnnInstallCmd,
#                              ("rm %s_* 2>/dev/null ; %s && touch %s"
#                               % (modelsPrefix, modelsDownloadCmd, modelsTarget),
#                               modelsTarget),
#                              ("echo ; echo ' > DeepLearning-Toolkit installed: %s' ; "
#                               "echo ; touch %s" % (', '.join(deepLearningToolsStr),
#                                                    target),
#                               target)],
#                    deps=deepLearningTools, tar='deepLearningToolkit.tgz')


def installDeepLearningToolkit(plugin, env):

  preMsgs = []
  cudaMsgs = []
  if os.environ.get('CUDA', 'True') == 'True':
    try:
      nvidiaDriversVersion = subprocess.Popen(["nvidia-smi", '--query-gpu=driver_version', "--format=csv,noheader"],
                                     env=plugin.getEnviron(),
                                     stdout=subprocess.PIPE).stdout.read().decode('utf-8').split(".")[0]
      if int(nvidiaDriversVersion)< 390:
        cudaMsgs.append("Your NVIDIA drivers are too old (<390). Tensorflow was installed without GPU "
                            "support. Just CPU computations enabled (slow computations)")
        nvidiaDriversVersion = None
    except Exception as e:
      preMsgs.append(str(e))
      nvidiaDriversVersion = None

    cmdsInstall=[]
  if nvidiaDriversVersion is not None:
    msg="Tensorflow installed with CUDA SUPPORT"
    cudaMsgs.append(msg)
    useGpu=True
  else:
    preMsgs.append("CUDA will NOT be USED")
    msg=("Tensorflow was installed without GPU. Just CPU computations enabled (slow computations)"
         "To enable CUDA, set CUDA=True in xmipp.conf and scipion.conf files")
    cudaMsgs.append(msg)
    useGpu=False

  for cmd, environName in CondaEnvManager.yieldInstallAllCmds(useGpu=useGpu):
    cmdsInstall.append((cmd, environName))

  url = "http://scipion.cnb.csic.es/downloads/scipion/software/em"
  modelsDownloadCmd = ("echo 'Downloading pre-trained models...' ; "
                       "%s update %s %s DLmodels"
                       % (plugin.getHome('bin/xmipp_sync_data'),
                          plugin.getHome('models'), url))

  now = datetime.now()
  modelsPrefix = "models_UPDATED_on"
  modelsTarget = "%s_%s_%s_%s" % (modelsPrefix, now.day, now.month, now.year)

  xmippLibToken = 'xmippLibToken'
  xmippInstallCheck = ("if ls %s > /dev/null ; then touch %s; echo ' > %s' ; "
                       "else echo ; echo ' > Xmipp installation not found, "
                       "please install it first (xmippSrc or xmippBin*).';echo;"
                       " fi" % (plugin.getHome('lib/libXmipp.so'), xmippLibToken,
                                ' ; '.join(preMsgs)), xmippLibToken)

  cmdsInstall = [(cmd, envName+".yml") for cmd, envName in cmdsInstall]
  env.addPackage('deepLearningToolkit', version='0.2', urlSuffix='external',
                 commands=[xmippInstallCheck]+cmdsInstall+[
                           ("rm %s_* 2>/dev/null ; %s && touch %s && echo ' > %s'"
                            % (modelsPrefix, modelsDownloadCmd, modelsTarget, "\n > ".join(cudaMsgs)),
                            modelsTarget),
                 ],
                 deps=[], tar='deepLearningToolkit.tgz')
