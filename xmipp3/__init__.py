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

import json
import subprocess
from datetime import datetime

from pyworkflow import Config
import pwem
import pyworkflow.utils as pwutils

from .base import *
from .constants import XMIPP_HOME, XMIPP_URL, XMIPP_DLTK_NAME

_logo = "xmipp_logo.png"
_references = ['delaRosaTrevin2013', 'Sorzano2013']
_currentVersion = '3.20.07'


class Plugin(pwem.Plugin):
    _homeVar = XMIPP_HOME
    _pathVars = [XMIPP_HOME]
    _supportedVersions = []
    _url = XMIPP_URL
    _condaRootPath = None

    @classmethod
    def _defineVariables(cls):
        cls._addVar(XMIPP_HOME, pwem.Config.XMIPP_HOME)

    @classmethod
    def getEnviron(cls, xmippFirst=True):
        """ Create the needed environment for Xmipp programs. """
        environ = pwutils.Environ(os.environ)
        pos = pwutils.Environ.BEGIN if xmippFirst else pwutils.Environ.END

        environ.update({
            'PATH': pwem.Config.CUDA_BIN,
            'LD_LIBRARY_PATH': pwem.Config.CUDA_LIB
        }, position=pwutils.Environ.END)

        if os.path.isfile(getXmippPath('xmippEnv.json')):
            with open(getXmippPath('xmippEnv.json'), 'r') as f:
                compilationEnv = json.load(f)
            environ.update(compilationEnv, position=pos)

        environ.update({
            'PATH': getXmippPath('bin'),
            'LD_LIBRARY_PATH': getXmippPath('lib'),
            'PYTHONPATH': getXmippPath('pylib')
                             }, position=pos)
        environ['XMIPP_HOME'] = getXmippPath()

        # Add path to python lib folder
        environ.addLibrary(Config.getPythonLibFolder())

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
        env = pwutils.Environ(os.environ)
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

        ## XMIPP SOFTWARE ##
        xmippDeps = []  # Deps should be at requirements.txt (old: scons, joblib, scikit_learn)

        # Installation vars for commands formating
        verToken = getXmippPath('v%s' % _currentVersion)
        confToken = getXmippPath("xmipp.conf")
        installVars = {'installedToken': "installation_finished",
                       'bindingsToken': "bindings_linked",
                       'verToken': verToken,
                       'nProcessors': env.getProcessors(),
                       'xmippHome': getXmippPath(),
                       'bindingsSrc': getXmippPath('bindings', 'python'),
                       'bindingsDst': Config.getBindingsFolder(),
                       'xmippLib': getXmippPath('lib', 'libXmipp.so'),
                       'coreLib': getXmippPath('lib', 'libXmippCore.so'),
                       'libsDst': Config.getLibFolder(),
                       'confToken': confToken,
                       'strPlaceHolder': '%s',  # to be replaced in the future
                       'currVersion': _currentVersion
                       }

        ## Installation commands (removing bindingsToken)
        installCmd = ("cd {cwd} && {configCmd} && {compileCmd} N={nProcessors:d} && "
                      "ln -srfn build {xmippHome} && cd - && "
                      "touch {installedToken} && rm {bindingsToken} 2> /dev/null")
        installTgt = [getXmippPath('bin', 'xmipp_reconstruct_significant'),
                      getXmippPath("lib/libXmippJNI.so"),
                      installVars['installedToken']]

        ## Linking bindings (removing installationToken)
        bindingsAndLibsCmd = ("find {bindingsSrc} -maxdepth 1 -mindepth 1 "
                              "! -name __pycache__ -exec ln -srfn {{}} {bindingsDst} \; && "
                              "ln -srfn {coreLib} {libsDst} && "
                              "touch {bindingsToken} && "
                              "rm {installedToken} 2> /dev/null")
        bindingsAndLibsTgt = [os.path.join(Config.getBindingsFolder(), 'xmipp_base.py'),
                              os.path.join(Config.getBindingsFolder(), 'xmippLib.so'),
                              os.path.join(Config.getLibFolder(), 'libXmipp.so'),
                              installVars['bindingsToken']]

        sourceTgt = [getXmippPath('xmipp.bashrc')]  # Target for xmippSrc and xmippDev
        ## Allowing xmippDev if devel mode detected
        # plugin  = scipion-em-xmipp  <--  xmipp3    <--     __init__.py
        pluginDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # bundle  = xmipp-bundle  <-  src  <-  scipion-em-xmipp
        bundleDir = os.path.dirname(os.path.dirname(pluginDir))

        isPypiDev = os.path.isfile(os.path.join(pluginDir, 'setup.py'))
        isXmippBu = (os.path.isdir(os.path.join(bundleDir, 'src')) and
                     os.path.isfile(os.path.join(bundleDir, 'xmipp')))
        develMode = isPypiDev and isXmippBu
        if develMode:
            env.addPackage('xmippDev', tar='void.tgz',
                           commands=[(installCmd.format(**installVars,
                                                        cwd=bundleDir,
                                                        configCmd='pwd',
                                                        compileCmd='./xmipp all'),
                                      installTgt+sourceTgt),
                                     (bindingsAndLibsCmd.format(**installVars),
                                      bindingsAndLibsTgt)],
                           deps=xmippDeps, default=False)

        avoidConfig = os.environ.get('XMIPP_NOCONFIG', 'False') == 'True'
        configSrc = ('./xmipp check_config' if avoidConfig
                     else './xmipp config noAsk && ./xmipp check_config')
        env.addPackage('xmippSrc', version=_currentVersion,
                       # adding 'v' before version to fix a package target (post-link)
                       tar='xmippSrc-v'+_currentVersion+'.tgz',
                       commands=[(installCmd.format(**installVars, cwd='.',
                                                    configCmd=configSrc,
                                                    compileCmd='./xmipp compileAndInstall'),
                                  installTgt + sourceTgt),
                                 (bindingsAndLibsCmd.format(**installVars),
                                  bindingsAndLibsTgt)],
                       deps=xmippDeps, default=not develMode)

        ## EXTRA PACKAGES ##
        installDeepLearningToolkit(cls, env)


def installDeepLearningToolkit(plugin, env):

    preMsgs = []
    cudaMsgs = []
    nvidiaDriverVer = None
    if os.environ.get('CUDA', 'True') == 'True':
        try:
            nvidiaDriverVer = subprocess.Popen(["nvidia-smi",
                                                "--query-gpu=driver_version",
                                                "--format=csv,noheader"],
                                               env=plugin.getEnviron(),
                                               stdout=subprocess.PIPE
                                               ).stdout.read().decode('utf-8').split(".")[0]
            if int(nvidiaDriverVer) < 390:
                preMsgs.append("Incompatible driver %s" % nvidiaDriverVer)
                cudaMsgs.append("Your NVIDIA drivers are too old (<390). "
                                "Tensorflow was installed without GPU support. "
                                "Just CPU computations enabled (slow computations).")
                nvidiaDriverVer = None
        except Exception as e:
            preMsgs.append(str(e))

    if nvidiaDriverVer is not None:
        preMsgs.append("CUDA support find. Driver version: %s" % nvidiaDriverVer)
        msg = "Tensorflow installed with CUDA SUPPORT."
        cudaMsgs.append(msg)
        useGpu = True
    else:
        preMsgs.append("CUDA will NOT be USED. (not found or incompatible)")
        msg = ("Tensorflow installed without GPU. Just CPU computations "
               "enabled (slow computations). To enable CUDA (drivers>390 needed), "
               "set CUDA=True in 'scipion.conf' file")
        cudaMsgs.append(msg)
        useGpu = False

    # commands  = [(command, target), (cmd, tgt), ...]
    cmdsInstall = [(cmd, envName + ".yml") for cmd, envName in
                   CondaEnvManager.yieldInstallAllCmds(useGpu=useGpu)]

    now = datetime.now()
    installDLvars = {'modelsUrl': "http://scipion.cnb.csic.es/downloads/scipion/software/em",
                     'syncBin': plugin.getHome('bin/xmipp_sync_data'),
                     'modelsDir': plugin.getHome('models'),
                     'modelsPrefix': "models_UPDATED_on",
                     'xmippLibToken': 'xmippLibToken',
                     'libXmipp': plugin.getHome('lib/libXmipp.so'),
                     'preMsgsStr': ' ; '.join(preMsgs),
                     'afterMsgs': "\n > ".join(cudaMsgs)}

    installDLvars.update({'modelsTarget': "%s_%s_%s_%s"
                                          % (installDLvars['modelsPrefix'],
                                             now.day, now.month, now.year)})

    modelsDownloadCmd = ("rm %(modelsPrefix)s_* %(xmippLibToken)s 2>/dev/null ; "
                         "echo 'Downloading pre-trained models...' ; "
                         "%(syncBin)s update %(modelsDir)s %(modelsUrl)s DLmodels && "
                         "touch %(modelsTarget)s && echo ' > %(afterMsgs)s'"
                         % installDLvars,                # End of command
                         installDLvars['modelsTarget'])  # Target

    xmippInstallCheck = ("if ls %(libXmipp)s > /dev/null ; "
                         "then touch %(xmippLibToken)s; echo ' > %(preMsgsStr)s' ; "
                         "else echo ; echo ' > Xmipp installation not found, "
                         "please install it first (xmippSrc or xmippBin*).';echo;"
                         " fi" % installDLvars,           # End of command
                         installDLvars['xmippLibToken'])  # Target

    env.addPackage(XMIPP_DLTK_NAME, version='0.2', urlSuffix='external',
                   commands=[xmippInstallCheck]+cmdsInstall+[modelsDownloadCmd],
                   deps=[], tar=XMIPP_DLTK_NAME+'.tgz')
