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
import re
from datetime import datetime
import pwem
from pyworkflow import Config
import pyworkflow.utils as pwutils
from scipion.install.funcs import CondaCommandDef
from .base import *
from .version import *
from .constants import XMIPP_HOME, XMIPP_URL, XMIPP_DLTK_NAME, XMIPP_CUDA_BIN, XMIPP_CUDA_LIB, XMIPP_GIT_URL


_references = ['delaRosaTrevin2013', 'Sorzano2013', 'Strelak2021']
_currentDepVersion = '1.0'
# Requirement version variables
NVIDIA_DRIVERS_MINIMUM_VERSION = 450

type_of_version = version.type_of_version
_logo = version._logo
_currentDepVersion = version._currentDepVersion
__version__ = version.__version__


class Plugin(pwem.Plugin):
    _homeVar = XMIPP_HOME
    _pathVars = [XMIPP_HOME]
    _supportedVersions = []
    _url = XMIPP_URL
    _condaRootPath = None
    # Refressing the plugin
    # Inspects the call stack to find 'installPipModule' and sets 'self._plugin' to None.
    # Uses CPython's internal API 'PyFrame_LocalsToFast' to sync changes.
    # Risky and CPython-specific; use only if redesign is not possible.
    try:
        import inspect
        import ctypes
        for frameInfo in inspect.stack():
            if frameInfo.function == "installPipModule":
                frame = frameInfo[0]
                frame.f_locals['self']._plugin = None
                ctypes.pythonapi.PyFrame_LocalsToFast(
				    ctypes.py_object(frame),
				    ctypes.c_int(1))
    except Exception as e:
        print(e)
    
    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(XMIPP_HOME, pwem.Config.XMIPP_HOME)
        cls._defineVar(XMIPP_CUDA_BIN, pwem.Config.CUDA_BIN)
        cls._defineVar(XMIPP_CUDA_LIB, pwem.Config.CUDA_LIB)

    @classmethod
    def getEnviron(cls, xmippFirst=True):
        """ Create the needed environment for Xmipp programs. """
        environ = pwutils.Environ(os.environ)
        pos = pwutils.Environ.BEGIN if xmippFirst else pwutils.Environ.END

        environ.update({
            'PATH': cls.getVar(XMIPP_CUDA_BIN),
            'LD_LIBRARY_PATH': cls.getVar(XMIPP_CUDA_LIB)
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
        
        # Determine if we are on a development 
        bundleDir = cls.__getBundleDirectory()
        develMode = bundleDir is not None
        
        COMPILE_TARGETS = [
            'dist/bin/xmipp_image_header', 
            'dist/xmipp.bashrc'
        ]
        
        # When changing dependencies, increment _currentDepVersion
        CONDA_DEPENDENCIES = [
	    "'cmake>=3.18,<4'", #cmake-4 is not compatible with Relion compilation
            "hdf5>=1.18",
            "sqlite>=3",
            "fftw>=3",
            "make",
            "zlib",
            "openjdk",
            "'libtiff<=4.5.1'",
            "libjpeg-turbo",
        ]

        CONDA_LIBSTDCXX_PACKAGE = "libstdcxx-ng"

        if Config.isCondaInstallation():
            condaEnvPath = os.environ['CONDA_PREFIX']
            scipionEnv=os.path.basename(condaEnvPath)  # TODO: use pyworkflow method when released: Config.getEnvName()

            commands = CondaCommandDef(scipionEnv, cls.getCondaActivationCmd())
            #commands.condaInstall('-c conda-forge --yes '  + ' '.join(CONDA_DEPENDENCIES))
            commands.condaInstall('-c conda-forge --yes '  + CONDA_LIBSTDCXX_PACKAGE)
            commands.touch("deps_installed.txt")
            env.addPackage(
                'xmippDep', version=_currentDepVersion,
                tar='void.tgz',
                commands=commands.getCommands(),
                neededProgs=['conda'],
                default=True
            )
        
        if develMode:
            xmippSrc = 'xmippDev'
            installCommands = [
		        (f'cd {pwem.Config.EM_ROOT} && rm -rf {xmippSrc} && '
		         f'git clone {XMIPP_GIT_URL} {xmippSrc} && '
		         f'cd {xmippSrc} && '
		         #f'git checkout {branchTest} && '
		         f'./xmipp ', COMPILE_TARGETS)
	        ]

            env.addPackage(
	            'xmippDev',
	            tar='void.tgz',
	            commands=installCommands,
	            neededProgs=['git', 'gcc', 'g++', 'cmake', 'make'],
	            updateCuda=True,
	            default=False
	        )
        else:
            xmippSrc = f'xmipp3-{version._binVersion}'
            installCommands = [
                (f'cd .. && rm -rf {xmippSrc} && '
                f'git clone {XMIPP_GIT_URL} -b {version._binVersion} {xmippSrc} && '
                f'cd {xmippSrc} && '
                f'./xmipp', COMPILE_TARGETS)
            ]
            env.addPackage(
	                'xmipp3', version=version._binVersion,
	                tar='void.tgz',
	                commands=installCommands,
	                neededProgs=['git', 'gcc', 'g++', 'cmake', 'make'],
	                updateCuda=True,
	                default=not develMode
	            )

        ## EXTRA PACKAGES ##
        installDeepLearningToolkit(cls, env)

    @classmethod
    def __getBundleDirectory(cls):
        # plugin  = scipion-em-xmipp  <--  xmipp3    <--     __init__.py
        pluginDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # bundle  = xmipp-bundle  <-  src   <-  scipion-em-xmipp
        bundleDir = os.path.dirname(os.path.dirname(pluginDir))

        isBundle = (os.path.isdir(os.path.join(bundleDir, 'src')) and
                    os.path.isfile(os.path.join(bundleDir, 'xmipp')))
        print(f'pluginDir: {pluginDir}\n, bundleDir: {bundleDir}\n, isBundle: {isBundle}')
        return bundleDir if isBundle else None

def getNvidiaDriverVersion(plugin):
    """Attempt to retrieve the NVIDIA driver version using different methods.
    Only returns if a valid version string is found.
    """
    commands = [
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        ["cat", "/sys/module/nvidia/version"]
    ]

    for cmd in commands:
        try:
            output = subprocess.Popen(
                cmd,
                env=plugin.getEnviron(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            ).communicate()[0].decode('utf-8').strip()

            # Check if the output matches a version pattern like 530.30 or 470
            match = re.match(r'^(\d+)', output)
            if match:
                return match.group(1)  # Return just the major version (e.g., "530")

        except (ValueError, TypeError, FileNotFoundError, subprocess.SubprocessError):
            continue  # Try next method

    return None  # No valid version found

def installDeepLearningToolkit(plugin, env):

    preMsgs = []
    cudaMsgs = []
    nvidiaDriverVer = None
    if os.environ.get('CUDA', 'True') == 'True':
        nvidiaDriverVer = getNvidiaDriverVersion(plugin)

    if nvidiaDriverVer is None:
        preMsgs.append("Not nvidia driver found. Type: "
                       " nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        preMsgs.append(
            "CUDA will NOT be USED. (not found or incompatible)")
        msg = ("Tensorflow installed without GPU. Just CPU computations "
               "enabled (slow computations).")
        cudaMsgs.append(msg)
        useGpu = False

    else:
        if int(nvidiaDriverVer) < NVIDIA_DRIVERS_MINIMUM_VERSION:
            preMsgs.append("Incompatible driver %s" % nvidiaDriverVer)
            cudaMsgs.append(f"Your NVIDIA drivers are too old (<{NVIDIA_DRIVERS_MINIMUM_VERSION}). "
                            "Tensorflow was installed without GPU support. "
                            "Just CPU computations enabled (slow computations)."
                            f"To enable CUDA (drivers>{NVIDIA_DRIVERS_MINIMUM_VERSION} needed), "
                            "set CUDA=True in 'scipion.conf' file")
            useGpu = False
        else:
            preMsgs.append("CUDA support found. Driver version: %s" % nvidiaDriverVer)
            msg = "Tensorflow will be installed with CUDA SUPPORT."
            cudaMsgs.append(msg)
            useGpu = True

    # commands  = [(command, target), (cmd, tgt), ...]
    cmdsInstall = list(CondaEnvManager.yieldInstallAllCmds(useGpu=useGpu))

    now = datetime.now()
    installDLvars = {'modelsUrl': "https://scipion.cnb.csic.es/downloads/scipion/software/em",
                     'syncBin': plugin.getHome('bin/xmipp_sync_data'),
                     'modelsDir': plugin.getHome('models'),
                     'modelsPrefix': "models_UPDATED_on",
                     'xmippLibToken': 'xmippLibToken',
                     'libXmipp': plugin.getHome('lib/libXmipp.so'),
                     'preMsgsStr': ' ; '.join(preMsgs),
                     'afterMsgs': ", > ".join(cudaMsgs)}

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

    env.addPackage(XMIPP_DLTK_NAME, version='1.0', urlSuffix='external',
                   commands=[xmippInstallCheck]+cmdsInstall+[modelsDownloadCmd],
                   deps=[], tar=XMIPP_DLTK_NAME+'.tgz')
