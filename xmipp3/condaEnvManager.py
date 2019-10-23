import os
import re
import subprocess

from .condaEnvsDef import DICT_OF_CONDA_ENVIRONS

class CondaEnvManager(object):
  DICT_OF_CONDA_ENVIRONS= DICT_OF_CONDA_ENVIRONS

  @staticmethod
  def getCoondaRoot(env=None):
    '''
    Tries to find the conda root path given an environment
    :param: env. An environ obtaining using Plugin.getEnviron()
    :return: None if conda not found or CONDA_ROOT_PATH (could be defined into config file??)
    '''
    if env==None:
      env=[]
    if "CONDA_HOME" in env:  # TODO. Allow CONDA_HOME to be in config file
      condaRoot = "CONDA_HOME"
    if "CONDA_ACTIVATION_CMD" in env:
      condaRoot = os.path.split(os.path.split(os.path.split(env["CONDA_ACTIVATION_CMD"])[0])[0])[0]
      condaRoot = re.sub("^\.", "", condaRoot).strip()
    else:
      if "CONDA_EXE" in env:
        condaRoot = env["CONDA_EXE"]
        success = True
      else:
        try:
          condaRoot = subprocess.check_output(["which", "conda"])
          success = True
        except subprocess.CalledProcessError:
          success = False

      if success:
        condaRoot = os.path.split(os.path.split(condaRoot)[0])[0]

    assert condaRoot is not None, "Error, conda was not found"+str(env)
    return condaRoot

  @staticmethod
  def getCondaPathInEnv(condaRoot, condaEnv, condaSubDir=""):
    '''
    :param condaRoot: The path where conda is installed. E.g.  ~/tools/miniconda3
    :param condaEnv: The name (prefix) of the conda environment that will be used
    :param condaSubDir: The path of the subdirectory within conda env. E.g. condaSubDir="bin -> path/to/env/bin
    :return: the path to a subdirectory within condaEnv
    '''
    return os.path.join(condaRoot, "envs", condaEnv, condaSubDir)

  @staticmethod
  def modifyEnvToUseConda(env, condaEnv):
      env.update( {"PATH": CondaEnvManager.getCondaPathInEnv(CondaEnvManager.getCoondaRoot(env),
                                                             condaEnv, "bin")+":"+env["PATH"] } )

      newPythonPath= CondaEnvManager.getCondaPathInEnv(CondaEnvManager.getCoondaRoot(env),
                                                   condaEnv, "lib/python*/site-packages/")
      if CondaEnvManager.DICT_OF_CONDA_ENVIRONS[condaEnv]["pythonVersion"][:3]=="2.7":
        newPythonPath+=":"+env["PYTHONPATH"]
      env.update( {"PYTHONPATH": newPythonPath})
      return env

  @staticmethod
  def getCondaActivationCmd():
    condaActivationCmd = os.environ.get('CONDA_ACTIVATION_CMD', "")
    if not condaActivationCmd:
      condaRoot= CondaEnvManager.getCoondaRoot()
      condaActivationCmd= ". "+os.path.join(condaRoot, "etc/profile.d/conda.sh")
    if condaActivationCmd[-1] == ";":
      condaActivationCmd= condaActivationCmd[:-1]+" &&  "
    elif condaActivationCmd[-2] != "&&":
      condaActivationCmd += " && "
    else:
      condaActivationCmd+= " && "

    return condaActivationCmd

  @staticmethod
  def yieldInstallAllCmds( useGpu):
    installCmdOptions={}
    if useGpu:
      installCmdOptions["gpuTag"]="-gpu"
    else:
      installCmdOptions["gpuTag"]=""

    for envName in CondaEnvManager.DICT_OF_CONDA_ENVIRONS:
      yield CondaEnvManager.installEnvironCmd(envName, installCmdOptions=installCmdOptions,
                                              **CondaEnvManager.DICT_OF_CONDA_ENVIRONS[envName])

  @staticmethod
  def installEnvironCmd( environName, pythonVersion, dependencies, channels,
                     defaultInstallOptions, pipPackages=[], installCmdOptions=None):

    cmd="export PYTHONPATH=\"\" && conda create -q --force --yes -n "+environName+" python="+pythonVersion+" "
    cmd += " "+ " ".join([dep for dep in dependencies])
    if len(channels)>0:
      cmd += " -c "+ " -c ".join([chan for chan in channels])
    if installCmdOptions is not None:
      try:
        cmd= cmd%installCmdOptions
      except KeyError:
        pass
    else:
      try:
        cmd= cmd%defaultInstallOptions
      except KeyError:
        pass
    cmd += " && " + CondaEnvManager.getCondaActivationCmd() + " conda activate " + environName
    # cmd += " && export PATH="+ os.path.join(CondaEnvManager.getCoondaRoot(), "bin")+":$PATH &&  conda activate "+environName

    if len(pipPackages)>0:
      cmd += " && pip install  "
      cmd += " " + " ".join([dep for dep in pipPackages])
    cmd += " && conda env export > "+environName+".yml"
    return cmd, environName
