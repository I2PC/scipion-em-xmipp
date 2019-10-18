from .constants import CONDA_DEFAULT_ENVIRON


DICT_OF_CONDA_ENVIRONS = {
  CONDA_DEFAULT_ENVIRON: {
    "pythonVersion": "2.7",
    "dependencies": ["pandas=0.23.4", "scikit-image=0.14.2", "opencv=3.4.2",
                     "tensorflow%(gpuTag)s==1.10.0", "keras=2.2.2"],
    "channels": ["anaconda"],
    "pipPackages": [],
    "defaultInstallOptions": {"gpuTag": ""},  # Tags to be replaced to %(tag)s
  },

  "deepLearningToolkit_v0.01": {
    "pythonVersion": "2.7",
    "dependencies": ["pandas=0.23.4", "scikit-image=0.14.2", "opencv=3.4.2",
                     "tensorflow%(gpuTag)s==1.10.0", "keras=2.1.5"],
    "channels": ["anaconda"],
    "pipPackages": [],
    "defaultInstallOptions": {"gpuTag": ""},  # Tags to be replaced to %(tag)s
  },

  "micrograph_cleaner_em": {
    "pythonVersion": "2.7",
    "dependencies": ["micrograph-cleaner-em"],
    "channels": ["rsanchez1369", "anaconda", "conda-forge"],
    "pipPackages": [],
    "defaultInstallOptions": {},
  }

}
