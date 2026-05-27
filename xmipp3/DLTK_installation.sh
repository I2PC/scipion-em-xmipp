#!/bin/sh
set -eu

# =========================
# GLOBAL CONFIGURATION
# =========================

# Enable/disable GPU versions (POSIX-safe: 1 = true, 0 = false)
USE_GPU=1

# Conda initialization script path
CONDA_INIT="/home/USERNAME/miniconda/etc/profile.d/conda.sh"

# Base Scipion directory
SCIPION_DIR="/home/USERNAME/scipion3"

# Xmipp base directory
XMIPP_DIR="${SCIPION_DIR}/xmipp-bundle"

# DeepLearningToolkit output directory
DLTK_DIR="${SCIPION_DIR}/software/em/deepLearningToolkit"

# Base environment names (POSIX: no arrays, use space-separated string)
ENV_LIST="xmipp_DLTK_v0.3 xmipp_DLTK_v1.0 xmipp_jax xmipp_pyTorch xmipp_deepEMhancer xmipp_MicCleaner"

# =========================
# ACTIVATE CONDA
# =========================
. "${CONDA_INIT}"

mkdir -p "${DLTK_DIR}"

# =========================
# FUNCTION (POSIX STYLE)
# =========================
get_yaml() {
  base="$1"

  # Try GPU version if enabled
  if [ "$USE_GPU" -eq 1 ]; then
    if [ -f "${XMIPP_DIR}/src/xmipp/bindings/python/envs_DLTK/${base}-gpu.yml" ]; then
      echo "${XMIPP_DIR}/src/xmipp/bindings/python/envs_DLTK/${base}-gpu.yml"
      return
    fi
  fi

  # Fallback CPU version
  echo "${XMIPP_DIR}/src/xmipp/bindings/python/envs_DLTK/${base}.yml"
}

# =========================
# CREATE ENVIRONMENTS
# =========================
for env in $ENV_LIST; do
  YAML=$(get_yaml "$env")

  echo "======================================"
  echo "Creating environment: $env"
  echo "Using YAML file: $YAML"
  echo "======================================"

  conda env create -f "$YAML" || {
    echo "ERROR while creating $env"
    exit 1
  }

  echo "Exporting environment: $env"

  conda env export -n "$env" > "${DLTK_DIR}/${env}-1.yml"
done

echo "✔ All environments processed successfully"
