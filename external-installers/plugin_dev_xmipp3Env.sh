#!/bin/bash

#------------------------------------------------
# Requirements
#------------------------------------------------
# - This script must be located in the Scipion root directory, alongside the
#   scipion3 executable.
# - Add XMIPP3_INSTALL_CONDA_DEPENDENCIES=ON at scipion.conf
# - Make the script executable before running it:
#       chmod +x plugin_dev_xmipp3Env.sh
# - Run it using:
#       ./plugin_dev_xmipp3Env.sh

#------------------------------------------------
# User configuration
#------------------------------------------------
# BUNDLE_DIR: Name of the Xmipp bundle directory.
# XMIPP_BRANCH: Xmipp repository branch to install.
# SCIPION_EM_XMIPP_BRANCH: scipion-em-xmipp repository branch.
#                       Defaults to XMIPP_BRANCH if not specified.


set -Ee
trap 'status=$?; echo "ERROR: Installation failed at line $LINENO (exit code: $status)." >&2' ERR

#------------------------------------------------
BUNDLE_DIR="xmipp-TestEnv"
XMIPP_BRANCH="${XMIPP_BRANCH:-main}" #-main or omad_xmipp3_env
SCIPION_EM_XMIPP_BRANCH="${SCIPION_EM_XMIPP_BRANCH:-main}" # -main or $XMIPP_BRANCH

ENV_NAME="xmipp3"
PYTHON_VERSION="3.11"
XMIPP3_ENV_VERSION="xmipp3-installer>=2.2.0,<3"
#------------------------------------------------
echo "=========================================="
echo "Xmipp installation configuration"
echo "=========================================="
printf "BUNDLE_DIR:              %s\n" "$BUNDLE_DIR"
printf "XMIPP_BRANCH:            %s\n" "$XMIPP_BRANCH"
printf "SCIPION_EM_XMIPP_BRANCH: %s\n" "$SCIPION_EM_XMIPP_BRANCH"
echo "=========================================="
#------------------------------------------------


echo "- Checking Scipion executable"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCIPION3_CMD="${SCRIPT_DIR}/scipion3"

if [[ ! -x "$SCIPION3_CMD" ]]; then
    echo "ERROR: Scipion executable not found or not executable: $SCIPION3_CMD"
    exit 1
fi

echo "Scipion executable found: $SCIPION3_CMD"

# Execute Scipion commands
scipion3_cmd() {
    "$SCIPION3_CMD" "$@"
}



echo "- Initializing Conda..."

CONDA_BASE=$(bash -ic 'conda info --base' 2>/dev/null)

if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
	echo ""
	echo "ERROR: Could not find a valid Conda installation."
	echo "Please make sure that Conda (Anaconda, Miniconda or Miniforge) is installed and initialized in your shell."
	echo "If Conda is already installed, initialize it by running:"
	echo "    conda init bash"
	echo "Then close and reopen your terminal and run this script again."
	echo ""
	exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Conda initialized from: $CONDA_BASE"

echo "- Create Conda environment"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
	echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
	echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
	conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi


echo "- Activating environment"

conda activate "$ENV_NAME"
echo "Activated Conda environment: $ENV_NAME"

# Install xmipp3-installer

echo "- Installing xmipp3-installer..."
pip install --only-binary :all: "$XMIPP3_ENV_VERSION"
# Temporary solution until 2.2.0 is released
#if [ ! -d "xmipp3-installer" ]; then
#    git clone https://github.com/I2PC/xmipp3-installer.git
#fi
#pip install -e ./xmipp3-installer


echo "- Cloning and install Xmipp bundle"

if [ -d "$BUNDLE_DIR" ]; then
echo "ERROR: Directory '$BUNDLE_DIR' already exists."
echo "Please remove or rename it before running this script."
exit 1
fi
echo "Cloning Xmipp bundle..."
git clone --branch "$XMIPP_BRANCH" https://github.com/I2PC/xmipp3 "$BUNDLE_DIR"
cd "$BUNDLE_DIR"
echo "Running Xmipp installer..."
(
    export SCIPION_HOME="$SCRIPT_DIR"
    export SCIPION_TESTS_CMD="scipion3 tests"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${SCRIPT_DIR}/software/lib"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${SCRIPT_DIR}/software/bindings"
    export PYTHONUSERBASE="${SCRIPT_DIR}/software/python-packages"

    ./xmipp
)

# Clone Scipion Xmipp plugin

cd src
echo "Cloning scipion-em-xmipp..."
git clone --branch "$SCIPION_EM_XMIPP_BRANCH" \
    https://github.com/I2PC/scipion-em-xmipp.git
echo "Installing scipion-em-xmipp in development mode..."
scipion3_cmd installp -p scipion-em-xmipp/ --devel

echo "=========================================="
echo "Xmipp installation completed successfully"
echo "=========================================="