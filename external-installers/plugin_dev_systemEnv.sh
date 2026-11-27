#!/bin/bash

#------------------------------------------------
# Requirements
#------------------------------------------------
# - This script must be located in the Scipion root directory, alongside the
#   scipion3 executable.
# - Make the script executable before running it:
#       chmod +x plugin_dev_systemEnv.sh
# - Run it using:
#       ./plugin_dev_systemEnv.sh

#------------------------------------------------
# User configuration
#------------------------------------------------
# BUNDLE_DIR: Name of the Xmipp bundle directory.
# XMIPP_BRANCH: Xmipp repository branch to install.
# SCIPION_EM_XMIPP_BRANCH: scipion-em-xmipp repository branch.
#                       Defaults to XMIPP_BRANCH if not specified.

set -Ee
trap 'echo "ERROR: Script failed at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

#------------------------------------------------
BUNDLE_DIR="xmipp-TestEnv"
XMIPP_BRANCH="${XMIPP_BRANCH:-main}" #-main or -omad_xmipp3_env or ...
SCIPION_EM_XMIPP_BRANCH="${SCIPION_EM_XMIPP_BRANCH:-main}" # -main or -$XMIPP_BRANCH ...
XMIPP3_ENV_VERSION="xmipp3-installer"

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

echo "- Installing xmipp3-installer..."
scipion3_cmd pip install --only-binary :all: "$XMIPP3_ENV_VERSION"

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
scipion3_cmd  run ./xmipp

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