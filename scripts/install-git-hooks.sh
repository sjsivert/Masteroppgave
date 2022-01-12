#!/bin/bash
SCRIPTS_DIR="$( cd "$( dirname "$0" )" && pwd)"
SCRIPT_PATH="$(realpath "$0")"

echo "$BASEDIR"
echo "Installs git hooks to .git/ directory with symlinks"
ln -s "$SCRIPTS_DIR/pre-commit" "$SCRIPTS_DIR/../.git/hooks/pre-commit"
echo "DONE!"
