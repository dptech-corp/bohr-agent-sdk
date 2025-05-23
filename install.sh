#!/bin/bash

OS="$(uname)"
REPO_URL="https://raw.githubusercontent.com/dptech-corp/science-agent-sdk/refs/heads/master"

echo "Installing dependencies..."

if [ "$OS" == "Linux" ] || [ "$OS" == "Darwin" ]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -e git+https://github.com/dptech-corp/science-agent-sdk.git#egg=science-agent-sdk
elif [ "$OS" == "Windows_NT" ] || [[ "$OS" == MINGW* ]] || [[ "$OS" == CYGWIN* ]]; then
  python -m pip install --upgrade pip
  python -m pip install -e git+https://github.com/dptech-corp/science-agent-sdk.git#egg=science-agent-sdk
else
  echo "Unsupported operating system: $OS"
  exit 1
fi

echo "DP Agent CLI installed successfully!"
echo "Use 'dp-agent --help' to see available commands."
