#!/bin/bash

# Default values
DEFAULT_PORT=8000
DEFAULT_OPTIONS="mlx,mlx-community/Nous-Hermes-2-Mixtral-8x7B-DPO-4bit"
DEFAULT_DEBUG_DIR="/tmp/sparrow_debug"
DEFAULT_LOG_LEVEL="INFO"
SPARROW_LLM_INFERENCE_BACKEND_CONFIG="" # For backend selection override

# Function to display help
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --port <number>                 Set the port number (default: $DEFAULT_PORT)"
  echo "  --options <csv_string>          Set the model options (default: '$DEFAULT_OPTIONS')"
  echo "  --debug-dir <path>              Set the debug directory (default: $DEFAULT_DEBUG_DIR)"
  echo "  --log-level <level>             Set the log level (INFO, DEBUG, etc.) (default: $DEFAULT_LOG_LEVEL)"
  echo "  --inference-backend <backend>   Set the LLM inference backend. Overrides config file and SPARROW_LLM_INFERENCE_BACKEND_CONFIG env var."
  echo "                                    Examples: pytorch_cuda, pytorch_cpu, pytorch_cuda_if_available"
  echo "  -h, --help                        Display this help message"
  exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT_CHOICE="$2"; shift ;;
    --options) OPTIONS_CHOICE="$2"; shift ;;
    --debug-dir) DEBUG_DIR_CHOICE="$2"; shift ;;
    --log-level) LOG_LEVEL_CHOICE="$2"; shift ;;
    --inference-backend) SPARROW_LLM_INFERENCE_BACKEND_CONFIG="$2"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Set values with defaults if not provided
PORT=${PORT_CHOICE:-$DEFAULT_PORT}
OPTIONS=${OPTIONS_CHOICE:-$DEFAULT_OPTIONS}
DEBUG_DIR=${DEBUG_DIR_CHOICE:-$DEFAULT_DEBUG_DIR}
LOG_LEVEL=${LOG_LEVEL_CHOICE:-$DEFAULT_LOG_LEVEL}

# Export environment variables
export SPARROW_PORT=$PORT
export SPARROW_OPTIONS=$OPTIONS
export SPARROW_DEBUG_DIR=$DEBUG_DIR
export SPARROW_LOG_LEVEL=$LOG_LEVEL
if [ -n "$SPARROW_LLM_INFERENCE_BACKEND_CONFIG" ]; then
  export SPARROW_LLM_INFERENCE_BACKEND_CONFIG # This makes it available to the Python app
  echo "LLM Inference Backend set via command line/script: $SPARROW_LLM_INFERENCE_BACKEND_CONFIG"
fi


echo "Starting Sparrow LLM API Server..."
echo "Port: $SPARROW_PORT"
echo "Options: $SPARROW_OPTIONS"
echo "Default Options: $SPARROW_OPTIONS"
echo "Debug Directory: $SPARROW_DEBUG_DIR"
echo "Log Level: $SPARROW_LOG_LEVEL"
if [ -n "${SPARROW_LLM_INFERENCE_BACKEND_CONFIG}" ]; then
    echo "LLM Inference Backend (from env/cmd): ${SPARROW_LLM_INFERENCE_BACKEND_CONFIG}"
else
    echo "LLM Inference Backend will be determined by llm/config.properties or application defaults."
fi

# The actual execution command:
# Replace with the correct way to start your FastAPI app for the LLM service.

command -v python >/dev/null 2>&1 || { echo >&2 "Python is required but it's not installed. Aborting."; exit 1; }

# Check Python version
PYTHON_VERSION=$(python --version 2>&1) # Capture both stdout and stderr
echo "Detected Python version: $PYTHON_VERSION"
if [[ ! "$PYTHON_VERSION" == *"3.10.4"* ]]; then
  echo "Python version 3.10.4 is required. Current version is $PYTHON_VERSION. Aborting."
  exit 1
fi

PYTHON_SCRIPT_PATH="engine.py"

if [ "$1" == "assistant" ]; then
    PYTHON_SCRIPT_PATH="assistant.py"
    shift # Shift the arguments to exclude the first one
fi

python "${PYTHON_SCRIPT_PATH}" "$@"

# make script executable with: chmod +x sparrow.sh