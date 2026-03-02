#!/usr/bin/env bash
# Load environment variables from .env file into current shell
# Usage: source scripts/load_env.sh [path/to/.env]

ENV_FILE="${1:-.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: .env file not found: $ENV_FILE" >&2
  return 1
fi

echo "Loading environment variables from $ENV_FILE"

# Auto-export all variables while sourcing
set -a
source "$ENV_FILE"
set +a

echo "Environment variables loaded"
