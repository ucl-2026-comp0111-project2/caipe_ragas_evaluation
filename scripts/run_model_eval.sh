#!/bin/bash

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Fetch OIDC credentials from Kubernetes
# Assumption: CAIPE is deployed using KinD (Kubernetes in Docker) with OIDC enabled.
# The credentials (Client ID and Secret) are fetched directly from the Kubernetes cluster secret 'caipe-ui-secret' in the 'caipe' namespace.
CLIENT_ID=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_ID}' | base64 --decode)
CLIENT_SECRET=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_SECRET}' | base64 --decode)

# Fetch OIDC token from Keycloak
export CAIPE_OIDC_TOKEN=$(curl -sk -X POST "https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "grant_type=client_credentials" | jq -r '.access_token')

# Ensure src/ is in the PYTHONPATH so python can find the package
export PYTHONPATH=src:$PYTHONPATH

# Activate virtual environment and run evaluation
# shellcheck source=.venv/bin/activate
source .venv/bin/activate
python3 -m ragas_eval.evals --compute-model-eval --limit 10 --top-k 5
