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
# The credentials (Client ID and Secret) are fetched directly from the Kubernetes cluster secret 'rag-ingestor-secret' in the 'caipe' namespace.
CLIENT_ID=$(kubectl get secret rag-ingestor-secret -n caipe -o jsonpath='{.data.INGESTOR_OIDC_CLIENT_ID}' | base64 --decode)
CLIENT_SECRET=$(kubectl get secret rag-ingestor-secret -n caipe -o jsonpath='{.data.INGESTOR_OIDC_CLIENT_SECRET}' | base64 --decode)

# Fetch OIDC token from Keycloak
export CAIPE_OIDC_TOKEN=$(curl -s -X POST "http://localhost:7080/realms/caipe/protocol/openid-connect/token" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "grant_type=client_credentials" | jq -r '.access_token')

# Ensure src/ is in the PYTHONPATH so python can find the package
export PYTHONPATH=src:$PYTHONPATH

# Run evaluation. Add flags as needed:
#   --short-answer          Use SemanticSimilarity + ContainsAnswer (for HotpotQA-style short-answer datasets)
#   --retrieval-only        Measure context_precision + context_recall only
#   --generation-only       Measure answer quality only (skip context metrics)
#   --compute-model-eval    Evaluate pre-existing model answers from the datasource
#   --limit-per-category N  Limit questions per category
#   --top-k N               Number of documents to retrieve

# Activate virtual environment and run evaluation
# shellcheck source=.venv/bin/activate
source .venv/bin/activate
python3 -m ragas_eval.evals --limit 1 --top-k 5 --agentic
