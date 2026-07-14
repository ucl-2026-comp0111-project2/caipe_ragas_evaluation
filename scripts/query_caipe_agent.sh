#!/bin/bash

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

INGESTOR_ID=$(kubectl get secret rag-ingestor-secret -n caipe -o jsonpath='{.data.INGESTOR_OIDC_CLIENT_ID}' | base64 --decode)
INGESTOR_SECRET=$(kubectl get secret rag-ingestor-secret -n caipe -o jsonpath='{.data.INGESTOR_OIDC_CLIENT_SECRET}' | base64 --decode)

TOKEN=$(curl -k -s -X POST "https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token" \
  -d "client_id=$INGESTOR_ID" \
  -d "client_secret=$INGESTOR_SECRET" \
  -d "grant_type=client_credentials" | jq -r '.access_token')

# curl -s -H "Authorization: Bearer $TOKEN" http://localhost:9446/v1/datasources | jq .
curl -k -s -H "Authorization: Bearer $TOKEN" https://rag.caipe.homelab/api/v1/chat/completions \
-X POST \
-H "Content-Type: application/json" \
-d '{
    "message": "Verify why the production build failed on ArgoCD and summarize the cluster errors.",
    "stream": false,
    "agent_profile": "platform-engineer",
    "context_variables": {
      "user_id": "eval-bot"
    }
  }' | jq .
