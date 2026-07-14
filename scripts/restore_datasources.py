import os
import sys
import time
import requests
from pymilvus import connections, utility, Collection

RAG_URL = os.getenv("RAG_URL", "http://localhost:9446")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

INSECURE_SSL = os.getenv("INSECURE_SSL", "false").lower() in ("true", "1", "yes")

def get_oidc_token():
    """Fetch OIDC token dynamically if client credentials are available, or reuse environment token."""
    token = os.getenv("CAIPE_OIDC_TOKEN")
    if token:
        return token
        
    print("No CAIPE_OIDC_TOKEN found in environment. Attempting to fetch from Keycloak...")
    try:
        import subprocess
        # Fetch client credentials using kubectl
        client_id_cmd = "kubectl get secret rag-ingestor-secret -n caipe -o jsonpath='{.data.INGESTOR_OIDC_CLIENT_ID}'"
        client_secret_cmd = "kubectl get secret rag-ingestor-secret -n caipe -o jsonpath='{.data.INGESTOR_OIDC_CLIENT_SECRET}'"
        
        client_id_b64 = subprocess.check_output(client_id_cmd, shell=True).decode().strip()
        client_secret_b64 = subprocess.check_output(client_secret_cmd, shell=True).decode().strip()
        
        import base64
        client_id = base64.b64decode(client_id_b64).decode()
        client_secret = base64.b64decode(client_secret_b64).decode()
        
        token_url = "http://localhost:7080/realms/caipe/protocol/openid-connect/token"
        resp = requests.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            verify=not INSECURE_SSL,
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception as e:
        print(f"Failed to fetch OIDC token dynamically: {e}")
        return None

def restore():
    # 1. Fetch token
    token = get_oidc_token()
    if not token:
        print("ERROR: Authentication token is required to register datasources.")
        sys.exit(1)
        
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 2. Connect to Milvus and scan all unique datasource IDs
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        sys.exit(1)

    collections = utility.list_collections()
    if "rag_default" not in collections:
        print("ERROR: Collection 'rag_default' not found in Milvus.")
        sys.exit(1)

    col = Collection("rag_default")
    col.load()
    print(f"Scanning 'rag_default' (Total chunks: {col.num_entities})...")

    unique_ds_ids = set()
    try:
        iterator = col.query_iterator(
            batch_size=5000,
            expr="id != ''",
            output_fields=["datasource_id"]
        )
        while True:
            res = iterator.next()
            if not res:
                iterator.close()
                break
            for r in res:
                ds_id = r.get("datasource_id")
                if ds_id:
                    unique_ds_ids.add(ds_id)
    except Exception as e:
        print(f"Failed to scan Milvus: {e}")
        sys.exit(1)

    print(f"Unique datasource IDs found in Milvus: {unique_ds_ids}")

    # 3. Fetch registered datasources from RAG server
    print("Fetching active datasources from RAG server...")
    try:
        resp = requests.get(
            f"{RAG_URL}/v1/datasources",
            headers=headers,
            verify=not INSECURE_SSL,
            timeout=10,
        )
        resp.raise_for_status()
        registered_ds = {ds["datasource_id"] for ds in resp.json().get("datasources", [])}
        print(f"Registered datasources in Redis: {registered_ds}")
    except Exception as e:
        print(f"Failed to fetch active datasources: {e}")
        sys.exit(1)

    # 4. Restore missing datasources
    missing_ds = unique_ds_ids - registered_ds
    if not missing_ds:
        print("No missing datasources. Everything is in sync!")
        return

    print(f"Restoring {len(missing_ds)} missing datasources...")
    for ds_id in missing_ds:
        # Determine source type and ingestor based on ID conventions
        source_type = "hotpotqa" if "hotpot" in ds_id else "webloader"
        ingestor_id = "hotpotqa:hotpotqa-eval-script" if source_type == "hotpotqa" else "webloader:webloader-ingestor"
        name = ds_id.replace("_", " ").title()

        payload = {
            "datasource_id": ds_id,
            "name": name,
            "ingestor_id": ingestor_id,
            "description": "Auto-restored from existing Milvus chunks",
            "source_type": source_type,
            "reload_interval": 315360000,  # 10 years (effectively disables cleanup)
            "last_updated": int(time.time()),
        }

        print(f"Registering datasource '{ds_id}'...")
        try:
            register_resp = requests.post(
                f"{RAG_URL}/v1/datasource",
                json=payload,
                headers=headers,
                verify=not INSECURE_SSL,
                timeout=10,
            )
            register_resp.raise_for_status()
            print(f"Successfully restored: {ds_id}")
        except Exception as e:
            print(f"Failed to restore '{ds_id}': {e}")
            if register_resp := getattr(e, "response", None):
                print(f"Response: {register_resp.text}")

if __name__ == "__main__":
    restore()
