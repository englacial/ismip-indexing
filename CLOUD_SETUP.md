# Cloud Setup for ISMIP6 Indexing

This document describes the cloud infrastructure setup required for running the ISMIP6 virtualization pipeline using Lithops on Google Cloud Platform.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and configured
    - Determine which email is authorized to see the project
    - Make sure you are logged into gcloud with that email
        - Check the active account with `gcloud auth list`
            - If the account shows up but is not active do `gcloud config set account <your_account>`
            - If not log in with `gcloud auth login`
- Access to a GCP project with billing enabled
- Permissions to create service accounts and IAM policies

## GCP Service Account Setup

Lithops requires a service account with proper permissions to deploy Cloud Functions and access Cloud Storage.

### 1. Set Your Project ID

```bash
export PROJECT_ID=$(gcloud config get-value project)
```

For this project, the project ID is: `astera-englacial`



### 2. Create Service Account

**If the service account was already created you can skip to Step 4.**

```bash
gcloud iam service-accounts create lithops-executor \
    --display-name="Lithops Executor Service Account" \
    --project=$PROJECT_ID
```

### 3. Grant Required Permissions

See [Lithops GCP Functions Documentation](https://lithops-cloud.github.io/docs/source/compute_config/gcp_functions.html) for a list of required permissions - THOSE BELOW NEED TO BE UPDATED.

Grant Cloud Functions Developer role (to deploy and manage serverless functions):

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:lithops-executor@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/cloudfunctions.developer"
```

Grant Storage Object Admin role (to read/write GCS buckets):

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:lithops-executor@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

```

### 4. Download Service Account Key

```bash
gcloud iam service-accounts keys create ~/lithops-sa-key.json --iam-account=lithops-executor@${PROJECT_ID}.iam.gserviceaccount.com
```

This creates a JSON key file at `~/lithops-sa-key.json`.

## Enable services

See [Lithops GCP Functions Documentation](https://lithops-cloud.github.io/docs/source/compute_config/gcp_functions.html) for a list of Google Cloud services which need to be enabled.

## Lithops Configuration

The `lithops.yaml` configuration file should reference the service account key:

```yaml
lithops:
    backend: gcp_functions
    storage: gcp_storage
    data_limit: False

gcp:
    region: us-west1
    credentials_path: ~/lithops-sa-key.json

gcp_functions:
    region: us-west1
    runtime: ismip6-icechunk
    runtime_memory: 8192
    runtime_timeout: 540
    project_id: astera-englacial

gcp_storage:
    storage_bucket: ismip6-icechunk # Lithops uses storage_bucket, not bucket
    region: us-west1
```

### Additional steps that Julius needed to reproduce
- Get permission for service account on `astera-englacial` to be `'storage.admin'` on `gs://ismip6-icechunk`

- Needed to manually activate the following APIS:
    -  Cloud Pub/Sub 
    - Cloud Functions
    - Cloud Build
    - Artifact Registry
    - All of these might take a bit of time to activate.
- The service account needed the following additional permissions:
    - Command `gcloud projects add-iam-policy-binding astera-englacial \
  --member=serviceAccount:lithops-executor@astera-englacial.iam.gserviceaccount.com \
  --role=roles<your_role>`
    - `pubsub.admin`
    - `Cloud Functions Admin` (forgot the exact syntax)
    - `iam.serviceAccountUser`
- Downgrade to python 3.12
    - Delete old function definitions (just to be sure)
    - Add `gen: 2` to lithops config.

### Important Notes

- **Do NOT use Application Default Credentials**: Lithops requires a service account key file with `client_email` and `token_uri` fields, which ADC doesn't provide
- **Keep the key file secure**: The service account key provides full access to the granted permissions
- **Storage bucket**: The `storage_bucket` parameter specifies where Lithops stores intermediate results and metadata

## GCS Buckets Used

- `gs://ismip6`: Source data bucket (public, read-only)
- `gs://ismip6-icechunk`: Target bucket for Icechunk repositories and failure logs

## Service Account Created

- **Name**: `lithops-executor`
- **Email**: `lithops-executor@ds-englacial.iam.gserviceaccount.com`
- **Roles**:
  - `roles/cloudfunctions.developer`
  - `roles/storage.objectAdmin`
  - OTHERS/ADD ME
- **Key Location**: `~/lithops-sa-key.json`

## Build the runtime

```bash
# To create the requirements I exported the locked uv env 
# `uv export --no-hashes --no-dev --no-editable > requirements.txt`
# and then manually populated requirements-lithops.txt with the same versions
# This could definitely be improved
# Then build and deploy the runtime (If the deploy succeeds you should be good to go!)
uv run lithops runtime build -f requirements-lithops.txt ismip6-icechunk -c lithops.yaml 
uv run lithops runtime deploy ismip6-icechunk -c lithops.yaml
```

## Running the script

And then run the script with uv
```
uv run python virtualize_with_lithops_combine_variables.py
```