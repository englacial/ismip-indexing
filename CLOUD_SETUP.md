# Cloud Setup for ISMIP6 Indexing

This document describes the cloud infrastructure setup required for running the ISMIP6 virtualization pipeline using Lithops on Google Cloud Platform.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and configured
- Access to a GCP project with billing enabled
- Permissions to create service accounts and IAM policies

## GCP Service Account Setup

Lithops requires a service account with proper permissions to deploy Cloud Functions and access Cloud Storage.

### 1. Set Your Project ID

```bash
export PROJECT_ID=$(gcloud config get-value project)
```

For this project, the project ID is: `ds-englacial`

### 2. Create Service Account

```bash
gcloud iam service-accounts create lithops-executor \
    --display-name="Lithops Executor Service Account" \
    --project=$PROJECT_ID
```

### 3. Grant Required Permissions

See [Lithops GCP Functions Documentation](https://lithops-cloud.github.io/docs/source/compute_config/gcp_functions.html) for a list of required permissions - those below need to be updated.

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
gcloud iam service-accounts keys create ~/lithops-sa-key.json \
    --iam-account=lithops-executor@${PROJECT_ID}.iam.gserviceaccount.com
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

gcp:
    region: us-west1
    credentials_path: ~/lithops-sa-key.json

gcp_functions:
    region: us-west1
    runtime: ismip6-icechunk

gcp_storage:
    storage_bucket: ismip6-icechunk
    region: us-west1
```

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
- **Key Location**: `~/lithops-sa-key.json`

## Build the runtime

```bash
# lithops runtime delete ismip6-icechunk -c lithops.yaml
lithops runtime build -f requirements.txt ismip6-icechunk -c lithops.yaml
lithops runtime deploy ismip6-icechunk -c lithops.yaml
```

## Troubleshooting

### Authentication Errors

If you see errors like:
```
Service account info was not in the expected format, missing fields client_email, token_uri
```

This means you're trying to use Application Default Credentials instead of a service account key. Make sure `credentials_path` in `lithops.yaml` points to the service account JSON key file.

### Permission Errors

If Cloud Functions fail to deploy or access storage, verify the service account has the correct roles:

```bash
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:lithops-executor@${PROJECT_ID}.iam.gserviceaccount.com"
```
