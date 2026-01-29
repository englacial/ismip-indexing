# Running ISMIP6 Virtualization on AWS

This document describes how to set up and run the ISMIP6 Lithops virtualization
pipeline on AWS Lambda with S3-backed Icechunk storage.

## Background: Why Move to AWS?

The existing GCP pipeline hits a fundamental bottleneck: **GCS rate-limits per-object
mutation operations** on the Icechunk branch ref file
(`refs/branch.main/ref.json`). Every commit updates this single object, and GCS
returns `429 SlowDown` errors when it's mutated too frequently. This forced the
current pipeline into fully sequential batch writes, making the ~6 hour runtime
dominated by ~440 serial commit cycles rather than actual compute.

S3 does not have per-object mutation rate limits in the same way. S3 supports
3,500 PUT requests/second per prefix, and the rate limit is at the prefix level,
not the individual object level. This means the Icechunk ref file updates that
choke on GCS should work without throttling on S3.

## Can We Write Batches Concurrently to Icechunk?

**Short answer: yes, with caveats.**

Icechunk uses an optimistic concurrency model similar to git. Each
`writable_session` is a snapshot-isolated transaction. When you call
`session.commit()`, Icechunk attempts to update the branch ref pointer (an
atomic compare-and-swap on the ref object in object storage). If the branch
tip has moved since the session was opened, the commit fails with a
`ConflictError`.

However, Icechunk provides a **rebase mechanism** for exactly this case:

```python
session.commit(
    "Added batch X",
    rebase_with=icechunk.ConflictDetector(),
    rebase_tries=10
)
```

When `rebase_with` is specified, a failed commit will automatically:
1. Fetch the new branch tip
2. Check that the new commits don't conflict with this session's writes
   (ConflictDetector verifies no overlapping paths were modified)
3. Rebase this session's changes on top of the new tip
4. Retry the commit

Since each batch writes to a unique group path
(`{institution}_{model}/{experiment}`), there are **no actual conflicts** --
the writes are to disjoint parts of the store. The ConflictDetector will
always succeed in rebasing.

**The problem on GCP was not the concurrency model -- it was GCS choking on
the ref file mutations.** Even sequential commits eventually triggered 429
errors because Icechunk's internal retry loop hammered the same object. The
commented-out retry/backoff code in the repo (`virtualize_with_lithops_combine_variables.py`
lines 35-96) was an attempt to work around this at the application level, but
it didn't help because the throttling is at the storage layer.

On S3, the rebase-based concurrent commit pattern should work. The approach:

1. Run virtualization in parallel via Lambda (map phase -- same as today)
2. Run Icechunk writes in parallel with `rebase_with=ConflictDetector()` and
   `rebase_tries=20`
3. If a commit loses the compare-and-swap race, it rebases and retries --
   this is a fast local operation, not a full re-virtualization

**Conservative approach:** Start with sequential writes on S3 to establish a
baseline, then enable concurrent writes and measure. If concurrent writes
hit issues, we have a known-good fallback.

## Lambda Memory: Use 2048 MB

The existing GCP config uses 1024 MB. On AWS Lambda, memory allocation is
directly proportional to CPU allocation. At 1024 MB you get a fractional
vCPU; at **1769 MB** you get exactly 1 full vCPU. Rounding up to **2048 MB**
gives a full core with headroom.

The virtualization workload is I/O-bound (reading NetCDF headers from GCS
over the network), but the xarray merge and coordinate fixing are CPU-bound.
A fractional core means these CPU bursts take longer, and since Lambda bills
per-ms, a slower function can cost the same or more than a faster one on a
bigger instance. 2048 MB is the right choice.

## Prerequisites

- AWS CLI configured with credentials (`aws sts get-caller-identity` should work)
- Podman or Docker installed (Lithops supports both; podman is auto-detected)
- Python 3.12 (must match the Lambda runtime -- see below)
- `uv` for Python environment management

### Python Version Constraint

The AWS Lambda base image uses Python 3.12. Lithops serializes (pickles)
functions locally and deserializes them in Lambda, so the local Python version
**must match exactly**. The `pyproject.toml` pins `requires-python = "==3.12.*"`.

If your system default is a different Python version:
```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv sync
```

## AWS Infrastructure Setup

### 1. Create IAM Role for Lambda

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-west-2

# Create IAM role for Lambda execution
aws iam create-role \
    --role-name lithops-ismip6-executor \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --tags Key=project,Value=ismip6-indexing

# Attach basic Lambda execution policy (CloudWatch logs)
aws iam attach-role-policy \
    --role-name lithops-ismip6-executor \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### 2. Create S3 Bucket with Cost Tracking Tags

```bash
aws s3api create-bucket \
    --bucket ismip6-icechunk \
    --region $AWS_REGION \
    --create-bucket-configuration LocationConstraint=$AWS_REGION

# Tag for cost tracking
aws s3api put-bucket-tagging \
    --bucket ismip6-icechunk \
    --tagging 'TagSet=[{Key=project,Value=ismip6-indexing}]'

# Enable S3 request metrics for cost visibility
aws s3api put-bucket-metrics-configuration \
    --bucket ismip6-icechunk \
    --id EntireBucket \
    --metrics-configuration '{"Id":"EntireBucket"}'
```

### 3. Grant Lambda S3 Access

Lithops uses **two** S3 buckets at runtime:
- `ismip6-icechunk` -- the Icechunk store (our data)
- `lithops-us-west-2-<suffix>` -- auto-created by Lithops for intermediate job
  data (pickled function payloads, results, etc.)

The Lambda execution role needs access to both. The Lithops bucket suffix is
derived from the account ID; for account `429435741471` it is `lithops-us-west-2-l2ic`.
You can find yours with `aws s3api list-buckets` after the first Lithops run.

```bash
aws iam put-role-policy \
    --role-name lithops-ismip6-executor \
    --policy-name ismip6-s3-access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:*"],
            "Resource": [
                "arn:aws:s3:::ismip6-icechunk",
                "arn:aws:s3:::ismip6-icechunk/*",
                "arn:aws:s3:::lithops-us-west-2-l2ic",
                "arn:aws:s3:::lithops-us-west-2-l2ic/*"
            ]
        }]
    }'
```

**If you get `403 Forbidden` / `HeadBucket` errors in Lambda logs**, this is
almost certainly the cause -- the policy is missing the Lithops bucket. Check
`aws s3api list-buckets` for the actual bucket name and update accordingly.

### 4. Set a Billing Alarm

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name ismip6-cost-alarm \
    --alarm-description "Alert when ISMIP6 processing exceeds $50" \
    --namespace AWS/Billing \
    --metric-name EstimatedCharges \
    --dimensions Name=Currency,Value=USD \
    --statistic Maximum \
    --period 21600 \
    --threshold 50 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions <your-sns-topic-arn>
```

### 5. Lithops AWS Configuration

Create `lithops_aws.yaml`:

```yaml
lithops:
    backend: aws_lambda
    storage: aws_s3
    data_limit: False

aws:
    region: us-west-2

aws_lambda:
    execution_role: arn:aws:iam::<ACCOUNT_ID>:role/lithops-ismip6-executor
    runtime_memory: 2048
    runtime_timeout: 300
    runtime: ismip6-icechunk
    architecture: x86_64

aws_s3:
    bucket: ismip6-icechunk
    region: us-west-2
```

**No `access_key_id`/`secret_access_key` needed in the config** if your local
AWS CLI is already configured (via environment variables, `~/.aws/credentials`,
or IAM role). Lithops uses boto3's default credential chain. The GCP setup
required an explicit service account key file; AWS does not.

### 6. Build and Deploy the Lambda Runtime

See the next section for how Lithops runtimes work and what the Dockerfile needs.

```bash
# Build the container image and push to ECR
uv run lithops runtime build -f Dockerfile.lithops ismip6-icechunk -c lithops_aws.yaml

# Create the Lambda function from the ECR image
uv run lithops runtime deploy ismip6-icechunk -c lithops_aws.yaml

# Verify end-to-end
uv run lithops hello -c lithops_aws.yaml
```

Expected output: `Hello <username>! Lithops is working as expected :)`

## How Lithops Lambda Runtimes Work

Understanding the runtime model is important because it affects how you
iterate on code changes and how CI/CD could be set up.

### Two Deployment Models

Lithops supports two Lambda deployment approaches:

1. **Zip-based (default):** Bundles a small `entry_point.py` handler into a zip,
   attaches dependencies as a Lambda Layer. Limited to 250 MB unzipped -- too
   small for our scientific Python stack (numpy, scipy, netcdf4, h5netcdf, etc.).

2. **Container image (Docker/ECR):** Builds a Docker image, pushes to ECR, deploys
   Lambda with `PackageType='Image'`. No size limit (up to 10 GB). **This is
   what we use.**

### What `lithops runtime build` Does

When you run `lithops runtime build -f Dockerfile.lithops ismip6-icechunk`:

1. **Creates `lithops_lambda.zip`** in the working directory. This zip contains
   the entire `lithops` Python package (the worker code) plus `entry_point.py`
   (the Lambda handler). This is how Lithops injects itself into your image.

2. **Runs `podman build`** (or `docker build` -- auto-detected) with your
   Dockerfile. The Dockerfile must `COPY lithops_lambda.zip` and unpack it.

3. **Pushes to ECR.** Authenticates via boto3, creates the ECR repo if needed
   (named `lithops_v<version>_<suffix>/<runtime-name>`), tags and pushes.

4. **Cleans up** the temporary zip file.

### What the Dockerfile Must Include

The `Dockerfile.lithops` must:
- Start from `public.ecr.aws/lambda/python:3.12` (provides the Lambda Runtime
  Interface Client)
- Install Python deps from `requirements-lithops.txt`
- Install the `ismip6_helper` module
- `COPY` and unpack `lithops_lambda.zip` (created by Lithops before the build)
- Set `CMD ["entry_point.lambda_handler"]`

The zip unpacking step is critical and non-obvious -- without it, the Lambda
function starts but immediately fails with `exit status 142` because the
entry point isn't importable.

### Can CI/CD Build the Image?

Yes, with a clean split:
- **CI/CD** builds and pushes the image to ECR (standard Docker workflow)
- **Lithops** creates the Lambda function via `lithops runtime deploy`

The catch: CI/CD must also generate `lithops_lambda.zip` and include it in
the build context. This requires having Lithops installed in CI and calling
its internal `create_handler_zip` utility. For now, building locally via
`lithops runtime build` is simpler.

### Rebuilding After Code Changes

If you change `ismip6_helper/`, `requirements-lithops.txt`, or the Dockerfile:

```bash
uv run lithops runtime build -f Dockerfile.lithops ismip6-icechunk -c lithops_aws.yaml
uv run lithops runtime deploy ismip6-icechunk -c lithops_aws.yaml
```

Podman caches earlier layers, so rebuilds that only touch `ismip6_helper/` or
the Lithops zip are fast (the `pip install` layer is cached).

## Code Changes Required

### Update Icechunk Storage Config

In `virtualize_with_lithops_combine_variables.py`, update `get_repo_kwargs()`:

```python
def get_repo_kwargs(local_storage: bool = False) -> dict:
    if local_storage:
        storage = icechunk.local_filesystem_storage("test-output/test_icechunk")
    else:
        storage = icechunk.s3_storage(
            bucket="ismip6-icechunk",
            prefix="combined-variables-v3",
            region="us-west-2",
            from_env=True,
        )
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "gs://ismip6/",
            store=icechunk.gcs_store()
        )
    )
    # Can be more aggressive on S3 than on GCS
    config.max_concurrent_requests = 10

    credentials = icechunk.containers_credentials({
        "gs://ismip6/": None
    })
    return {
        'storage': storage,
        'config': config,
        'authorize_virtual_chunk_access': credentials
    }
```

Note: the source data still lives on `gs://ismip6` (public GCS). The virtual
chunk container still points there. Only the Icechunk metadata store moves
to S3. Cross-cloud reads of public data work fine -- there is no egress cost
for reading from a public GCS bucket.

### Accept Config File as CLI Argument

```python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lithops_aws.yaml")
    parser.add_argument("--local-storage", action="store_true")
    parser.add_argument("--local-execution", action="store_true")
    args = parser.parse_args()
    result = process_all_files(
        local_storage=args.local_storage,
        local_execution=args.local_execution,
        config_file=args.config
    )
```

### Re-enable Concurrent Writes (Phase 2)

Once sequential writes are confirmed working on S3, update the write loop:

```python
# Instead of serial loop, use lithops or concurrent.futures:
from concurrent.futures import ThreadPoolExecutor, as_completed

writing_results = []
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {
        pool.submit(batch_write_func, r['batch'], r['virtual_dataset'], repo): r
        for r in successful_results
    }
    for future in as_completed(futures):
        res = future.result()
        writing_results.append(res)
        print(f"Done {'OK' if res.get('success') else 'FAIL'} {res['batch']['path']}")
```

And in `batch_write_func`, enable rebasing:

```python
commit_id = session.commit(
    commit_msg,
    rebase_with=icechunk.ConflictDetector(),
    rebase_tries=20
)
```

### Increase Zarr Concurrency

The current config throttles zarr to work around GCS rate limits. On S3 these
can be raised:

```python
zarr.config.set({
    'async.concurrency': 20,
    'threading.max_workers': 10
})
```

## Cost Tracking

### Tagging Strategy

All AWS resources are tagged with `project=ismip6-indexing`. To view costs:

```bash
# After the run, check Lambda costs
aws ce get-cost-and-usage \
    --time-period Start=2026-01-01,End=2026-02-01 \
    --granularity DAILY \
    --metrics BlendedCost \
    --filter '{
        "Tags": {
            "Key": "project",
            "Values": ["ismip6-indexing"]
        }
    }'
```

### Estimated Per-Run Costs

| Resource | Estimate | Notes |
|----------|----------|-------|
| Lambda compute | ~$0.70 | 440 invocations x ~60s x 2048 MB |
| S3 writes | ~$0.03 | ~5,000 PUT requests |
| S3 storage | ~$0.01/month | Icechunk metadata only (small) |
| GCS egress | $0.00 | Public bucket, no egress charges |
| CloudWatch | $0.00 | Included with Lambda |
| **Total per run** | **~$1-3** | |

### Real-Time Monitoring

Lambda publishes metrics to CloudWatch automatically:
- `Invocations` -- how many functions ran
- `Duration` -- per-invocation runtime in ms
- `Errors` -- failed invocations
- `ConcurrentExecutions` -- peak parallelism

View during a run:
```bash
aws cloudwatch get-metric-statistics \
    --namespace AWS/Lambda \
    --metric-name Duration \
    --dimensions Name=FunctionName,Value=lithops-worker-l2ic-362-aa35266255 \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --period 300 \
    --statistics Average Maximum \
    --region us-west-2
```

## Execution Plan

### Phase 1: Validate Sequential Writes on S3

1. Set up AWS infrastructure (IAM, S3 bucket, billing alarm)
2. Write `lithops_aws.yaml`
3. Build and deploy the Lambda runtime
4. Update `get_repo_kwargs()` to support S3 storage
5. Run with a small test filter (single model, e.g. `fETISh_32km`)
6. Verify the Icechunk store is readable from S3
7. Check CloudWatch metrics and Cost Explorer

### Phase 2: Test Concurrent Writes

1. Re-enable `rebase_with=ConflictDetector()` in `batch_write_func`
2. Replace serial write loop with `ThreadPoolExecutor(max_workers=4)`
3. Run on test subset, verify no errors
4. Gradually increase `max_workers` to 8, then 16
5. Compare runtime vs sequential baseline

### Phase 3: Full Production Run

1. Remove test filters (`files_df` model/experiment filter on line 232-233)
2. Run full pipeline: `uv run python virtualize_with_lithops_combine_variables.py --config lithops_aws.yaml`
3. Verify all ~440 batches committed
4. Spot-check with notebooks (`notebooks/open_icechunk.ipynb`)
5. Record final cost from Cost Explorer

### Phase 4: Cleanup

1. Delete the Lambda runtime if no longer needed: `uv run lithops runtime delete ismip6-icechunk -c lithops_aws.yaml`
2. Optionally remove the Lambda execution role if done
3. Keep the S3 bucket -- it holds the Icechunk store

## Troubleshooting

### `exit status 142` on deploy

The Lambda function can't start its entry point. Check that `Dockerfile.lithops`
copies and unpacks `lithops_lambda.zip` and sets `CMD ["entry_point.lambda_handler"]`.
The zip is created by `lithops runtime build` in the working directory before
`podman build` runs -- if the Dockerfile doesn't `COPY` it, the Lambda container
will have no handler.

### `403 Forbidden` / `HeadBucket` errors in Lambda logs

The Lambda execution role doesn't have S3 access. Lithops auto-creates its own
bucket (`lithops-us-west-2-<suffix>`) for intermediate data, and the Lambda role
needs access to it in addition to `ismip6-icechunk`. Check:

```bash
# Find the Lithops bucket
aws s3api list-buckets --query 'Buckets[?contains(Name, `lithops`)].Name'

# Check Lambda logs
aws logs tail /aws/lambda/lithops-worker-l2ic-362-aa35266255 --region us-west-2
```

### Python version mismatch

Lithops enforces that local and remote Python versions match (for pickle
compatibility). If you see `"is running Python 3.12 and it is not compatible
with the local Python version 3.13"`, recreate your venv:

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv sync
```

### `lithops runtime build` fails with "docker: command not found"

Lithops auto-detects podman via `utils.get_docker_path()`. Ensure `podman` is
installed and on PATH. When using podman, Lithops automatically adds
`--format docker --remove-signatures` to push commands for ECR compatibility.
