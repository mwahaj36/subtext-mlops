# Subtext Course Serving Runbook (AWS ECR + EC2 + Isolated Vercel Frontend)

## 1. Environment Isolation
- Every AWS resource uses tag `Project=subtext-mlops-course` in region `us-east-1`.
- Database state stays inside the isolated `postgres:16-alpine` container (`subtext_course`) with zero credentials or links to any external Neon or AstraDB database.
- Separate Vercel project `subtext-mlops-course` connects only via `NEXT_PUBLIC_API_URL=https://<API_DOMAIN>` and `CORS_ALLOW_ORIGINS`.

## 2. Digest-Pinned Build & Push (ECR)
```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/subtext-mlops-serve"

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker build -f Dockerfile.serve -t "${ECR_REPO}:d2-serve" .
docker push "${ECR_REPO}:d2-serve"

export IMAGE_DIGEST=$(aws ecr describe-images \
  --repository-name subtext-mlops-serve \
  --image-ids imageTag=d2-serve \
  --query 'imageDetails[0].imageDigest' --output text)
echo "Pinned image: ${ECR_REPO}@${IMAGE_DIGEST}"
```

## 3. Startup, Persistence & Restart on EC2
Place `.env.serve` (course-only `LLM_PROVIDER_BASE_URL`, `LLM_MODEL_ID`, `LLM_API_KEY`, `CORS_ALLOW_ORIGINS`) and `.env` (`SERVE_IMAGE`, `API_DOMAIN`, `POSTGRES_PASSWORD`) in `~/subtext-deploy/`:
```bash
cd ~/subtext-deploy
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com"
docker compose -f docker-compose.prod.yml pull serve
docker compose -f docker-compose.prod.yml up -d
```
- **Persistence:** Named Docker volumes `postgres_data`, `caddy_data`, and `caddy_config` preserve TLS certificates and demo DB state across container restarts.
- **Restart & Rollback:** `restart: unless-stopped` recovers automatically on reboot. To roll back immediately to a prior digest:
  ```bash
  SERVE_IMAGE="${ECR_REPO}@sha256:<previous_digest>" docker compose -f docker-compose.prod.yml up -d --no-deps serve
  ```
- **Verification:** `curl -fsS "https://${API_DOMAIN}/ready"` and `curl -fsS "https://${API_DOMAIN}/version"`.

## 4. Cost Bounds & Full Teardown
- AWS Budgets cap configured at **$10/month** (50% / 80% / 100% alerts); EC2 `t3.micro` with 2 GB swap runs under Free Tier (or stop instance when idle to incur $0/hr compute).
- Full course teardown (leaves unrelated projects untouched):
  ```bash
  aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=subtext-mlops-course" "Name=instance-state-name,Values=running,stopped" \
    --query "Reservations[].Instances[].InstanceId" --output text \
    | xargs -r aws ec2 terminate-instances --instance-ids
  aws ecr delete-repository --repository-name subtext-mlops-serve --force --region us-east-1
  aws s3 rb s3://subtext-mlops-course-bucket --force --region us-east-1
  ```
- Delete the separate `subtext-mlops-course` Vercel project in Vercel Dashboard and revoke the course LLM API key.
