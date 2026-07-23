#!/usr/bin/env bash
set -euo pipefail

# This script predates the S8 target-provider and S10 authentication/isolation
# work. It creates an unauthenticated public LoadBalancer and does not define the
# privileged Kubernetes securityContext required by the debugger. Keep it
# fail-closed unless an operator explicitly acknowledges those limitations.
LEGACY_UNSAFE_ACK="I_UNDERSTAND_PAO_LEGACY_PUBLIC_DEPLOY_IS_UNSAFE"
if [[ "${PAO_ACK_LEGACY_UNSAFE_PUBLIC_DEPLOY:-}" != "$LEGACY_UNSAFE_ACK" ]]; then
  cat >&2 <<EOF
ERROR: deploy.sh is a deprecated, unsafe legacy deployment path and is disabled.

It publishes the unauthenticated MCP endpoint through a public LoadBalancer and
does not configure the debugger privileges/isolation required for a supported
Kubernetes runtime. Complete the S8 target-provider and S10 authentication/
isolation design before using this path in production.

For a disposable, network-isolated test only, acknowledge the risk explicitly:
  export PAO_ACK_LEGACY_UNSAFE_PUBLIC_DEPLOY=$LEGACY_UNSAFE_ACK
EOF
  exit 64
fi

cat >&2 <<'EOF'
WARNING: proceeding with the deprecated legacy public deployment.
The MCP endpoint has no authentication, and this script intentionally does not
grant SYS_PTRACE/SYS_ADMIN or an unconfined security profile. Debugger operations
may fail. Do not use this path for production or on an untrusted network.
EOF

CLUSTER=${PAO_CLUSTER:-pao-cluster}
REGION=${PAO_REGION:-us-central1}
PROJECT=${PAO_GCP_PROJECT:?Set PAO_GCP_PROJECT to your Google Cloud project id}
IMAGE=${PAO_IMAGE:?Set PAO_IMAGE to the container image to deploy}
DEPLOY=${PAO_DEPLOYMENT:-pao-tools}
SERVICE=${PAO_SERVICE:-pao-tools-lb}

echo "Fetching credentials for $CLUSTER"
gcloud container clusters get-credentials "$CLUSTER" \
    --region "$REGION" \
    --project "$PROJECT"

if kubectl get deployment "$DEPLOY" &>/dev/null; then
  echo "Updating $DEPLOY → $IMAGE"
  kubectl set image deployment/"$DEPLOY" "$DEPLOY"="$IMAGE" --record
else
  echo "Creating deployment $DEPLOY"
  kubectl create deployment "$DEPLOY" --image="$IMAGE"
fi

kubectl set resources deployment/"$DEPLOY" \
  --containers="$DEPLOY" \
  --requests=cpu=500m,memory=512Mi \
  --limits=cpu=500m,memory=512Mi

kubectl patch deployment "$DEPLOY" --type='strategic' -p "
spec:
  template:
    spec:
      containers:
      - name: $DEPLOY
        ports:
        - containerPort: 5500
        readinessProbe:
          httpGet:
            path: /healthz
            port: 5500
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /healthz
            port: 5500
          initialDelaySeconds: 10
          periodSeconds: 20
"

if kubectl get svc "$SERVICE" &>/dev/null; then
  echo "Service $SERVICE already exists"
else
  echo "Exposing $DEPLOY as LoadBalancer → $SERVICE"
  kubectl expose deployment "$DEPLOY" \
    --type=LoadBalancer \
    --name="$SERVICE" \
    --port=5500 \
    --target-port=5500
fi

echo -n "Waiting for EXTERNAL-IP"
while :; do
  IP=$(kubectl get svc "$SERVICE" \
       -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  if [[ -n "$IP" ]]; then
    echo -e "\nlive at http://$IP:5500/mcp"
    break
  fi
  echo -n "."
  sleep 2
done
