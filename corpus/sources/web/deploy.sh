#!/usr/bin/env bash
# Roll a service to a new image tag and wait for it to become healthy.
set -Eeuo pipefail

NAMESPACE="${NAMESPACE:-prod}"
TIMEOUT="${TIMEOUT:-300}"
usage() { printf 'usage: %s <service> <tag>\n' "${0##*/}" >&2; exit 2; }

[[ $# -eq 2 ]] || usage
service="$1"; tag="$2"

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }
cleanup() { log "rolling back ${service}"; kubectl -n "$NAMESPACE" rollout undo "deploy/${service}" || true; }

current=$(kubectl -n "$NAMESPACE" get "deploy/${service}" -o jsonpath='{.spec.template.spec.containers[0].image}')
log "current image: ${current}"
trap cleanup ERR

kubectl -n "$NAMESPACE" set image "deploy/${service}" "${service}=registry.internal/${service}:${tag}" --record
if ! kubectl -n "$NAMESPACE" rollout status "deploy/${service}" --timeout="${TIMEOUT}s"; then
  log "rollout did not converge in ${TIMEOUT}s"
  exit 1
fi
trap - ERR

for _ in $(seq 1 10); do
  if curl -fsS --max-time 5 "http://${service}.${NAMESPACE}.svc/healthz" >/dev/null; then
    log "healthy on ${tag}"; exit 0
  fi
  sleep 3
done
log "health check never passed"; cleanup; exit 1
