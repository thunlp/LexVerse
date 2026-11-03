#!/bin/sh
# Derived from Harbor (Apache-2.0), reference revision 71c77fdd.
# Modified by LexVerse contributors for task-scoped Docker network control.
set -eu

# GOST_CONFIG/ALLOWLIST must stay in sync with gost.yaml
# and bin/network-policy; READY_FILE with the healthcheck in
# docker-compose-egress-control.yaml.
GOST_CONFIG=/opt/egress-sidecar/gost.yaml
ALLOWLIST=/opt/egress-sidecar/allowlist.txt
READY_FILE=/tmp/harbor-docker-egress-control-sidecar.ready

INITIAL_NETWORK_MODE="${EGRESS_CONTROL_INITIAL_NETWORK_MODE:-public}"
INITIAL_ALLOWED_HOSTS="${EGRESS_CONTROL_INITIAL_ALLOWED_HOSTS:-}"

cleanup() {
  rm -f "$READY_FILE"
  # Keep enforcement installed if the proxy exits while an agent still runs.
  # Redirected connections then fail instead of falling back to public access.
}

mkdir -p "$(dirname "$ALLOWLIST")"
: > "$ALLOWLIST"
cleanup

trap cleanup INT TERM EXIT

/bin/gost -C "$GOST_CONFIG" &
gost_pid="$!"

case "$INITIAL_NETWORK_MODE" in
  public)
    network-policy allow-all
    ;;
  no-network)
    network-policy deny-all
    ;;
  allowlist)
    set -f
    # shellcheck disable=SC2086
    network-policy allow $INITIAL_ALLOWED_HOSTS
    set +f
    ;;
  *)
    echo "invalid EGRESS_CONTROL_INITIAL_NETWORK_MODE: $INITIAL_NETWORK_MODE" >&2
    exit 2
    ;;
esac

touch "$READY_FILE"
echo "harbor-docker-egress-control-sidecar ready: initial egress policy is $INITIAL_NETWORK_MODE"

wait "$gost_pid"
