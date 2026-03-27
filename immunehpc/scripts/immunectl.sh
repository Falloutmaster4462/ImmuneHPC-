#!/usr/bin/env bash
# scripts/immunectl.sh — ImmuneHPC+ operations CLI
#
# Usage:
#   immunectl status              — cluster status report
#   immunectl nodes               — list all nodes + health
#   immunectl drain  <node>       — gracefully drain and quarantine a node
#   immunectl release <node>      — release a node from quarantine
#   immunectl heal    <node>      — manually trigger healing on a node
#   immunectl logs    [node]      — tail controller or node logs
#   immunectl metrics             — dump metrics as JSON
#   immunectl audit   [--last N]  — tail audit log
#   immunectl inject  <node> <fault>  — fault injection (testing only)

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/immunehpc}"
CONFIG_DIR="${CONFIG_DIR:-/etc/immunehpc}"
KEY_PATH="${SSH_KEY:-$CONFIG_DIR/id_ed25519}"
SSH_USER="${SSH_USER:-root}"
PYTHON="$INSTALL_DIR/venv/bin/python3"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

_ssh() {
    local host="$1"; shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
        -i "$KEY_PATH" "${SSH_USER}@${host}" "$@"
}

cmd_status() {
    echo -e "${BOLD}ImmuneHPC+ Status${NC}"
    systemctl is-active immunehpc && echo -e "  Service: ${GREEN}running${NC}" \
        || echo -e "  Service: ${RED}stopped${NC}"
    echo ""
    "$PYTHON" "$INSTALL_DIR/main.py" \
        --config "$CONFIG_DIR/production.yaml" --status 2>/dev/null
}

cmd_nodes() {
    echo -e "${BOLD}Cluster Nodes${NC}"
    # Read node list from config
    NODES=$(python3 -c "
import yaml
with open('$CONFIG_DIR/production.yaml') as f:
    c = yaml.safe_load(f)
print('\n'.join(c['cluster']['nodes']))
" 2>/dev/null)

    printf "  %-30s %-12s %-8s %-8s\n" "NODE" "STATUS" "CPU" "TEMP"
    printf "  %-30s %-12s %-8s %-8s\n" "----" "------" "---" "----"

    for node in $NODES; do
        DATA=$(_ssh "$node" "
python3 -c \"
import json, time
def cpu():
    def r():
        with open('/proc/stat') as f: p = f.readline().split()
        v=[int(x) for x in p[1:]]; return v[3]+v[4], sum(v)
    i1,t1=r(); time.sleep(0.3); i2,t2=r()
    dt=t2-t1; return 0 if dt==0 else round((1-(i2-i1)/dt)*100,1)
def temp():
    import os, glob
    for p in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
        try: return round(int(open(p).read())/1000,1)
        except: pass
    return 'N/A'
print(json.dumps({'cpu': cpu(), 'temp': temp()}))
\"" 2>/dev/null || echo '{"cpu":"ERR","temp":"ERR"}')

        CPU=$(echo "$DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(str(d.get('cpu','?'))+'%')" 2>/dev/null || echo "ERR")
        TEMP=$(echo "$DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(str(d.get('temp','?'))+'°C')" 2>/dev/null || echo "ERR")

        if [[ "$CPU" == "ERR" ]]; then
            STATUS="${RED}UNREACHABLE${NC}"
        else
            STATUS="${GREEN}HEALTHY${NC}"
        fi

        printf "  %-30s " "$node"
        printf "${STATUS}"
        printf " %-8s %-8s\n" "$CPU" "$TEMP"
    done
}

cmd_drain() {
    local node="${1:-}"
    [[ -z "$node" ]] && { echo "Usage: immunectl drain <node>"; exit 1; }
    echo -e "${YELLOW}Draining node: $node${NC}"
    # Remove from Slurm scheduling
    _ssh "$node" "scontrol update NodeName=$node State=DRAIN Reason='ImmuneHPC+ manual drain'" \
        2>/dev/null || warn "scontrol not available on node"
    # Signal controller via sentinel file
    echo "$node" >> "$INSTALL_DIR/config/.quarantine_queue"
    echo -e "${GREEN}Node $node drained. Controller will quarantine on next tick.${NC}"
}

cmd_release() {
    local node="${1:-}"
    [[ -z "$node" ]] && { echo "Usage: immunectl release <node>"; exit 1; }
    echo -e "${YELLOW}Releasing node: $node${NC}"
    _ssh "$node" "scontrol update NodeName=$node State=RESUME" 2>/dev/null || true
    # Remove from quarantine sentinel
    sed -i "/^${node}$/d" "$INSTALL_DIR/config/.quarantine_queue" 2>/dev/null || true
    echo -e "${GREEN}Node $node released.${NC}"
}

cmd_heal() {
    local node="${1:-}"
    [[ -z "$node" ]] && { echo "Usage: immunectl heal <node>"; exit 1; }
    echo -e "${YELLOW}Triggering manual heal for: $node${NC}"
    echo "$node" >> "$INSTALL_DIR/config/.heal_queue"
    echo "Written to heal queue. Check logs: journalctl -u immunehpc -f"
}

cmd_logs() {
    local node="${1:-}"
    if [[ -z "$node" ]]; then
        journalctl -u immunehpc -f --no-pager
    else
        echo -e "${BOLD}Logs from $node:${NC}"
        _ssh "$node" "journalctl -n 100 --no-pager"
    fi
}

cmd_metrics() {
    echo -e "${BOLD}ImmuneHPC+ Metrics${NC}"
    if [[ -f "$INSTALL_DIR/logs/audit.jsonl" ]]; then
        echo -e "\nRecent events (last 10):"
        tail -10 "$INSTALL_DIR/logs/audit.jsonl" | python3 -c "
import json, sys, datetime
for line in sys.stdin:
    try:
        e = json.loads(line)
        ts = datetime.datetime.fromtimestamp(e['ts']).strftime('%H:%M:%S')
        print(f\"  {ts}  {e['event']:<35}  src={e['source']}\")
    except: pass
"
    fi
}

cmd_audit() {
    local n="${2:-50}"
    echo -e "${BOLD}ImmuneHPC+ Audit Log (last $n events)${NC}"
    if [[ -f "$INSTALL_DIR/logs/audit.jsonl" ]]; then
        tail -"$n" "$INSTALL_DIR/logs/audit.jsonl" | python3 -c "
import json, sys, datetime
for line in sys.stdin:
    try:
        e = json.loads(line)
        ts = datetime.datetime.fromtimestamp(e['ts']).strftime('%Y-%m-%d %H:%M:%S')
        print(f\"{ts}  {e['event']:<40}  {e['source']}\")
    except: pass
"
    else
        echo "Audit log not found: $INSTALL_DIR/logs/audit.jsonl"
    fi
}

cmd_inject() {
    local node="${1:-}"; local fault="${2:-}"
    [[ -z "$node" || -z "$fault" ]] && {
        echo "Usage: immunectl inject <node> <cpu|thermal|kill|trust|port>"
        exit 1
    }
    echo -e "${RED}[FAULT INJECT]${NC} type=$fault node=$node"
    case "$fault" in
        cpu)
            _ssh "$node" "nohup stress-ng --cpu 0 --timeout 30s &>/dev/null &"
            echo "CPU spike running for 30s on $node"
            ;;
        thermal)
            _ssh "$node" "
                nohup stress-ng --cpu 0 --timeout 60s &>/dev/null &
                ipmitool raw 0x30 0x70 0x66 0x01 0x00 0x00 2>/dev/null || true"
            echo "Thermal stress running on $node"
            ;;
        kill)
            _ssh "$node" "systemctl stop slurmd 2>/dev/null || true"
            echo "slurmd stopped on $node"
            ;;
        trust)
            # Simulate trust degradation by opening a suspicious listener
            _ssh "$node" "ncat -l 19999 &>/dev/null & echo \$! > /tmp/immunehpc_test_listener.pid"
            echo "Unknown listener opened on $node:19999 (trust will degrade)"
            ;;
        port)
            _ssh "$node" "python3 -m http.server 18080 &>/dev/null & echo \$! > /tmp/immunehpc_test_http.pid"
            echo "Unexpected HTTP server opened on $node:18080"
            ;;
        *)
            echo "Unknown fault: $fault"
            exit 1
            ;;
    esac
}

warn() { echo -e "${YELLOW}WARN:${NC} $*"; }

# ── Dispatch ──────────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "$CMD" in
    status)   cmd_status ;;
    nodes)    cmd_nodes ;;
    drain)    cmd_drain "$@" ;;
    release)  cmd_release "$@" ;;
    heal)     cmd_heal "$@" ;;
    logs)     cmd_logs "$@" ;;
    metrics)  cmd_metrics ;;
    audit)    cmd_audit "$@" ;;
    inject)   cmd_inject "$@" ;;
    help|--help|-h)
        echo "ImmuneHPC+ Operations CLI"
        echo ""
        echo "Usage: immunectl <command> [args]"
        echo ""
        echo "Commands:"
        echo "  status              — controller status + cluster snapshot"
        echo "  nodes               — list all nodes with live CPU/temp"
        echo "  drain  <node>       — drain and quarantine a node"
        echo "  release <node>      — release a node from quarantine"
        echo "  heal    <node>      — manually trigger healing"
        echo "  logs    [node]      — tail controller or node logs"
        echo "  metrics             — dump metrics + recent events"
        echo "  audit   [--last N]  — show audit trail"
        echo "  inject  <node> <fault>  — fault injection (testing)"
        ;;
    *)
        echo "Unknown command: $CMD. Run 'immunectl help'."
        exit 1
        ;;
esac
