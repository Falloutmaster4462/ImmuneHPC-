#!/usr/bin/env bash
# scripts/setup_node.sh — Prepare a Rocky Linux node for ImmuneHPC+ management
#
# Run from the controller after install.sh:
#   bash scripts/setup_node.sh node01 node02 node03
#
# What this does on each node (via SSH):
#   - Installs prometheus-node-exporter, auditd, ipmitool
#   - Opens firewall ports (9100 for node_exporter, Slurm ports)
#   - Enables and starts auditd and node_exporter
#   - Sets SELinux to enforcing
#   - Deploys HPC kernel sysctl tuning
#   - Verifies SSH connectivity and prints fingerprint

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

info()    { echo -e "${BLUE}[${NODE}]${NC}  $*"; }
success() { echo -e "${GREEN}[${NODE}]${NC}  ✓ $*"; }
warn()    { echo -e "${YELLOW}[${NODE}]${NC}  ⚠ $*"; }
error()   { echo -e "${RED}[${NODE}]${NC}  ✗ $*"; }

KEY_PATH="${SSH_KEY:-/etc/immunehpc/id_ed25519}"
SSH_USER="${SSH_USER:-root}"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"
[[ -f "$KEY_PATH" ]] && SSH_OPTS="$SSH_OPTS -i $KEY_PATH"

_ssh()  { ssh  $SSH_OPTS "${SSH_USER}@${NODE}" "$@"; }
_scp()  { scp  $SSH_OPTS "$@"; }

setup_node() {
    NODE="$1"

    echo -e "\n${BOLD}── Setting up node: $NODE ──${NC}"

    # Connectivity check
    if ! _ssh "echo ok" &>/dev/null; then
        error "Cannot reach $NODE via SSH — skipping"
        return 1
    fi
    success "SSH connectivity OK"

    # Print host key fingerprint (for TOFU verification)
    FP=$(ssh-keyscan -t ed25519 "$NODE" 2>/dev/null | ssh-keygen -lf - 2>/dev/null | awk '{print $2}')
    info "Host key fingerprint: ${FP:-unknown}"

    # ── OS check ──────────────────────────────────────────────────────
    OS=$(_ssh "grep -oP '(?<=^ID=).*' /etc/os-release | tr -d '\"'" 2>/dev/null || echo "unknown")
    info "OS: $OS"

    # ── Install packages ──────────────────────────────────────────────
    info "Installing packages..."
    _ssh "
        dnf install -y epel-release 2>/dev/null || true
        dnf install -y \
            prometheus-node-exporter \
            audit \
            ipmitool \
            python3 \
            numactl \
            hwloc \
            2>/dev/null
    "
    success "Packages installed"

    # ── SELinux ───────────────────────────────────────────────────────
    SEL=$(_ssh "getenforce 2>/dev/null" || echo "unknown")
    info "SELinux: $SEL"
    if [[ "$SEL" != "Enforcing" ]]; then
        _ssh "setenforce 1 2>/dev/null || true"
        _ssh "sed -i 's/^SELINUX=.*/SELINUX=enforcing/' /etc/selinux/config"
        success "SELinux set to enforcing"
    fi

    # ── node_exporter ─────────────────────────────────────────────────
    info "Enabling prometheus-node-exporter..."
    _ssh "
        systemctl enable --now prometheus-node-exporter
        systemctl is-active prometheus-node-exporter
    "
    success "node_exporter running on port 9100"

    # ── auditd ────────────────────────────────────────────────────────
    info "Enabling auditd..."
    _ssh "
        systemctl enable --now auditd
        cat > /etc/audit/rules.d/immunehpc.rules << 'AUDITEOF'
# ImmuneHPC+ audit rules
-w /etc/passwd -p wa -k identity
-w /etc/sudoers -p wa -k sudoers
-a always,exit -F arch=b64 -S setuid -k privilege_escalation
-a always,exit -F arch=b64 -S setgid -k privilege_escalation
-w /usr/bin/su -p x -k su_execution
-w /usr/bin/sudo -p x -k sudo_execution
AUDITEOF
        augenrules --load 2>/dev/null || service auditd reload
    "
    success "auditd running with ImmuneHPC rules"

    # ── firewalld ─────────────────────────────────────────────────────
    info "Configuring firewall..."
    _ssh "
        systemctl enable --now firewalld
        # node_exporter
        firewall-cmd --add-port=9100/tcp --permanent --quiet
        # Slurm
        firewall-cmd --add-port=6817/tcp --permanent --quiet
        firewall-cmd --add-port=6818/tcp --permanent --quiet
        firewall-cmd --add-port=6819/tcp --permanent --quiet
        firewall-cmd --reload --quiet
    "
    success "Firewall configured"

    # ── HPC kernel tuning ─────────────────────────────────────────────
    info "Applying HPC kernel parameters..."
    _ssh "
        cat > /etc/sysctl.d/99-immunehpc-hpc.conf << 'SYSCTLEOF'
# ImmuneHPC+ HPC tuning
vm.swappiness = 10
vm.nr_hugepages = 1024
kernel.numa_balancing = 0
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 30000
net.ipv4.tcp_timestamps = 0
SYSCTLEOF
        sysctl --system -q
    "
    success "Kernel parameters applied"

    # ── SSH hardening ─────────────────────────────────────────────────
    info "Hardening SSH..."
    _ssh "
        sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
        sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
        sed -i 's/^#\?X11Forwarding.*/X11Forwarding no/' /etc/ssh/sshd_config
        systemctl reload sshd
    "
    success "SSH hardened (password auth disabled)"

    # ── Verify node_exporter reachable from controller ────────────────
    if command -v curl &>/dev/null; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            "http://${NODE}:9100/metrics" --connect-timeout 5 2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" == "200" ]]; then
            success "node_exporter reachable: http://${NODE}:9100/metrics"
        else
            warn "node_exporter not reachable from controller (HTTP $HTTP_CODE) — check firewall"
        fi
    fi

    success "Node $NODE is ready for ImmuneHPC+ management"
}

# ── Main ──────────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <node1> [node2] [node3] ..."
    echo ""
    echo "Environment variables:"
    echo "  SSH_KEY=/path/to/key   (default: /etc/immunehpc/id_ed25519)"
    echo "  SSH_USER=root          (default: root)"
    exit 1
fi

FAILED=()
for NODE_ARG in "$@"; do
    NODE="$NODE_ARG"
    setup_node "$NODE_ARG" || FAILED+=("$NODE_ARG")
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All nodes configured successfully.${NC}"
    echo ""
    echo "Next: start the controller:"
    echo "  systemctl start immunehpc"
    echo "  journalctl -u immunehpc -f"
else
    echo -e "${RED}${BOLD}Failed nodes:${NC} ${FAILED[*]}"
    echo "Fix SSH access or network issues and re-run for failed nodes."
    exit 1
fi
