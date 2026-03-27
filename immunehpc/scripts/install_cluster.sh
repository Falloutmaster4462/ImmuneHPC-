#!/usr/bin/env bash
# =============================================================================
# ImmuneHPC+ Install — 2-node Rocky Linux 9.7 cluster
#
# team  (172.21.12.41) — head node  — run this script HERE
# com1  (172.21.12.21) — compute    — script SSHes in automatically
#
# Run as: sudo bash install_cluster.sh
# =============================================================================
set -euo pipefail

HEAD_IP="172.21.12.41"
COM1_IP="172.21.12.21"
HEAD_HOST="team"
COM1_HOST="com1"
ALL_NODES=("$HEAD_IP" "$COM1_IP")

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
step() { echo -e "\n${BOLD}══ $* ══${NC}"; }

# Must run as root
[[ $EUID -ne 0 ]] && { echo "Run as root: sudo bash $0"; exit 1; }

echo -e "${BOLD}"
cat << 'BANNER'
  ___                            _   _ ____   ____
 |_ _|_ __ ___  _ __ ___  _   _| \ | |  _ \ / ___|  _
  | || '_ ` _ \| '_ ` _ \| | | |  \| | |_) | |     | |
  | || | | | | | | | | | | |_| | |\  |  __/| |___  |_|
 |___|_| |_| |_|_| |_| |_|\__,_|_| \_|_|    \____|  +
BANNER
echo -e "${NC}"
info "Head: $HEAD_HOST ($HEAD_IP)"
info "Compute: $COM1_HOST ($COM1_IP)"
echo ""

# =============================================================================
# STEP 1 — Fix /etc/hosts on both nodes (Slurm can't resolve 'headnode')
# =============================================================================
step "1/6  Fix /etc/hosts — repair broken Slurm DNS"

HOSTS_BLOCK="
# ImmuneHPC+ cluster nodes
$HEAD_IP  $HEAD_HOST team headnode
$COM1_IP  $COM1_HOST com1
"

# Fix on head node
if ! grep -q "headnode" /etc/hosts; then
    echo "$HOSTS_BLOCK" >> /etc/hosts
    ok "Added cluster hosts to /etc/hosts on $HEAD_HOST"
else
    # Make sure headnode alias is on the head IP line
    if ! grep "$HEAD_IP" /etc/hosts | grep -q "headnode"; then
        sed -i "/$HEAD_IP/s/$/ headnode/" /etc/hosts
    fi
    ok "/etc/hosts already has headnode entry"
fi

# Fix on com1 via SSH (use IP since DNS is broken)
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    "root@$COM1_IP" "
    grep -q 'headnode' /etc/hosts || echo '$HOSTS_BLOCK' >> /etc/hosts
    grep '$HEAD_IP' /etc/hosts | grep -q 'headnode' || \
        sed -i '/$HEAD_IP/s/\$/ headnode/' /etc/hosts
    echo 'hosts fixed on com1'
" 2>/dev/null && ok "Fixed /etc/hosts on $COM1_HOST" \
             || warn "Could not SSH to com1 as root — fix /etc/hosts manually (see below)"

# Fix slurmd — it was retrying against wrong hostname, restart to pick up new hosts
systemctl restart slurmd 2>/dev/null && ok "slurmd restarted on $HEAD_HOST" || true
ssh -o StrictHostKeyChecking=no "root@$COM1_IP" \
    "systemctl restart slurmd 2>/dev/null && echo slurmd restarted" 2>/dev/null || true

# =============================================================================
# STEP 2 — SSH key from head → com1 (needed for ImmuneHPC+ SSH collectors)
# =============================================================================
step "2/6  SSH key setup (head → com1)"

KEY="/etc/immunehpc/id_ed25519"
mkdir -p /etc/immunehpc
chmod 700 /etc/immunehpc

if [[ ! -f "$KEY" ]]; then
    ssh-keygen -t ed25519 -N "" -C "immunehpc-controller" -f "$KEY"
    chmod 600 "$KEY"; chmod 644 "$KEY.pub"
    ok "SSH keypair generated: $KEY"
else
    ok "SSH key already exists"
fi

PUB=$(cat "$KEY.pub")

# Push to both nodes
for IP in "${ALL_NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "root@$IP" "
        mkdir -p /root/.ssh
        chmod 700 /root/.ssh
        grep -qF '$(cat $KEY.pub)' /root/.ssh/authorized_keys 2>/dev/null \
            || echo '$PUB' >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        echo 'key installed'
    " 2>/dev/null && ok "SSH key installed on $IP" \
                  || warn "Could not push key to $IP — do manually: ssh-copy-id -i $KEY.pub root@$IP"
done

# Verify passwordless access
ssh -o BatchMode=yes -i "$KEY" "root@$COM1_IP" "echo ok" 2>/dev/null \
    && ok "Passwordless SSH to com1 verified" \
    || warn "Passwordless SSH not working yet — you may need to enter password once above"

# =============================================================================
# STEP 3 — Install system packages
# =============================================================================
step "3/6  System packages"

dnf install -y --quiet \
    python3 python3-pip \
    ansible \
    ipmitool \
    prometheus-node-exporter \
    audit \
    iproute-tc \
    libcgroup-tools \
    kernel-tools \
    tuned \
    jq \
    2>/dev/null

ok "System packages installed on head node"

# Install on com1
ssh -i "$KEY" "root@$COM1_IP" "
    dnf install -y --quiet \
        python3 \
        prometheus-node-exporter \
        audit \
        iproute-tc \
        libcgroup-tools \
        kernel-tools \
        tuned \
        2>/dev/null
    systemctl enable --now prometheus-node-exporter
    systemctl enable --now auditd
    systemctl enable --now tuned
    echo 'com1 packages done'
" && ok "Packages installed on com1" || warn "Package install on com1 failed"

# Start services on head node
systemctl enable --now prometheus-node-exporter audit tuned 2>/dev/null || true
ok "Services started on head node"

# =============================================================================
# STEP 4 — Install ImmuneHPC+
# =============================================================================
step "4/6  Install ImmuneHPC+"

INSTALL_DIR="/opt/immunehpc"

# Copy source (assumes you unpacked the zip here)
if [[ -d "immunehpc-real" ]]; then
    rsync -a --exclude '__pycache__' --exclude '*.pyc' \
        immunehpc-real/ "$INSTALL_DIR/"
    ok "Source copied to $INSTALL_DIR"
elif [[ -d "/root/immunehpc-real" ]]; then
    rsync -a --exclude '__pycache__' --exclude '*.pyc' \
        /root/immunehpc-real/ "$INSTALL_DIR/"
    ok "Source copied to $INSTALL_DIR"
else
    warn "immunehpc-real/ directory not found — copy the zip here first:"
    warn "  scp immunehpc-full-autonomic.zip root@$HEAD_IP:/root/"
    warn "  unzip immunehpc-full-autonomic.zip"
    warn "  bash immunehpc-real/scripts/install_cluster.sh"
    exit 1
fi

# Python virtualenv
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet pyyaml
ok "Python venv ready"

# Logs dir
mkdir -p "$INSTALL_DIR/logs"
chmod 755 "$INSTALL_DIR/logs"

# =============================================================================
# STEP 5 — Write production config for this cluster
# =============================================================================
step "5/6  Write production config"

CONFIG="/etc/immunehpc/production.yaml"

cat > "$CONFIG" << YAML
# ImmuneHPC+ — auto-generated for your Rocky 9.7 cluster
# team (172.21.12.41) + com1 (172.21.12.21)

system:
  name: ImmuneHPC+
  loop_interval_sec: 10
  log_level: INFO
  audit_log: /opt/immunehpc/logs/audit.jsonl

ssh:
  user: root
  key_path: /etc/immunehpc/id_ed25519

cluster:
  nodes:
    - 172.21.12.41   # team  (head node — monitors itself too)
    - 172.21.12.21   # com1

monitor:
  telemetry_interval_sec: 10
  health_threshold: 0.6
  node_exporter_port: 9100
  use_ipmi: false          # no BMC/IPMI on this hardware
  use_gpu: false           # no GPUs detected
  use_slurm: true          # slurmd is running

anomaly:
  method: hybrid
  window_size: 60
  z_score_threshold: 3.0
  sensitivity: 0.85

quarantine:
  auto_isolate: true
  max_quarantine_sec: 300

healing:
  strategy_order:
    - restart_service
    - reapply_config
    - rollback
    - reimage_node
  max_attempts: 2
  backoff_sec: 20
  ansible_playbook: /opt/immunehpc/ansible/site.yml
  ansible_inventory: /opt/immunehpc/ansible/inventory.ini
  cobbler_host: ""         # no PXE server — reimage disabled
  bmc_user: ""
  bmc_password: ""

optimizer:
  objectives:
    alpha: 0.25
    beta:  0.25
    gamma: 0.25
    delta: 0.25
  interval_sec: 60
  network_interface: eno1
  network_wire_speed_gbit: 1

defense:
  auto_mitigate: true
  known_keys_path: /etc/immunehpc/known_host_keys.json

asl:
  enabled: true
  patch_trust_threshold: 0.75
  sandbox_timeout_sec: 60
  canary_fraction: 0.50    # 1 of 2 nodes = 50% canary
  canary_soak_sec: 30      # shorter for 2-node cluster
  code_gen_interval: 20

  # LLM for delta-C code generation
  # Using Ollama with CPU-friendly model (no GPU needed)
  llm:
    provider: auto
    ollama_host: http://localhost:11434
    ollama_model: qwen2.5-coder:7b    # ~4GB RAM, runs on CPU

    # Free cloud fallback — add keys when you get them
    groq_api_key: ""       # console.groq.com — free, no card
    google_api_key: ""     # aistudio.google.com — free
    openrouter_api_key: "" # openrouter.ai — free community models

    timeout_sec: 60
    max_tokens: 1200
    temperature: 0.15

  rl:
    learning_rate: 0.001
    discount_factor: 0.95
    epsilon: 0.1

prometheus:
  pushgateway_url: ""
YAML

chmod 640 "$CONFIG"
ok "Config written to $CONFIG"

# Write Ansible inventory for this cluster
cat > "$INSTALL_DIR/ansible/inventory.ini" << INV
[hpc_nodes]
172.21.12.41 ansible_host=172.21.12.41
172.21.12.21 ansible_host=172.21.12.21

[hpc_nodes:vars]
ansible_user=root
ansible_ssh_private_key_file=/etc/immunehpc/id_ed25519
ansible_ssh_common_args=-o StrictHostKeyChecking=no

[controller]
localhost ansible_connection=local
INV
ok "Ansible inventory written"

# =============================================================================
# STEP 6 — Systemd service
# =============================================================================
step "6/6  systemd service"

cat > /etc/systemd/system/immunehpc.service << 'SVC'
[Unit]
Description=ImmuneHPC+ Autonomous HPC Controller
After=network-online.target slurmd.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/immunehpc
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/opt/immunehpc
EnvironmentFile=-/etc/immunehpc/environment
ExecStart=/opt/immunehpc/venv/bin/python3 /opt/immunehpc/main.py \
    --config /etc/immunehpc/production.yaml
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=immunehpc

[Install]
WantedBy=multi-user.target
SVC

# Environment file for secrets
if [[ ! -f /etc/immunehpc/environment ]]; then
    cat > /etc/immunehpc/environment << 'ENV'
# ImmuneHPC+ secrets
# Add API keys here (never in the YAML)
GROQ_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
ENV
    chmod 600 /etc/immunehpc/environment
fi

systemctl daemon-reload
systemctl enable immunehpc
ok "systemd service installed and enabled"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}╔════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ImmuneHPC+ installed successfully!        ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Start it:${NC}"
echo "  systemctl start immunehpc"
echo "  journalctl -u immunehpc -f"
echo ""
echo -e "${BOLD}Optional — add free LLM (Ollama, no GPU needed):${NC}"
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo "  ollama pull qwen2.5-coder:7b"
echo "  # Then restart: systemctl restart immunehpc"
echo ""
echo -e "${BOLD}Optional — add free cloud LLM (Groq, no card):${NC}"
echo "  # Sign up at console.groq.com → API Keys"
echo "  echo 'GROQ_API_KEY=gsk_...' >> /etc/immunehpc/environment"
echo "  systemctl restart immunehpc"
echo ""
echo -e "${BOLD}Check status:${NC}"
echo "  bash /opt/immunehpc/scripts/immunectl.sh status"
echo "  bash /opt/immunehpc/scripts/immunectl.sh nodes"
echo ""
echo -e "${BOLD}Also fix the Slurm slurmctld hostname if not done yet:${NC}"
echo "  grep ControlMachine /etc/slurm/slurm.conf"
echo "  # Should say: ControlMachine=team  (or headnode if you kept that name)"
echo "  # If wrong: sed -i 's/ControlMachine=.*/ControlMachine=team/' /etc/slurm/slurm.conf"
echo "  # Then: systemctl restart slurmctld slurmd"
echo ""
