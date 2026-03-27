#!/usr/bin/env bash
# scripts/install.sh — ImmuneHPC+ installer for Rocky Linux 8/9
#
# Run as root on the controller node:
#   curl -fsSL https://your-repo/install.sh | bash
#   -- or --
#   bash scripts/install.sh
#
# What this does:
#   1. Installs system dependencies (python3, ansible, ipmitool, etc.)
#   2. Creates immunehpc system user + directories
#   3. Generates SSH keypair for node access
#   4. Installs Python packages into a virtualenv
#   5. Installs and enables the systemd service
#   6. Prints next steps

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()    { echo -e "\n${BOLD}══ $* ${NC}"; }

# ── Sanity checks ─────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root."
    exit 1
fi

if ! grep -qiE "rocky|rhel|almalinux|centos" /etc/os-release 2>/dev/null; then
    warn "This script targets Rocky Linux / RHEL. Proceeding anyway..."
fi

INSTALL_DIR="${INSTALL_DIR:-/opt/immunehpc}"
CONFIG_DIR="${CONFIG_DIR:-/etc/immunehpc}"
LOG_DIR="${LOG_DIR:-/var/log/immunehpc}"
SERVICE_USER="${SERVICE_USER:-immunehpc}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "\n${BOLD}╔══════════════════════════════════════════════╗"
echo    "║  ImmuneHPC+ — Rocky Linux Installer          ║"
echo -e "╚══════════════════════════════════════════════╝${NC}\n"
info "Install dir:  $INSTALL_DIR"
info "Config dir:   $CONFIG_DIR"
info "Log dir:      $LOG_DIR"
info "Service user: $SERVICE_USER"

# ── Step 1: System packages ───────────────────────────────────────────────
step "1/7  Installing system packages"

dnf install -y epel-release 2>/dev/null || true
dnf install -y \
    python3 python3-pip python3-venv \
    ansible \
    ipmitool \
    openssh-clients \
    sshpass \
    jq \
    git \
    gcc python3-devel \
    2>/dev/null

success "System packages installed"

# ── Step 2: Create user and directories ──────────────────────────────────
step "2/7  Creating service user and directories"

if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --shell /sbin/nologin \
            --home-dir "$INSTALL_DIR" \
            --comment "ImmuneHPC+ service account" \
            "$SERVICE_USER"
    success "Created user: $SERVICE_USER"
else
    info "User $SERVICE_USER already exists"
fi

install -d -o root      -g root         -m 755 "$INSTALL_DIR"
install -d -o root      -g root         -m 755 "$INSTALL_DIR/config"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 750 "$INSTALL_DIR/logs"
install -d -o root      -g root         -m 755 "$CONFIG_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 700 "$LOG_DIR"

success "Directories created"

# ── Step 3: Copy source files ─────────────────────────────────────────────
step "3/7  Installing source files"

rsync -a --exclude '__pycache__' --exclude '*.pyc' \
    "$SOURCE_DIR/" "$INSTALL_DIR/"

chown -R root:"$SERVICE_USER" "$INSTALL_DIR"
chmod -R o-rwx "$INSTALL_DIR"
chmod -R g+r   "$INSTALL_DIR"
# Logs directory writable by service user
chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR/logs"

success "Source files installed to $INSTALL_DIR"

# ── Step 4: Python virtualenv ─────────────────────────────────────────────
step "4/7  Setting up Python virtualenv"

python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"

success "Python virtualenv ready: $INSTALL_DIR/venv"

# ── Step 5: SSH keypair ───────────────────────────────────────────────────
step "5/7  SSH keypair for node access"

KEY_PATH="$CONFIG_DIR/id_ed25519"

if [[ ! -f "$KEY_PATH" ]]; then
    ssh-keygen -t ed25519 -N "" -C "immunehpc-controller" -f "$KEY_PATH"
    chmod 600 "$KEY_PATH"
    chmod 644 "$KEY_PATH.pub"
    chown "$SERVICE_USER":"$SERVICE_USER" "$KEY_PATH" "$KEY_PATH.pub"
    success "SSH keypair generated: $KEY_PATH"
    echo ""
    warn "IMPORTANT: Distribute the public key to all managed nodes:"
    echo -e "  ${BOLD}Public key:${NC}"
    cat "$KEY_PATH.pub"
    echo ""
    warn "Run on each node (or add to your provisioning):"
    echo "  ssh-copy-id -i $KEY_PATH.pub root@<node>"
    echo "  -- or --"
    echo "  cat $KEY_PATH.pub >> /root/.ssh/authorized_keys  (on each node)"
else
    info "SSH key already exists: $KEY_PATH"
fi

# ── Step 6: Config file ───────────────────────────────────────────────────
step "6/7  Config file"

PROD_CONFIG="$CONFIG_DIR/production.yaml"
if [[ ! -f "$PROD_CONFIG" ]]; then
    cp "$INSTALL_DIR/config/production.yaml" "$PROD_CONFIG"
    chown root:"$SERVICE_USER" "$PROD_CONFIG"
    chmod 640 "$PROD_CONFIG"
    success "Config template copied to $PROD_CONFIG"
    warn "IMPORTANT: Edit $PROD_CONFIG before starting:"
    echo "  - Set cluster.nodes to your actual node hostnames"
    echo "  - Set ssh.key_path to $KEY_PATH"
    echo "  - Set cobbler_host if using PXE reimage"
    echo "  - Set bmc_user/bmc_password for IPMI (or use $CONFIG_DIR/environment)"
else
    info "Config already exists: $PROD_CONFIG"
fi

ENV_FILE="$CONFIG_DIR/environment"
if [[ ! -f "$ENV_FILE" ]]; then
    cp "$INSTALL_DIR/systemd/environment.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    chown "$SERVICE_USER":"$SERVICE_USER" "$ENV_FILE"
    success "Environment file created: $ENV_FILE"
    warn "Edit $ENV_FILE to set secrets (BMC password, etc.)"
fi

# ── Step 7: systemd service ───────────────────────────────────────────────
step "7/7  Installing systemd service"

cp "$INSTALL_DIR/systemd/immunehpc.service" /etc/systemd/system/immunehpc.service
systemctl daemon-reload
systemctl enable immunehpc.service
success "systemd service installed and enabled"

# ── Final instructions ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗"
echo    "║  Installation complete!                      ║"
echo -e "╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo ""
echo -e "  ${BOLD}1. Edit the config:${NC}"
echo    "     vim $PROD_CONFIG"
echo ""
echo -e "  ${BOLD}2. Set secrets:${NC}"
echo    "     vim $ENV_FILE"
echo ""
echo -e "  ${BOLD}3. Distribute the SSH public key to each node:${NC}"
echo    "     ssh-copy-id -i $KEY_PATH.pub root@<each-node>"
echo ""
echo -e "  ${BOLD}4. Install node_exporter on each node:${NC}"
echo    "     bash scripts/setup_node.sh <node1> <node2> ..."
echo ""
echo -e "  ${BOLD}5. Start the controller:${NC}"
echo    "     systemctl start immunehpc"
echo    "     journalctl -u immunehpc -f"
echo ""
echo -e "  ${BOLD}6. Check status:${NC}"
echo    "     systemctl status immunehpc"
echo    "     $INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/main.py --status"
echo ""
