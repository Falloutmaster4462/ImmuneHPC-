# ImmuneHPC+ — Rocky Linux Production Build

Autonomous self-healing HPC cluster controller for Rocky Linux 8/9.

## What's different from the simulation build

| Layer | Simulation | This build (Rocky Linux) |
|---|---|---|
| **Telemetry** | Random walk generator | Prometheus node_exporter scraper + SSH /proc reader + IPMI |
| **GPU metrics** | Simulated | `nvidia-smi` (optional) |
| **Job state** | Synthetic jobs | `squeue` / Slurm REST API |
| **Service restart** | Stub | SSH → `systemctl reset-failed && restart` |
| **Config reapply** | Stub | Ansible playbook (`ansible-playbook --limit <node>`) |
| **Rollback** | Stub | LVM snapshot merge _or_ `snapper rollback` (btrfs) |
| **Reimage** | Stub | Cobbler netboot + IPMI power-cycle + PXE Kickstart |
| **Node auth** | HMAC token | SSH host key TOFU fingerprint store |
| **IDS** | Random flag | `ausearch` (auditd) + process scanner + `ss` listener scan |
| **Process scan** | Random flag | SSH → `ps` scan for known-bad binaries |
| **SELinux** | None | `getenforce` check + auto-enforce via `setenforce 1` |
| **Firewall mitigation** | Stub | `firewall-cmd` rich-rules |
| **Metrics export** | Local only | Prometheus Pushgateway |
| **Audit trail** | None | Structured JSONL log + systemd journal |
| **Service** | Foreground | systemd unit (`immunehpc.service`) |

---

## Architecture

```
Controller Node (Rocky Linux)
│
├── ImmuneHPC+ Controller (systemd: immunehpc.service)
│   │
│   ├── MONITOR ─── Prometheus scraper (:9100) ──→ each node
│   │           ─── SSH /proc reader  (fallback)
│   │           ─── IPMI sdr (temperature, power)
│   │           ─── nvidia-smi (GPU, optional)
│   │           ─── squeue (Slurm jobs)
│   │
│   ├── ANALYZE ─── Z-score anomaly detector
│   │           ─── Isolation Forest (ML)
│   │
│   ├── EXECUTE ─── Defense: ausearch + ps + ss + SELinux + firewalld
│   │           ─── Healer: systemctl → Ansible → LVM/btrfs → Cobbler
│   │           ─── Optimizer: NSGA-II (DVFS, QoS, scheduling weights)
│   │           ─── Quarantine: firewall isolation + Slurm drain
│   │
│   └── LEARN ──── ASL pipeline: RL agent + sandbox + canary deploy
│
Managed Nodes (Rocky Linux)
├── prometheus-node-exporter  :9100
├── auditd
├── slurmd
├── firewalld
└── SELinux (enforcing)
```

---

## Requirements

### Controller node
- Rocky Linux 8 or 9
- Python 3.8+
- Ansible: `dnf install -y ansible`
- ipmitool: `dnf install -y ipmitool`
- SSH client + passwordless key access to all nodes
- (Optional) Cobbler PXE server for node reimaging

### Each managed node
- Rocky Linux 8 or 9
- `prometheus-node-exporter` running on port 9100
- `auditd` running
- `firewalld` running
- SELinux enforcing
- SSH: public key from controller in `authorized_keys`

---

## Quick Start

### 1. Install on controller

```bash
git clone <repo> /opt/immunehpc-src
cd /opt/immunehpc-src
sudo bash scripts/install.sh
```

This will:
- Install system dependencies (`ansible`, `ipmitool`, etc.)
- Create the `immunehpc` service user
- Generate an SSH keypair at `/etc/immunehpc/id_ed25519`
- Install the systemd service

### 2. Configure

```bash
vim /etc/immunehpc/production.yaml
```

Minimum changes required:
```yaml
ssh:
  key_path: /etc/immunehpc/id_ed25519   # already set by installer

cluster:
  nodes:
    - node01.cluster.local              # YOUR actual hostnames
    - node02.cluster.local
    - node03.cluster.local
```

Set secrets (never put passwords in YAML):
```bash
vim /etc/immunehpc/environment
# Set IMMUNEHPC_BMC_PASS and IMMUNEHPC_COBBLER_PASS
```

### 3. Prepare managed nodes

Distribute the SSH public key:
```bash
cat /etc/immunehpc/id_ed25519.pub
# Add to each node's /root/.ssh/authorized_keys
```

Run the node setup script:
```bash
bash scripts/setup_node.sh node01 node02 node03
```

This installs `prometheus-node-exporter`, `auditd`, hardens SSH,
sets SELinux to enforcing, and opens the required firewall ports.

### 4. Start

```bash
systemctl start immunehpc
journalctl -u immunehpc -f
```

### 5. Operate

```bash
# Status and node health
bash scripts/immunectl.sh status
bash scripts/immunectl.sh nodes

# Manual operations
bash scripts/immunectl.sh drain  node02
bash scripts/immunectl.sh release node02
bash scripts/immunectl.sh heal   node03

# Audit trail
bash scripts/immunectl.sh audit --last 100

# Fault injection (testing)
bash scripts/immunectl.sh inject node01 cpu
bash scripts/immunectl.sh inject node01 kill
```

---

## File Structure

```
immunehpc-real/
├── core/
│   ├── state.py          # ClusterState + NodeState (shared with sim)
│   ├── controller.py     # MAPE-K orchestrator (production wiring)
│   └── supervisor.py     # Lyapunov gating + ASL trigger
├── modules/
│   ├── monitor.py        # ★ REAL: Prometheus/SSH/IPMI/nvidia-smi/Slurm
│   ├── healer.py         # ★ REAL: systemctl/Ansible/LVM-btrfs/Cobbler
│   ├── defense.py        # ★ REAL: auditd/SELinux/ps/ss/firewalld
│   ├── anomaly.py        # Statistical + ML (shared with sim)
│   ├── quarantine.py     # Node isolation (shared with sim)
│   ├── optimizer.py      # NSGA-II (shared with sim)
│   └── scheduler.py      # Slurm-aware scheduler (shared with sim)
├── asl/
│   ├── patch.py          # Patch model P=(ΔC,Δθ,Δπ)
│   ├── sandbox.py        # Safety validation + canary deploy
│   ├── pipeline.py       # ASL orchestration
│   └── rl_agent.py       # Q-learning agent
├── utils/
│   ├── events.py         # Event bus
│   ├── logger.py         # Structured logging
│   └── metrics.py        # Metrics registry
├── config/
│   └── production.yaml   # ★ Edit this before running
├── ansible/
│   ├── site.yml          # Idempotent config reapply playbook
│   └── inventory.ini     # Node inventory (mirrors config.yaml)
├── systemd/
│   ├── immunehpc.service # systemd unit
│   └── environment.example # Secret env vars template
├── scripts/
│   ├── install.sh        # ★ Run first: full installer
│   ├── setup_node.sh     # Prepare each managed node
│   └── immunectl.sh      # Operations CLI
├── tests/
│   ├── test_integration_real.py  # ★ Real backend tests (mocked SSH)
│   ├── test_anomaly.py   # Shared with sim
│   └── test_asl.py       # Shared with sim
├── main.py               # Entry point
└── requirements.txt      # pyyaml (+ optional scikit-learn, prometheus-client)
```

---

## Telemetry collection priority

For each node, per metric:

```
CPU / Memory / Network
  └─ 1st: Prometheus node_exporter HTTP scrape  (fast, reliable)
  └─ 2nd: SSH /proc reader                       (fallback, zero extra deps)
  └─ fail: node marked DEGRADED

Temperature / Power
  └─ 1st: IPMI (ipmitool sdr / dcmi power)      (most accurate)
  └─ 2nd: node_exporter hwmon metrics            (fallback)

Jobs
  └─ Slurm squeue (if use_slurm: true)

GPU
  └─ nvidia-smi (if use_gpu: true)
```

---

## Healing strategy detail

| # | Strategy | Tool | What it does |
|---|---|---|---|
| 1 | `restart_service` | `systemctl` via SSH | Lists failed units, resets and restarts each |
| 2 | `reapply_config` | `ansible-playbook` | Idempotent playbook: packages, services, configs, firewall, sysctl |
| 3 | `rollback` | LVM `lvconvert --merge` or `snapper rollback` | Reverts root FS to last snapshot; reboots node |
| 4 | `reimage_node` | Cobbler + IPMI | Netboot → Kickstart → wait up to 20 min for node to rejoin |

---

## Running tests

```bash
cd /opt/immunehpc
venv/bin/python3 -m unittest discover tests/ -v
```

Tests use mocked SSH subprocess calls — no real nodes needed.
