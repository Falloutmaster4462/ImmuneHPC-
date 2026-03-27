# ImmuneHPC+

**Autonomous self-healing HPC cluster controller for Rocky Linux 8/9**

ImmuneHPC+ continuously monitors, diagnoses, and repairs your HPC cluster — without human intervention. Inspired by biological immune systems, it combines real-time telemetry, ML-based anomaly detection, graduated self-healing, and an **Autonomous Self-improvement Loop (ASL)** that lets the controller evolve its own policies over time.

> **Production build** — this repo wires every module to real infrastructure (Prometheus, SSH, IPMI, Slurm, Ansible, LVM/btrfs, Cobbler). A simulation build (random-walk telemetry, stubbed actions) is available separately for development and testing.

---

## ✨ Features

| Capability | What it does |
|---|---|
| **Multi-source telemetry** | Prometheus node_exporter scrape → SSH /proc fallback → IPMI (temperature, power) → `nvidia-smi` (optional) → `squeue` (Slurm jobs) |
| **Anomaly detection** | Z-score detector + Isolation Forest (ML); shared statistical baseline across the cluster |
| **Graduated healing** | `systemctl` restart → Ansible config reapply → LVM/btrfs rollback → Cobbler PXE reimage — stops at the lightest fix that works |
| **Active defense** | `auditd` audit log scanning, rogue-process detection (`ps`), listener audit (`ss`), SELinux enforcement, `firewalld` rich-rule isolation |
| **Node quarantine** | Firewall isolation + Slurm drain; automatic release when node is healthy |
| **NSGA-II optimiser** | Multi-objective scheduling: DVFS power targets, QoS, Slurm weight tuning |
| **Autonomous Self-improvement (ASL)** | RL agent + LLM code generator propose patches → sandboxed validation → trust-scored canary deploy → full rollout or rollback |
| **Structured audit trail** | Append-only JSONL log + systemd journal; query with `immunectl audit` |
| **Prometheus export** | Pushgateway integration for cluster-wide dashboards |

---

## 🏗️ Architecture

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
│   ├── EXECUTE ─── Defense:   ausearch + ps + ss + SELinux + firewalld
│   │           ─── Healer:    systemctl → Ansible → LVM/btrfs → Cobbler
│   │           ─── Optimizer: NSGA-II (DVFS, QoS, scheduling weights)
│   │           ─── Quarantine: firewall isolation + Slurm drain
│   │
│   └── LEARN ──── ASL pipeline: RL agent + LLM codegen + sandbox + canary deploy
│
Managed Nodes (Rocky Linux)
├── prometheus-node-exporter  :9100
├── auditd
├── slurmd
├── firewalld
└── SELinux (enforcing)
```

The controller follows a **MAPE-K loop** (Monitor → Analyze → Plan → Execute) gated by a Lyapunov stability check in the `AutonomousSupervisor` — policy changes are only applied when the cluster is provably stable enough to tolerate them.

---

## 📦 Requirements

### Controller node
- Rocky Linux 8 or 9
- Python 3.8+
- `ansible` — `dnf install -y ansible`
- `ipmitool` — `dnf install -y ipmitool`
- SSH client with passwordless key access to all managed nodes
- *(Optional)* Cobbler PXE server for automated node reimaging

### Each managed node
- Rocky Linux 8 or 9
- `prometheus-node-exporter` on port 9100
- `auditd` running
- `firewalld` running
- SELinux in enforcing mode
- Controller's SSH public key in `/root/.ssh/authorized_keys`

### Python dependencies
```
pyyaml>=6.0

# Optional but recommended:
# paramiko>=3.0           # pure-Python SSH fallback
# prometheus-client>=0.17 # richer Prometheus export
# scikit-learn>=1.3       # real IsolationForest implementation
```

---

## 🚀 Quick Start

### 1. Install on the controller

```bash
git clone <repo> /opt/immunehpc-src
cd /opt/immunehpc-src
sudo bash scripts/install.sh
```

The installer will:
- Install system dependencies (`ansible`, `ipmitool`, etc.)
- Create the `immunehpc` service user
- Generate an SSH keypair at `/etc/immunehpc/id_ed25519`
- Install and enable the systemd service

### 2. Configure

```bash
vim /etc/immunehpc/production.yaml
```

Minimum required changes:

```yaml
ssh:
  key_path: /etc/immunehpc/id_ed25519   # already set by installer

cluster:
  nodes:
    - node01.cluster.local
    - node02.cluster.local
    - node03.cluster.local
```

Set secrets via environment (never put passwords in YAML):

```bash
vim /etc/immunehpc/environment
# IMMUNEHPC_BMC_PASS=<your-ipmi-password>
# IMMUNEHPC_COBBLER_PASS=<your-cobbler-password>
```

### 3. Prepare managed nodes

Distribute the SSH public key:

```bash
cat /etc/immunehpc/id_ed25519.pub
# Append to /root/.ssh/authorized_keys on each managed node
```

Then run the node setup script:

```bash
bash scripts/setup_node.sh node01 node02 node03
```

This installs `prometheus-node-exporter`, `auditd`, hardens SSH, sets SELinux to enforcing, and opens required firewall ports on each node.

### 4. Start the service

```bash
systemctl start immunehpc
journalctl -u immunehpc -f
```

---

## 🛠️ Operations (`immunectl`)

```bash
# Cluster overview
bash scripts/immunectl.sh status
bash scripts/immunectl.sh nodes

# Manual node operations
bash scripts/immunectl.sh drain   node02
bash scripts/immunectl.sh release node02
bash scripts/immunectl.sh heal    node03

# Audit trail
bash scripts/immunectl.sh audit --last 100

# Fault injection for testing
bash scripts/immunectl.sh inject node01 cpu
bash scripts/immunectl.sh inject node01 kill
```

---

## 🩺 Healing Strategy

ImmuneHPC+ attempts the lightest effective intervention first:

| Step | Strategy | Tool | What happens |
|---|---|---|---|
| 1 | `restart_service` | `systemctl` via SSH | Lists failed units; resets and restarts each one |
| 2 | `reapply_config` | `ansible-playbook` | Idempotent playbook: packages, services, configs, firewall, sysctl |
| 3 | `rollback` | LVM `lvconvert --merge` or `snapper rollback` | Reverts root FS to last snapshot and reboots the node |
| 4 | `reimage_node` | Cobbler + IPMI | Netboot → Kickstart; waits up to 20 min for node to rejoin |

---

## 📡 Telemetry Priority

```
CPU / Memory / Network
  └─ 1st: Prometheus node_exporter HTTP scrape  (fast, reliable)
  └─ 2nd: SSH /proc reader                       (fallback, zero extra deps)
  └─ fail: node marked DEGRADED

Temperature / Power
  └─ 1st: IPMI (ipmitool sdr / dcmi power)
  └─ 2nd: node_exporter hwmon metrics (fallback)

Jobs
  └─ Slurm squeue  (if use_slurm: true)

GPU
  └─ nvidia-smi    (if use_gpu: true)
```

---

## 🧠 Autonomous Self-improvement Loop (ASL)

The ASL pipeline enables the controller to improve its own behaviour at runtime:

1. **RL agent** — Q-learning agent proposes parameter (`Δθ`) and policy (`Δπ`) patches every control cycle
2. **LLM code generator** — when a clear fault diagnosis exists, generates code-level patches (`ΔC`) via a configurable LLM backend
3. **Sandboxed validation** — every patch is tested in an isolated environment; safety, stability, and performance gates must all pass
4. **Trust scoring** — patches are scored: sandbox (40%) + stability (25%) + performance (20%) + safety (15%)
5. **Canary deploy** — high-trust patches are deployed to one node first; the system monitors outcome before full rollout
6. **Rollback** — any regression triggers automatic reversion; the RL agent receives the real post-deploy reward signal

The `AutonomousSupervisor` enforces a Lyapunov stability gate — patches are only promoted when the cluster is in a safe-enough state to tolerate the change.

---

## 📁 Repository Structure

```
immunehpc-real/
├── core/
│   ├── state.py          # ClusterState + NodeState
│   ├── controller.py     # MAPE-K orchestrator
│   └── supervisor.py     # Lyapunov gating + ASL trigger
├── modules/
│   ├── monitor.py        # Prometheus / SSH / IPMI / nvidia-smi / Slurm
│   ├── healer.py         # systemctl / Ansible / LVM-btrfs / Cobbler
│   ├── defense.py        # auditd / SELinux / ps / ss / firewalld
│   ├── anomaly.py        # Z-score + Isolation Forest
│   ├── quarantine.py     # Node isolation
│   ├── optimizer.py      # NSGA-II multi-objective optimiser
│   └── scheduler.py      # Slurm-aware adaptive scheduler
├── asl/
│   ├── patch.py          # Patch model P=(ΔC, Δθ, Δπ)
│   ├── sandbox.py        # Safety validation + canary deploy
│   ├── pipeline.py       # ASL orchestration
│   ├── rl_agent.py       # Q-learning agent
│   ├── llm_backend.py    # LLM integration for code generation
│   └── code_generator.py # Patch synthesis
├── utils/
│   ├── events.py         # Event bus
│   ├── logger.py         # Structured logging
│   └── metrics.py        # Metrics registry
├── config/
│   └── production.yaml   # ← Edit before running
├── ansible/
│   ├── site.yml          # Idempotent config reapply playbook
│   └── inventory.ini     # Node inventory
├── systemd/
│   ├── immunehpc.service # systemd unit
│   └── environment.example
├── scripts/
│   ├── install.sh        # Full installer (run first)
│   ├── setup_node.sh     # Prepare managed nodes
│   ├── install_cluster.sh# Cluster-wide install helper
│   ├── fix_slurm.sh      # Slurm repair utility
│   └── immunectl.sh      # Operations CLI
├── tests/
│   ├── test_integration_real.py  # Integration tests (mocked SSH)
│   ├── test_anomaly.py
│   └── test_asl.py
└── main.py               # Entry point
```

---

## 🧪 Running Tests

Tests use mocked SSH subprocess calls — no live nodes required.

```bash
cd /opt/immunehpc
venv/bin/python3 -m unittest discover tests/ -v
```

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `IMMUNEHPC_BMC_PASS` | IPMI BMC password |
| `IMMUNEHPC_COBBLER_PASS` | Cobbler server password |
| `IMMUNEHPC_SSH_KEY` | Path to SSH private key (overrides config) |

---

## 🔒 Security Notes

- All secrets are injected via environment variables — never stored in YAML config
- SSH uses Ed25519 keypairs with host-key TOFU fingerprint verification
- ASL code patches are executed only inside an isolated sandbox before any deployment
- SELinux enforcing mode is validated on each managed node at startup
- The `immunehpc` service runs as a dedicated low-privilege user

---

## 📄 License

See [LICENSE](LICENSE) for details.
