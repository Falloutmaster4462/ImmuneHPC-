#!/usr/bin/env bash
# fix_slurm.sh — Fix the broken Slurm on your cluster RIGHT NOW
#
# The problem: slurmd is trying to reach "headnode:6817" but that hostname
# doesn't exist in DNS. Everything is running but Slurm thinks it's offline.
#
# Run this on the team (head) node as root: sudo bash fix_slurm.sh

set -euo pipefail

HEAD_IP="172.21.12.41"
COM1_IP="172.21.12.21"

echo "=== Fixing Slurm hostname resolution ==="

# 1. Add headnode alias to /etc/hosts on THIS node
if ! grep -q "headnode" /etc/hosts; then
    echo "$HEAD_IP  team headnode" >> /etc/hosts
    echo "Added 'headnode' alias to /etc/hosts"
else
    # Make sure headnode is on the right line
    if grep -q "^$HEAD_IP" /etc/hosts; then
        grep "$HEAD_IP" /etc/hosts | grep -q "headnode" || \
            sed -i "/^$HEAD_IP/s/$/ headnode/" /etc/hosts
    else
        echo "$HEAD_IP  team headnode" >> /etc/hosts
    fi
    echo "headnode alias confirmed in /etc/hosts"
fi

# Also add com1
grep -q "^$COM1_IP" /etc/hosts || echo "$COM1_IP  com1" >> /etc/hosts
echo "com1 confirmed in /etc/hosts"

echo ""
echo "=== Check slurm.conf ==="
SLURM_CONF="/etc/slurm/slurm.conf"
if [[ -f "$SLURM_CONF" ]]; then
    echo "Current ControlMachine setting:"
    grep "ControlMachine" "$SLURM_CONF" || echo "  (not found)"
    echo ""

    # If ControlMachine is 'headnode', that's fine now — we just added the alias
    # If it's something else entirely, fix it
    CTRL=$(grep "^ControlMachine" "$SLURM_CONF" 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo "")
    if [[ "$CTRL" == "headnode" ]]; then
        echo "ControlMachine=headnode — now resolves to $HEAD_IP ✓"
    elif [[ "$CTRL" == "team" ]]; then
        echo "ControlMachine=team — already correct ✓"
    elif [[ -n "$CTRL" ]]; then
        echo "ControlMachine=$CTRL — this might be wrong"
        echo "If the slurmctld is on THIS node ($HEAD_IP), run:"
        echo "  sed -i 's/^ControlMachine=.*/ControlMachine=team/' $SLURM_CONF"
    fi
else
    echo "slurm.conf not found at $SLURM_CONF"
    echo "Try: find / -name slurm.conf 2>/dev/null | head -5"
fi

echo ""
echo "=== Restarting Slurm ==="
# Restart slurmctld if it exists (head node)
if systemctl list-units --type=service | grep -q slurmctld; then
    systemctl restart slurmctld && echo "slurmctld restarted" || echo "slurmctld restart failed"
fi

# Restart slurmd (both nodes run this)
systemctl restart slurmd && echo "slurmd restarted on head" || echo "slurmd restart failed"

echo ""
echo "=== Verifying ==="
sleep 3
if systemctl is-active slurmd &>/dev/null; then
    echo "slurmd: ACTIVE ✓"
else
    echo "slurmd: still not running — check: journalctl -u slurmd -n 20"
fi

# Check if it can now resolve headnode
if getent hosts headnode &>/dev/null; then
    echo "headnode DNS: resolves to $(getent hosts headnode | awk '{print $1}') ✓"
else
    echo "headnode DNS: still not resolving — /etc/hosts may need nscd flush"
    systemctl restart nscd 2>/dev/null || true
fi

echo ""
echo "=== Still seeing errors? Fix com1 too ==="
echo "Run this on com1 (172.21.12.21):"
echo ""
echo "  sudo bash -c \""
echo "    echo '$HEAD_IP  team headnode' >> /etc/hosts"
echo "    echo '$COM1_IP  com1' >> /etc/hosts"
echo "    systemctl restart slurmd"
echo "  \""
echo ""
echo "Then check: sinfo"
echo "Should show your nodes as 'idle' not 'down' or 'drain'"
