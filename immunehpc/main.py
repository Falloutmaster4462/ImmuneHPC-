"""
main.py — ImmuneHPC+ Production Entry Point (Rocky Linux)

Usage:
  python3 main.py                              # default config
  python3 main.py --config /etc/immunehpc/production.yaml
  python3 main.py --nodes node01 node02 node03  # override node list
  python3 main.py --status                     # print status and exit
"""

import argparse
import json
import os
import sys

# Allow env overrides for secrets (never put passwords in config files)
_ENV_OVERRIDES = {
    "IMMUNEHPC_COBBLER_PASS": ("healing", "cobbler_password"),
    "IMMUNEHPC_BMC_PASS":     ("healing", "bmc_password"),
    "IMMUNEHPC_SSH_KEY":      ("ssh", "key_path"),
}


def apply_env_overrides(config: dict) -> dict:
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        val = os.environ.get(env_var)
        if val:
            config.setdefault(section, {})[key] = val
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ImmuneHPC+ — Autonomous Self-Healing HPC Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (override config secrets):
  IMMUNEHPC_COBBLER_PASS    Cobbler server password
  IMMUNEHPC_BMC_PASS        IPMI BMC password
  IMMUNEHPC_SSH_KEY         Path to SSH private key

Examples:
  python3 main.py --config /etc/immunehpc/production.yaml
  IMMUNEHPC_BMC_PASS=secret python3 main.py
  python3 main.py --nodes node01 node02 --status
        """
    )
    parser.add_argument("--config", default="config/production.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--nodes", nargs="+", metavar="HOST",
                        help="Override cluster node list")
    parser.add_argument("--status", action="store_true",
                        help="Print a one-shot status report and exit")
    args = parser.parse_args()

    from core.controller import ImmuneHPCController
    import yaml

    # Validate config exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        print("Copy and edit config/production.yaml before running.", file=sys.stderr)
        sys.exit(1)

    controller = ImmuneHPCController(config_path=args.config)
    apply_env_overrides(controller.config)

    node_list = args.nodes or None
    controller.provision_nodes(node_list)

    if args.status:
        # One-shot: collect once and print report
        controller.monitor.collect_once()
        report = controller.status_report()
        print(json.dumps(report, indent=2, default=str))
        return

    controller.start(block=True)


if __name__ == "__main__":
    main()
