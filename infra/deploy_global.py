#!/usr/bin/env python3
"""
VANTAGE-STR — Global Multi-Node Deployment Script (Cross-Platform)
Phase 6: Spins up 5 simulated forensic DNA nodes across continents.

Replaces deploy_global.sh for better Windows compat without WSL.

Usage:
    python infra/deploy_global.py [up|down|status]
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_NAME = "vantage-str"
NETWORK_NAME = f"{PROJECT_NAME}-global"
ORCHESTRATOR_PORT = 8000

NODES = {
    "jandarma-tr": {"country": "Turkey", "city": "Istanbul", "code": "TR", "port": 8101, "profiles": 100},
    "bka-de": {"country": "Germany", "city": "Frankfurt", "code": "DE", "port": 8102, "profiles": 150},
    "npa-jp": {"country": "Japan", "city": "Tokyo", "code": "JP", "port": 8103, "profiles": 80},
    "fbi-us": {"country": "USA", "city": "Washington", "code": "US", "port": 8104, "profiles": 200},
    "pf-br": {"country": "Brazil", "city": "Brasilia", "code": "BR", "port": 8105, "profiles": 120},
}

BASE_DIR = Path(__file__).parent.parent
INFRA_DIR = BASE_DIR / "infra"
CERTS_DIR = INFRA_DIR / "certs"
COMPOSE_FILE = INFRA_DIR / "docker-compose.global.yml"

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_cmd(cmd, cwd=None, check=True, env=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, shell=True if os.name == 'nt' else False) # shell=True needed for some commands on Windows
    if check and result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        sys.exit(result.returncode)

def log(msg):
    print(f"[VANTAGE] {msg}")

def ok(msg):
    print(f"[  OK  ] {msg}")

# ── Certificate Generation ────────────────────────────────────────────────────

def generate_mtls_certs():
    log("Generating mTLS certificates...")
    
    # Use a temporary directory with ASCII-only path to avoid OpenSSL issues with "Masaüstü"
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_certs_dir = Path(temp_dir)
        log(f"Using temp dir for cert generation: {temp_certs_dir}")
        
        # Root CA
        ca_key = temp_certs_dir / "ca.key"
        ca_crt = temp_certs_dir / "ca.crt"
        
        # Windows-safe subject
        subj_root = "//CN=VantageRoot" 
        subj_orch = "//CN=orchestrator"

        # Environment
        env = os.environ.copy()
        env["MSYS_NO_PATHCONV"] = "1"
        env["OPENSSL_CONF"] = str(INFRA_DIR / "openssl.cnf")

        run_cmd(["openssl", "genrsa", "-out", str(ca_key), "4096"], env=env)
        run_cmd(["openssl", "req", "-new", "-x509", "-days", "365", "-key", str(ca_key), 
                 "-out", str(ca_crt), "-subj", subj_root], env=env)
        ok("Root CA generated")

        # Orchestrator cert
        orch_key = temp_certs_dir / "orchestrator.key"
        orch_csr = temp_certs_dir / "orchestrator.csr"
        orch_crt = temp_certs_dir / "orchestrator.crt"
        
        run_cmd(["openssl", "genrsa", "-out", str(orch_key), "2048"], env=env)
        run_cmd(["openssl", "req", "-new", "-key", str(orch_key), "-out", str(orch_csr), 
                 "-subj", subj_orch], env=env)
        run_cmd(["openssl", "x509", "-req", "-in", str(orch_csr), "-CA", str(ca_crt), 
                 "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(orch_crt), "-days", "365"], env=env)
        
        ok("Orchestrator certificate generated")

        # Node certs
        for node_id, info in NODES.items():
            node_key = temp_certs_dir / f"{node_id}.key"
            node_csr = temp_certs_dir / f"{node_id}.csr"
            node_crt = temp_certs_dir / f"{node_id}.crt"
            subj_node = f"//CN={node_id}"
            
            run_cmd(["openssl", "genrsa", "-out", str(node_key), "2048"], env=env)
            run_cmd(["openssl", "req", "-new", "-key", str(node_key), "-out", str(node_csr), 
                     "-subj", subj_node], env=env)
            run_cmd(["openssl", "x509", "-req", "-in", str(node_csr), "-CA", str(ca_crt), 
                     "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(node_crt), "-days", "365"], env=env)
            
            ok(f"Certificate for {node_id} ({info['country']})")

        # Copy all certs to final destination
        CERTS_DIR.mkdir(parents=True, exist_ok=True)
        for item in temp_certs_dir.glob("*"):
            if item.is_file() and item.suffix in ['.crt', '.key']:
                 shutil.copy2(item, CERTS_DIR / item.name)
        log(f"Certificates moved to {CERTS_DIR}")

# ── Docker Compose Generation ─────────────────────────────────────────────────

def generate_compose():
    log("Generating Docker Compose configuration...")
    
    services_yaml = ""
    ip_suffix = 10
    
    # Sort nodes for stability
    sorted_nodes = sorted(NODES.items())
    
    for node_id, info in sorted_nodes:
        services_yaml += f"""
  # ── Node: {node_id} ({info['country']}, {info['city']}) ──
  {node_id}:
    build:
      context: ../backend
      dockerfile: Dockerfile
    container_name: vantage-{node_id}
    ports:
      - "{info['port']}:8000"
    environment:
      - VANTAGE_ROLE=node
      - VANTAGE_NODE_ID={node_id}
      - VANTAGE_COUNTRY={info['code']}
      - VANTAGE_CITY={info['city']}
      - VANTAGE_ORCHESTRATOR_URL=http://172.28.0.2:8000
      - VANTAGE_SYNTHETIC_PROFILES={info['profiles']}
      - VANTAGE_MTLS_CA=/certs/ca.crt
      - VANTAGE_MTLS_CERT=/certs/{node_id}.crt
      - VANTAGE_MTLS_KEY=/certs/{node_id}.key
      - VANTAGE_ZKP_ENABLED=true
      - VANTAGE_DSPY_ENABLED=true
    volumes:
      - ../infra/certs:/certs:ro
    depends_on:
      orchestrator:
        condition: service_healthy
    networks:
      vantage-global:
        ipv4_address: 172.28.0.{ip_suffix}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
"""
        ip_suffix += 1

    content = f"""# ═══════════════════════════════════════════════════════════════════════════════
# VANTAGE-STR — Global Network Docker Compose
# Auto-generated by deploy_global.py
# ═══════════════════════════════════════════════════════════════════════════════

version: "3.9"

networks:
  vantage-global:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16

services:
  # ── Global Orchestrator ──
  orchestrator:
    build:
      context: ../backend
      dockerfile: Dockerfile
    container_name: vantage-orchestrator
    ports:
      - "8000:8000"
    environment:
      - VANTAGE_ROLE=orchestrator
      - VANTAGE_MTLS_CA=/certs/ca.crt
      - VANTAGE_MTLS_CERT=/certs/orchestrator.crt
      - VANTAGE_MTLS_KEY=/certs/orchestrator.key
    volumes:
      - ../infra/certs:/certs:ro
    networks:
      vantage-global:
        ipv4_address: 172.28.0.2
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

{services_yaml}
"""
    
    with open(COMPOSE_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    
    ok(f"Docker Compose file generated at {COMPOSE_FILE}")

# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_up():
    print("\n" + "═"*60)
    print("  VANTAGE-STR — Global Network Deployment")
    print("═"*60 + "\n")

    generate_mtls_certs()
    generate_compose()

    log(f"Launching global network ({len(NODES)} nodes + orchestrator)...")
    print("")

    for node_id, info in sorted(NODES.items()):
        print(f"  ● {node_id} — {info['country']}, {info['city']} ({info['profiles']} profiles) → :{info['port']}")

    print("")
    # subprocess.run requires list for Popen but shell=True on windows can take string. 
    # For compatibility, let's use list and shell=True for windows.
    run_cmd(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT_NAME, "up", "-d", "--build"])

    print("")
    ok("Global network is live!")
    print(f"  Orchestrator: http://localhost:{ORCHESTRATOR_PORT}")
    for node_id, info in sorted(NODES.items()):
        print(f"  {node_id}: http://localhost:{info['port']}")
    print("")

def cmd_down():
    log("Shutting down global network...")
    if COMPOSE_FILE.exists():
        run_cmd(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT_NAME, "down", "-v"])
        ok("Global network stopped.")
    else:
        print("Compose file not found, nothing to down.")

def cmd_status():
    log("Network status:")
    if COMPOSE_FILE.exists():
        run_cmd(["docker", "compose", "-f", str(COMPOSE_FILE), "-p", PROJECT_NAME, "ps"])
    else:
        print("Compose file not found.")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        cmd_up()
    else:
        cmd = sys.argv[1]
        if cmd == "up":
            cmd_up()
        elif cmd == "down":
            cmd_down()
        elif cmd == "status":
            cmd_status()
        else:
            print("Usage: python infra/deploy_global.py [up|down|status]")
            sys.exit(1)
