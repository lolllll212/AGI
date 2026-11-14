"""The deployment files are checked as far as they can be without Docker: they parse, they agree
with each other, and every metric they mention is one the server really exports."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agi.infrastructure.api.server import AgentServer
from agi.infrastructure.api.telemetry import AgentMetrics
from tests.fakes import sales_service

yaml = pytest.importorskip("yaml")
REPO = Path(__file__).resolve().parents[2]
MONITORING = REPO / "monitoring"
METRIC = re.compile(r"\bagi_[a-z_]+\b")


def load(path):
    return yaml.safe_load((REPO / path).read_text())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Every series name the server can export, including the _bucket/_sum/_count of histograms."""
    service = sales_service(tmp_path_factory.mktemp("metrics"))
    try:
        registry = AgentMetrics(service).registry
        names = set()
        for name in registry.names():
            names.add(name)
            if registry._metrics[name].kind == "histogram":
                names |= {f"{name}_bucket", f"{name}_sum", f"{name}_count"}
        return names
    finally:
        service.close()


def test_alert_rules_only_use_metrics_the_server_exports(exported):
    groups = load("monitoring/prometheus/alerts.yml")["groups"]
    rules = [rule for group in groups for rule in group["rules"]]
    assert len(rules) >= 8 and len({rule["alert"] for rule in rules}) == len(rules)
    for rule in rules:
        used = set(METRIC.findall(rule["expr"]))
        assert used or "up{" in rule["expr"], f"{rule['alert']} watches nothing"
        assert used <= exported, f"{rule['alert']} uses unknown metrics: {sorted(used - exported)}"
        assert rule["labels"]["severity"] in {"info", "warning", "critical"}
        assert rule["annotations"]["summary"] and rule["annotations"]["description"]
        assert rule["expr"].count("(") == rule["expr"].count(")") and rule["expr"].count("{") == rule["expr"].count("}")


def test_alert_label_values_are_ones_the_server_uses(tmp_path):
    text = (MONITORING / "prometheus" / "alerts.yml").read_text()
    assert set(re.findall(r'outcome=~?"([^"]+)"', text)) == {"failure|error|timeout"}
    assert 'reason="unauthorized"' in text and 'result="dropped"' in text
    service = sales_service(tmp_path)
    try:
        with AgentServer(service, port=0, token="the-shared-token-1") as server:
            import urllib.error
            import urllib.request
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(server.url + "/v1/info", timeout=10)
            request = urllib.request.Request(server.url + "/metrics", headers={"Authorization": "Bearer the-shared-token-1"})
            page = urllib.request.urlopen(request, timeout=10).read().decode()
        assert 'agi_http_rejected_total{reason="unauthorized"} 1' in page
    finally:
        service.close()


def test_dashboard_is_the_generated_one_and_only_uses_exported_metrics(exported):
    spec = importlib.util.spec_from_file_location("build_dashboard", REPO / "scripts" / "build_dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = (MONITORING / "grafana" / "dashboards" / "agi-overview.json").read_text()
    assert committed == module.render(), "run: python scripts/build_dashboard.py"
    dashboard = json.loads(committed)
    panels = dashboard["panels"]
    assert len(panels) >= 15 and len({panel["id"] for panel in panels}) == len(panels)
    sources = {source["uid"] for source in load("monitoring/grafana/provisioning/datasources/datasources.yml")["datasources"]}
    cells = set()
    for panel in panels:
        assert panel["datasource"]["uid"] in sources
        for target in panel["targets"]:
            used = set(METRIC.findall(target["expr"]))
            assert used <= exported, f"{panel['title']!r} uses unknown metrics: {sorted(used - exported)}"
        box = panel["gridPos"]
        assert box["x"] + box["w"] <= 24
        for x in range(box["x"], box["x"] + box["w"]):
            for y in range(box["y"], box["y"] + box["h"]):
                assert (x, y) not in cells, f"{panel['title']!r} overlaps another panel"
                cells.add((x, y))
    # Nearly everything the server exports is on the dashboard.
    shown = {name for panel in panels for target in panel["targets"] for name in METRIC.findall(target["expr"])}
    base = {re.sub(r"_(bucket|sum|count)$", "", name) for name in shown}
    assert {name for name in exported if not name.endswith(("_bucket", "_sum", "_count"))} - base <= {"agi_model_configured"}


def test_prometheus_scrapes_the_api_with_the_token_and_loads_the_rules():
    config = load("monitoring/prometheus/prometheus.yml")
    job = next(job for job in config["scrape_configs"] if job["job_name"] == "agi")
    assert job["metrics_path"] == "/metrics" and job["static_configs"][0]["targets"] == ["api:8000"]
    assert job["authorization"] == {"type": "Bearer", "credentials_file": "/run/secrets/agi_api_token"}
    assert config["rule_files"] == ["/etc/prometheus/alerts.yml"]
    assert config["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"] == ["alertmanager:9093"]
    alertmanager = load("monitoring/alertmanager/alertmanager.yml")
    receivers = {receiver["name"] for receiver in alertmanager["receivers"]}
    assert alertmanager["route"]["receiver"] in receivers
    assert all(route["receiver"] in receivers for route in alertmanager["route"].get("routes", []))


def test_compose_file_is_consistent_with_the_files_around_it():
    compose = load("docker-compose.yml")
    services = compose["services"]
    assert set(services) == {"api", "nginx", "prometheus", "alertmanager", "loki", "promtail", "grafana"}
    for name, service in services.items():
        for volume in service.get("volumes", []):
            source = volume.split(":")[0]
            if source.startswith("./"):
                assert (REPO / source).exists(), f"{name} mounts {source}, which does not exist"
            elif not source.startswith("/"):
                assert source in compose["volumes"], f"{name} uses the undeclared volume {source}"
        for secret in service.get("secrets", []):
            assert secret in compose["secrets"]
        for other in service.get("depends_on", []):
            assert other in services
        assert "build" in service or re.search(r":v?\d", service["image"]), f"{name}: pin the image version"
    # Only the proxy is reachable from outside; the token reaches the api and Prometheus as a secret file.
    assert [name for name, service in services.items() if "ports" in service] == ["nginx"]
    assert "agi_api_token" in services["api"]["secrets"] and "agi_api_token" in services["prometheus"]["secrets"]
    assert "AGI_API_TOKEN" not in json.dumps(services["api"])
    command = services["api"]["command"]
    assert command[:3] == ["serve", "--config", "config/docker.toml"] and (REPO / command[2]).is_file()
    # The names the other configs use to reach each other are service names in this file.
    nginx = (MONITORING / "nginx" / "nginx.conf").read_text()
    assert set(re.findall(r"proxy_pass http://([a-z]+):", nginx)) <= set(services)
    assert "http://loki:3100" in (MONITORING / "promtail" / "promtail-config.yml").read_text()
    assert nginx.count("{") == nginx.count("}") and "location = /metrics" in nginx


def test_the_cli_accepts_every_flag_the_container_setup_relies_on(tmp_path):
    token = tmp_path / "token.txt"
    token.write_text("  \n")
    done = subprocess.run([sys.executable, "-m", "agi", "serve", "--demo", "--port", "0", "--token-file", str(token)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "is empty" in done.stderr
    helped = subprocess.run([sys.executable, "-m", "agi", "serve", "--help"], capture_output=True, text=True, timeout=60)
    for flag in ("--config", "--token-file", "--plugins", "--access-log", "--demo", "--no-ui", "--no-metrics"):
        assert flag in helped.stdout


def test_secrets_and_certificates_are_ignored_by_git():
    ignored = (REPO / ".gitignore").read_text().split()
    assert {".env", "/secrets/*", "/monitoring/nginx/ssl/*", "node_modules/"} <= set(ignored)
    example = (REPO / ".env.example").read_text()
    for line in example.splitlines():
        if "=" in line and not line.startswith("#"):
            name, value = line.split("=", 1)
            assert "KEY" not in name and "PASSWORD" not in name or value == "", f"{name} must be blank in the example"


def test_smoke_script_passes_against_a_live_server_and_fails_against_a_wrong_token(tmp_path):
    service = sales_service(tmp_path)
    try:
        with AgentServer(service, port=0, token="the-shared-token-1") as server:
            good = tmp_path / "good.txt"
            good.write_text("the-shared-token-1\n")
            bad = tmp_path / "bad.txt"
            bad.write_text("the-wrong-token-99\n")
            script = [sys.executable, str(REPO / "scripts" / "smoke.py"), server.url]
            passed = subprocess.run(script + ["--token-file", str(good)], capture_output=True, text=True, timeout=120)
            assert passed.returncode == 0, passed.stdout + passed.stderr
            assert "all checks passed" in passed.stdout and "FAIL" not in passed.stdout
            failed = subprocess.run(script + ["--token-file", str(bad)], capture_output=True, text=True, timeout=120)
            assert failed.returncode == 1 and "FAIL" in failed.stdout
        down = subprocess.run(script, capture_output=True, text=True, timeout=60)
        assert down.returncode == 1 and "cannot reach" in down.stdout
    finally:
        service.close()
