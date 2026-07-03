#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-paper-cycle.service" /etc/systemd/system/samosbor-paper-cycle.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-paper-cycle.timer" /etc/systemd/system/samosbor-paper-cycle.timer
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-daily-review.service" /etc/systemd/system/samosbor-daily-review.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-daily-review.timer" /etc/systemd/system/samosbor-daily-review.timer
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-dashboard.service" /etc/systemd/system/samosbor-dashboard.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-microstructure.service" /etc/systemd/system/samosbor-microstructure.service

sudo systemctl daemon-reload
sudo systemctl disable --now samosbor-updater.timer samosbor-updater.service 2>/dev/null || true
sudo systemctl enable --now samosbor-paper-cycle.timer
sudo systemctl enable --now samosbor-daily-review.timer
sudo systemctl enable --now samosbor-dashboard.service
sudo systemctl enable --now samosbor-microstructure.service
sudo systemctl restart samosbor-dashboard.service
sudo systemctl restart samosbor-microstructure.service
