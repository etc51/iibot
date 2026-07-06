#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-paper-cycle.service" /etc/systemd/system/samosbor-paper-cycle.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-paper-cycle.timer" /etc/systemd/system/samosbor-paper-cycle.timer
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-daily-review.service" /etc/systemd/system/samosbor-daily-review.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-daily-review.timer" /etc/systemd/system/samosbor-daily-review.timer
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-dashboard.service" /etc/systemd/system/samosbor-dashboard.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-microstructure.service" /etc/systemd/system/samosbor-microstructure.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-updater.service" /etc/systemd/system/samosbor-updater.service
sudo install -m 0644 "$ROOT_DIR/deploy/systemd/samosbor-updater.timer" /etc/systemd/system/samosbor-updater.timer

sudo systemctl daemon-reload
sudo systemctl enable --now samosbor-paper-cycle.timer
sudo systemctl enable --now samosbor-daily-review.timer
sudo systemctl enable --now samosbor-dashboard.service
sudo systemctl enable --now samosbor-microstructure.service
sudo systemctl enable --now samosbor-updater.timer
sudo systemctl restart samosbor-dashboard.service
sudo systemctl restart samosbor-microstructure.service
