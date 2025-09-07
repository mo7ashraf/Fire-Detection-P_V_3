#!/usr/bin/env bash
set -e
mkdir -p code/data/hpwren_raw code/data/hpwren_voc
echo "[*] Downloading HPWREN/AIFM (VOC) and clouds negatives via gdown..."
gdown --id 1sEB77bfp2yMkgsSW9703vwDHol_cK6D5 -O code/data/hpwren_raw/v1.tar.gz
gdown --id 1mUgVvnctpdZ8VZgihIDMoeUgkhsx1iAH -O code/data/hpwren_raw/clouds.tar.gz
tar -xzf code/data/hpwren_raw/v1.tar.gz -C code/data/hpwren_voc || true
tar -xzf code/data/hpwren_raw/clouds.tar.gz -C code/data/hpwren_voc || true
echo "[✓] Extracted to code/data/hpwren_voc"
