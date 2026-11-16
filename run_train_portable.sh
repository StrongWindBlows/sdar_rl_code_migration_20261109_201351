#!/usr/bin/env bash
set -euo pipefail
MIGRATION_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
    TORCHRUN_BIN=$(command -v torchrun || true)
fi
if [[ -z "$TORCHRUN_BIN" ]]; then
    printf '%s\n' '请先激活包含 PyTorch 的 Python 环境，或设置 TORCHRUN_BIN。' >&2
    exit 1
fi
export TORCHRUN_BIN
exec bash "$MIGRATION_ROOT/sdar_justgmpo/run_train.sh" "$@"
