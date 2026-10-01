#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv não encontrado no WSL. Instale-o: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2
  exit 1
fi

if (($# == 0)); then
  targets=(common fedosd fedup fastfedul)
else
  targets=("$@")
fi

for target in "${targets[@]}"; do
  case "$target" in
    common) directory="$root" ;;
    fedosd) directory="$root/repositories/FedOSD" ;;
    fedup) directory="$root/repositories/FedUP-code" ;;
    fastfedul) directory="$root/repositories/fastFedUL" ;;
    *) echo "Ambiente desconhecido: $target" >&2; exit 2 ;;
  esac

  echo "==> $target: uv sync --locked"
  (cd "$directory" && uv sync --locked)
done

echo 'Ambientes preparados. Use uv run --locked dentro da pasta de cada projeto.'
