#!/bin/bash
set -euo pipefail

EA="${EA:-/root/dev/eacompute/target/release/ea}"
KERNEL_DIR="$(dirname "$0")/kernels"
LIB_DIR="$(dirname "$0")/lib"

mkdir -p "$LIB_DIR"

if [ ! -f "$EA" ]; then
    echo "ERROR: ea compiler not found at $EA"
    exit 1
fi

echo "=== eabrain kernel build ==="
echo "Compiler: $EA"
echo "Kernels:  $KERNEL_DIR"
echo "Output:   $LIB_DIR"
echo ""

build_kernel() {
    local src="$1"
    local name="$(basename "$src" .ea)"
    local extra_flags="${2:-}"
    echo "  $name ..."
    "$EA" "$src" --lib -o "$LIB_DIR/lib${name}.so" $extra_flags
    "$EA" bind "$src" --python > "$LIB_DIR/_${name}_bind.py"
}

build_kernel "$KERNEL_DIR/scan.ea"
build_kernel "$KERNEL_DIR/search.ea"
build_kernel "$KERNEL_DIR/fuzzy.ea"
build_kernel "$KERNEL_DIR/ref_search.ea"

echo ""
echo "Done. Built $(ls "$LIB_DIR"/*.so 2>/dev/null | wc -l) shared libraries."
