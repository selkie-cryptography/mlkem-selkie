#!/bin/sh
# Ensure the data directories exist on the persistent volume. The volume
# mount replaces /data at runtime, so these can't be created at image-build
# time. Keep in sync with ALL_KINDS in .github/scripts/ci-upload.rs and the
# location blocks in nginx.conf.
mkdir -p \
    /data/coverage \
    /data/bench-generic /data/bench-avx2 /data/bench-neon \
    /data/instructions-generic /data/instructions-avx2 \
    /data/cycles-generic /data/cycles-avx2 \
    /data/mutants /data/mutants-aarch64 \
    /data/dudect /data/tacet /data/ctgrind \
    /data/deny /data/unsafe /data/platform /data/kat \
    /data/rustdoc
exec nginx -g 'daemon off;'
