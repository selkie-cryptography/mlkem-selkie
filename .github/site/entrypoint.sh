#!/bin/sh
# Ensure the report directories exist on the persistent volume. The volume mount
# replaces /data at runtime, so these can't be created at image-build time.
mkdir -p \
    /data/coverage /data/bench /data/instructions /data/cycles \
    /data/mutants /data/dudect /data/tacet /data/ctgrind \
    /data/kat /data/wycheproof /data/deny /data/docs \
    /data/flamegraphs /data/api
exec nginx -g 'daemon off;'
