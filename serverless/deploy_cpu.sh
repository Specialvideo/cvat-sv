#!/bin/bash
# Sample commands to deploy nuclio functions on CPU

set -eu

. copyweights.sh
. getnctl.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}

export DOCKER_BUILDKIT=1

docker build -t cvat.openvino.base "$SCRIPT_DIR/openvino/base"

./nuctl create project cvat --platform local

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/**/function.yaml
do
    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$(dirname "$func_root")")"

    if [ -f "$func_root/Dockerfile" ]; then
        docker build -t "cvat.${func_rel_path//\//.}.base" "$func_root"
    fi

    echo "Deploying $func_rel_path function..."
    ./nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local
done

./nuctl get function --platform local

