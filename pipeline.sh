#!/usr/bin/env bash
# Build the astronaut end to end: source textures -> rig + correctives -> SimReady USD -> Blender file -> assets/
# -> Newton example tests -> Cycles renders and README images.
# Usage: ./pipeline.sh [step ...] with steps from
#   [textures|rig|poses|package|blend|assets|update|test|grasp|walk|hero|screenshots|gallery|images|release|all]   (default: all)
# PY needs bpy 5.2, usd-core, scipy and pillow; NEWTON_PY needs newton[examples] (both default to .venv).
set -euo pipefail
cd "$(dirname "$0")"
export BLENDER=${BLENDER:-blender}   # Blender 5.2 executable (override with BLENDER=/path/to/blender)
PY=${PY:-.venv/bin/python}
NEWTON_PY=${NEWTON_PY:-$PY}

textures() { $PY scripts/fetch_textures.py --verify || $PY scripts/fetch_textures.py; }
rig()      { $PY scripts/build_rig.py && $PY scripts/correctives.py; }        # -> build/astronaut_rig.usd
poses()    { $PY scripts/poses.py; }                                         # -> build/hero_<pose>.usd
package()  { $PY scripts/package_simready.py; }                              # -> build/asset/astronaut
blend()    { $BLENDER -b --factory-startup -P scripts/export_blend.py; }     # -> build/blender/astronaut.blend
# Copy build outputs into the committed assets/ folder (Git LFS). ASTRONAUT_ASSETS picks formats (default: both).
assets()   { mkdir -p assets
             for f in ${ASTRONAUT_ASSETS:-usd blend}; do
               case $f in
                 usd)   rsync -a --delete build/asset/astronaut/ assets/astronaut/ ;;
                 blend) mkdir -p assets/blender && cp build/blender/astronaut.blend assets/blender/ ;;
               esac; echo "assets: updated $f"
             done; }
update()   { rig; package; blend; assets; }   # rebuild every format and refresh assets/
# Headless checks of the Newton examples (they load assets/)
test()     { for e in ik poses walk; do $NEWTON_PY examples/newton/example_astronaut_$e.py --viewer null --test --num-frames 120; done
             $NEWTON_PY examples/newton/example_astronaut_grasp.py --viewer null --test --num-frames 320; }
grasp()    { $NEWTON_PY examples/newton/utils/record_grasp.py; }            # physics grasp -> build/grasp
walk()     { $NEWTON_PY examples/newton/utils/record_walk.py; }             # CMU mocap walk -> build/walk
# Cycles: pose shots, grasp close-ups and the grasp GIF frames -> renders/hero, renders/grasp
hero()     { poses; grasp; walk
             $BLENDER -b --factory-startup -P scripts/render_hero.py
             $BLENDER -b --factory-startup -P scripts/render_hero.py -- --grasp
             $BLENDER -b --factory-startup -P scripts/render_hero.py -- --walk --samples 96; }
screenshots() { $NEWTON_PY examples/newton/utils/screenshots.py; }          # Newton GL viewer -> renders/newton
gallery()  { $PY scripts/gallery.py; }                                       # renders -> docs/images (JPEG, GIF)
images()   { hero; screenshots; gallery; package; assets; }                  # README images + asset thumbnail
# Prepare a release for the version in VERSION: rebuild, refresh assets/ (stamped with the version) and build the
# zips CI will publish into build/release/v<VERSION>/ for a local check. .github/workflows/release.yml publishes
# them when the tag is pushed.
release()  { update; $PY scripts/make_release.py; v=$(cat VERSION)
             echo "next: git add -A assets VERSION && git commit -m \"Release v$v\" && git tag v$v && git push origin main v$v"; }

for step in "${@:-all}"; do
  case $step in
    all) textures; rig; package; blend; assets; test ;;
    *) $step ;;
  esac
done
