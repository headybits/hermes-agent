#!/usr/bin/env bash
# Build "Hermes Light" (the remote-only desktop client) for macOS from this checkout.
#
#   scripts/build-mac-light.sh [zip|dmg|pack]     (default: zip)
#
# zip and pack need only Node. dmg also needs a Python 3.11+ (dmgbuild runs on it):
# set HERMES_PYTHON, or have python3.11+ on PATH (brew install python@3.12).
# The app is built as a commit build ("Hermes Light <sha7>"): its own app ID and
# userData dir, side by side with any installed Hermes, and no auto-update feed,
# so an official release can never replace this patched build.
# Output lands in apps/desktop/release/. The app is unsigned and not notarized.
set -euo pipefail

format="${1:-zip}"
case "$format" in
  dmg)  npm_script="dist:mac:dmg" ;;
  zip)  npm_script="dist:mac:zip" ;;
  pack) npm_script="pack" ;;
  -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "build-mac-light: format must be dmg, zip or pack" >&2; exit 2 ;;
esac

die() { echo "build-mac-light: $*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "run this on macOS"
command -v git >/dev/null || die "git not found (xcode-select --install)"
# npm ci compiles native modules (node-pty) with node-gyp: needs make + clang.
xcode-select -p >/dev/null 2>&1 && command -v make >/dev/null && command -v clang >/dev/null \
  || die "Xcode Command Line Tools missing (run: xcode-select --install)"
command -v node >/dev/null || die "node not found (needs ^22.22, ^24.11 or >=26)"
command -v npm >/dev/null || die "npm not found"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

# Node engine check, mirroring package.json "engines".
node -e '
  const [ma, mi] = process.versions.node.split(".").map(Number)
  const ok = (ma === 22 && mi >= 22) || (ma === 24 && mi >= 11) || ma >= 26
  if (!ok) { console.error(`build-mac-light: node ${process.versions.node} unsupported (needs ^22.22, ^24.11 or >=26)`); process.exit(1) }
'

# The build stamps HEAD; a dirty tree would package something HEAD does not describe.
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  git status --short --untracked-files=all >&2
  die "the checkout has uncommitted changes; commit or stash them first"
fi

# dmgbuild is a Python tool; the packager runs it through HERMES_PYTHON.
if [ "$format" = dmg ] && [ -z "${HERMES_PYTHON:-}" ]; then
  for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      HERMES_PYTHON="$(command -v "$candidate")"; break
    fi
  done
  [ -n "${HERMES_PYTHON:-}" ] || die "dmg needs Python 3.11+ (macOS ships 3.9). Use the default zip, or: brew install python@3.12"
  export HERMES_PYTHON
fi

# pyproject.toml carries 0.0.0; the release version comes from the newest vX.Y.Z tag.
if [ -z "${HERMES_PAYLOAD_VERSION:-}" ]; then
  tag="$(git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' 2>/dev/null || true)"
  tag="${tag%%+*}"
  [ -n "$tag" ] || die "no vX.Y.Z tag reachable from HEAD; run 'git fetch --tags' or set HERMES_PAYLOAD_VERSION=X.Y.Z"
  HERMES_PAYLOAD_VERSION="${tag#v}"
fi
[[ "$HERMES_PAYLOAD_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "HERMES_PAYLOAD_VERSION must be X.Y.Z (got $HERMES_PAYLOAD_VERSION)"

commit="$(git rev-parse HEAD)"
short="${commit:0:7}"

# A local build never signs with credentials from the caller's shell.
unset APPLE_API_KEY APPLE_API_KEY_ID APPLE_API_ISSUER APPLE_NOTARY_PROFILE APPLE_SIGNING_IDENTITY \
      CSC_LINK CSC_KEY_PASSWORD CSC_NAME CSC_KEYCHAIN
export CSC_IDENTITY_AUTO_DISCOVERY=false
export HERMES_DESKTOP_VARIANT=light
export HERMES_BUILD_COMMIT="$commit"
export HERMES_PAYLOAD_VERSION

echo "build-mac-light: Hermes Light $short (v$HERMES_PAYLOAD_VERSION, $(git rev-parse --abbrev-ref HEAD)), format: $format"
npm ci
npm run "$npm_script" --workspace apps/desktop

echo
echo "build-mac-light: done. Artifacts in $repo/apps/desktop/release:"
ls -1 "$repo/apps/desktop/release" 2>/dev/null || true
cat <<EOF

Install: zip: double-click the .zip and move "Hermes Light $short.app" to Applications.
dmg: open it and drag the app to Applications. pack: the .app is under mac*/ in release.
It is unsigned, so clear the quarantine flag once (or right-click > Open):

  xattr -dr com.apple.quarantine "/Applications/Hermes Light $short.app"

Then: Add connection > SSH > user@your-gateway-host.
EOF
