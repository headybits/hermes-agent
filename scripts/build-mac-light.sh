#!/usr/bin/env bash
# Build "Hermes Light" (the remote-only desktop client) for macOS from this checkout.
#
#   scripts/build-mac-light.sh [dmg|zip|pack]     (default: dmg)
#
# Node-only path: no host Python needed (scripts/build-bundle.sh wants 3.11+).
# The app is built as a commit build ("Hermes Light <sha7>"): its own app ID and
# userData dir, side by side with any installed Hermes, and no auto-update feed,
# so an official release can never replace this patched build.
# Output lands in apps/desktop/release/. The app is unsigned and not notarized.
set -euo pipefail

format="${1:-dmg}"
case "$format" in
  dmg)  npm_script="dist:mac:dmg" ;;
  zip)  npm_script="dist:mac:zip" ;;
  pack) npm_script="pack" ;;
  -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "build-mac-light: format must be dmg, zip or pack" >&2; exit 2 ;;
esac

die() { echo "build-mac-light: $*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "run this on macOS"
command -v git >/dev/null || die "git not found (xcode-select --install)"
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

Install: open the DMG and drag "Hermes Light $short" to Applications. It is unsigned, so
clear the quarantine flag once (or right-click > Open):

  xattr -dr com.apple.quarantine "/Applications/Hermes Light $short.app"

Then: Add connection > SSH > user@your-gateway-host.
EOF
