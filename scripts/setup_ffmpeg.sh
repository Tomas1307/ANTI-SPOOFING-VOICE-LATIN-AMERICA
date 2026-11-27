#!/usr/bin/env bash
# Fetch the exact pinned ffmpeg static build used to generate the released
# MARSA corpus's codec-degradation augmentation layer, and verify it by
# checksum rather than by trusting whatever "latest release" happens to be
# at download time.
#
# WHY A SCRIPT INSTEAD OF COMMITTING THE BINARY
# ----------------------------------------------
# The static build is ~70-100 MB and is not something git should carry
# directly. A pinned, checksum-verified download script is the standard way
# to keep Code Availability both true (the exact mechanism to obtain the
# exact build IS in the repository) and the repository itself lightweight.
#
# WHY CHECKSUM VERIFICATION MATTERS HERE SPECIFICALLY
# ----------------------------------------------------
# johnvansickle.com/ffmpeg/releases/ always serves whatever the CURRENT
# ffmpeg release is under a fixed filename (ffmpeg-release-amd64-static.tar.xz).
# At the time this script was written, that happened to be version 7.0.2,
# the version MARSA was generated and validated against (decision-log,
# 2026-08-02: AAC container fix, AMR-NB via pinned static ffmpeg, iLBC
# replaced by Speex). Once ffmpeg publishes a newer release, that same URL
# will silently start serving a different version. The checksum below pins
# this script to the exact 7.0.2 archive; if it ever stops matching, that
# means the live URL has moved on, and this script intentionally fails
# rather than installing a different, unvalidated ffmpeg build.
#
# If the checksum check fails because of exactly that version drift, the
# 7.0.2 archive should eventually reappear under
# https://johnvansickle.com/ffmpeg/old-releases/ (it listed versions up to
# 6.0.1 there as of 2026-09-30); update FFMPEG_URL to that archived path
# once available, or source the same 7.0.2 archive from another mirror and
# verify it against the same checksum before changing it in this script.
#
# USAGE (run on ml-server03, from the repository root):
#   bash scripts/setup_ffmpeg.sh
#   export MARSA_FFMPEG_BINARY="$(pwd)/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"

set -euo pipefail

FFMPEG_VERSION="7.0.2"
FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
EXPECTED_MD5="7fa72b652e19bf84c9461e332ea1cdf3"

TOOLS_DIR="tools"
TARGET_DIR="${TOOLS_DIR}/ffmpeg-${FFMPEG_VERSION}-amd64-static"
ARCHIVE_PATH="${TOOLS_DIR}/ffmpeg-release-amd64-static.tar.xz"

mkdir -p "${TOOLS_DIR}"

if [ -x "${TARGET_DIR}/ffmpeg" ]; then
    echo "Already present: ${TARGET_DIR}/ffmpeg"
    exit 0
fi

echo "Downloading ffmpeg ${FFMPEG_VERSION} static build..."
curl -fL --output "${ARCHIVE_PATH}" "${FFMPEG_URL}"

echo "Verifying checksum..."
ACTUAL_MD5="$(md5sum "${ARCHIVE_PATH}" | awk '{print $1}')"
if [ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]; then
    echo "ERROR: checksum mismatch." >&2
    echo "  expected: ${EXPECTED_MD5}" >&2
    echo "  actual:   ${ACTUAL_MD5}" >&2
    echo "This usually means johnvansickle.com has moved on to a newer" >&2
    echo "ffmpeg release under the same URL. Fetch ffmpeg ${FFMPEG_VERSION}" >&2
    echo "specifically (see the comment header of this script) rather than" >&2
    echo "whatever 'latest' currently resolves to." >&2
    rm -f "${ARCHIVE_PATH}"
    exit 1
fi

echo "Extracting to ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"
tar -xf "${ARCHIVE_PATH}" --strip-components=1 -C "${TARGET_DIR}"
rm -f "${ARCHIVE_PATH}"

chmod +x "${TARGET_DIR}/ffmpeg"
echo "Done. Set this before running the augmentation pipeline:"
echo "  export MARSA_FFMPEG_BINARY=\"$(pwd)/${TARGET_DIR}/ffmpeg\""
