#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/data/raw"
MANIFEST="${RAW_DIR}/SHA256SUMS.txt"
RECORDS="${RAW_DIR}/RECORDS"
BASE_URL="https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0"
PARALLEL_DOWNLOADS="${EEG_PARALLEL_DOWNLOADS:-8}"

mkdir -p "${RAW_DIR}"

if [[ ! -s "${MANIFEST}" ]]; then
  curl --fail --location --retry 5 --retry-all-errors \
    --output "${MANIFEST}" \
    "${BASE_URL}/SHA256SUMS.txt"
fi

download_one() {
  local relative_path="$1"
  local expected_hash="$2"
  local destination="${RAW_DIR}/${relative_path}"
  local actual_hash=""

  mkdir -p "$(dirname "${destination}")"

  if [[ -f "${destination}" ]]; then
    actual_hash="$(shasum -a 256 "${destination}" | awk '{print $1}')"
    if [[ "${actual_hash}" == "${expected_hash}" ]]; then
      printf 'Verified (already present): %s\n' "${relative_path}"
      return 0
    fi
  fi

  printf 'Downloading: %s\n' "${relative_path}"
  curl --fail --location --silent --show-error \
    --retry 5 --retry-all-errors --continue-at - \
    --output "${destination}" \
    "${BASE_URL}/${relative_path}"

  actual_hash="$(shasum -a 256 "${destination}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    printf 'Checksum mismatch: %s\n' "${relative_path}" >&2
    return 1
  fi

  printf 'Verified: %s\n' "${relative_path}"
}

export RAW_DIR BASE_URL
export -f download_one

while read -r expected_hash relative_path; do
  printf '%s\0%s\0' "${relative_path}" "${expected_hash}"
done < "${MANIFEST}" | xargs -0 -n 2 -P "${PARALLEL_DOWNLOADS}" bash -c 'download_one "$1" "$2"' _

verification_failed=0
while read -r expected_hash relative_path; do
  actual_hash="$(shasum -a 256 "${RAW_DIR}/${relative_path}" | awk '{print $1}')"
  if [[ "${actual_hash}" == "${expected_hash}" ]]; then
    printf 'Final check OK: %s\n' "${relative_path}"
  else
    printf 'Final check failed: %s\n' "${relative_path}" >&2
    verification_failed=1
  fi
done < "${MANIFEST}"

if [[ "${verification_failed}" -ne 0 ]]; then
  exit 1
fi

# SHA256SUMS validates every downloaded file. RECORDS is the dataset's EDF
# inventory, so check it separately and make any missing EEG recording obvious.
if [[ ! -s "${RECORDS}" ]]; then
  printf 'Missing or empty EDF inventory: %s\n' "${RECORDS}" >&2
  exit 1
fi

missing_edf=0
while IFS= read -r relative_path; do
  [[ -z "${relative_path}" ]] && continue
  if [[ ! -s "${RAW_DIR}/${relative_path}" ]]; then
    printf 'Missing expected EDF: %s\n' "${relative_path}" >&2
    missing_edf=1
  fi
done < "${RECORDS}"

if [[ "${missing_edf}" -ne 0 ]]; then
  exit 1
fi

printf 'All Siena Scalp EEG Database files downloaded, verified, and present in the EDF inventory.\n'
