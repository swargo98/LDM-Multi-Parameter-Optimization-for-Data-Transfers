# #!/bin/bash
set -euo pipefail

output_dir="/mnt/nvme0n1/src"
file_count=20
file_size=1024      # MiB
parallel_jobs=10

mkdir -p "$output_dir"

create_file() {
  local file_index=$1
  local path="$output_dir/random_file_$file_index"
  echo "Generating $path (${file_size} MiB)..."
  # AES-CTR pseudorandom stream, exact size via fullblock
  openssl enc -aes-256-ctr -nosalt -pass pass:"seed-$file_index" </dev/zero 2>/dev/null \
    | dd of="$path" bs=1M count="$file_size" iflag=fullblock status=none conv=fdatasync
  # sanity: show exact size in bytes
  stat -c '%n %s bytes' "$path"
}

export -f create_file
export output_dir file_size

seq 1 "$file_count" | xargs -n1 -P "$parallel_jobs" -I{} bash -c 'create_file "$@"' _ {}
echo "Created $file_count files of ${file_size} MiB in $output_dir"
#!/usr/bin/env bash
# set -euo pipefail

# ################################################################################
# # Config – tweak these if you need a different mix
# ################################################################################
# output_dir="/mnt/nvme0n1/files_mixed"
# target_total_gb=40

# # Byte-share across categories (adjust if needed)
# pct_small=5                  # 100 KB–1 MB
# pct_medium=80                # 10–100 MB
# pct_large=$((100 - pct_small - pct_medium))  # 1–2 GB

# parallel_jobs="$(nproc)"     # writers in parallel
# DATA_MODE="${DATA_MODE:-random}"  # random|zero (env override)

# # Size ranges (bytes)
# SMALL_MIN=$((100 * 1024))                # 100 KB
# SMALL_MAX=$((  1 * 1024 * 1024))         #   1 MB
# MED_MIN=$(( 10 * 1024 * 1024))           #  10 MB
# MED_MAX=$((100 * 1024 * 1024))           # 100 MB
# LARGE_MIN=$((  1 * 1024 * 1024 * 1024))  #   1 GB
# LARGE_MAX=$((  2 * 1024 * 1024 * 1024))  #   2 GB
# ################################################################################

# mkdir -p "$output_dir"

# total_bytes=$((target_total_gb * 1024 * 1024 * 1024))
# bytes_small=$((total_bytes * pct_small   / 100))
# bytes_medium=$((total_bytes * pct_medium / 100))
# bytes_large=$((total_bytes - bytes_small - bytes_medium))

# ################################################################################
# # Helpers
# ################################################################################
# # Better-than-$RANDOM range RNG
# rand_int() {  # MIN MAX -> prints integer in [MIN,MAX]
#   local min=$1 max=$2 range=$((max - min + 1))
#   # stitch multiple $RANDOM calls to widen the entropy
#   local r=$(( ( (RANDOM << 15 | RANDOM) << 15 | RANDOM ) & 0x7fffffffffffffff ))
#   echo "$(( min + (r % range) ))"
# }

# # Write exactly SIZE bytes of random content to PATH (no pipes → no SIGPIPE)
# write_random_bytes() { # SIZE PATH
#   local sz=$1 path=$2
#   if command -v openssl >/dev/null 2>&1; then
#     # OpenSSL RAND writes exactly sz bytes directly to file
#     openssl rand -out "$path" "$sz" 2>/dev/null
#   elif [ -r /dev/urandom ]; then
#     # Fallback: kernel RNG (can be slower)
#     head -c "$sz" /dev/urandom > "$path"
#   else
#     # Last resort: zeros
#     dd if=/dev/zero of="$path" bs=1M count=$((sz/1048576)) status=none
#     local rem=$((sz % 1048576))
#     if (( rem > 0 )); then dd if=/dev/zero of="$path" bs=1 count="$rem" \
#         oflag=append conv=notrunc status=none; fi
#   fi
# }


# # Write SIZE bytes of zeros quickly
# write_zero_bytes() { # SIZE PATH
#   local sz=$1 path=$2
#   if command -v fallocate >/dev/null 2>&1; then
#     fallocate -l "$sz" "$path"
#   else
#     dd if=/dev/zero of="$path" bs=1M count=$((sz/1048576)) status=none
#     local rem=$((sz % 1048576))
#     if (( rem > 0 )); then dd if=/dev/zero of="$path" bs=1 count="$rem" oflag=append conv=notrunc status=none; fi
#   fi
# }

# make_file() {  # SIZE_BYTES PATH
#   local sz=$1 path=$2
#   case "$DATA_MODE" in
#     random) write_random_bytes "$sz" "$path" ;;
#     zero)   write_zero_bytes   "$sz" "$path" ;;
#     *)      echo "Unknown DATA_MODE='$DATA_MODE' (use random|zero)"; exit 1 ;;
#   esac
# }

# # Generate "size:path" lines until a byte budget is met
# generate_category() {  # BYTES_BUDGET MIN MAX PREFIX
#   local budget=$1 min=$2 max=$3 prefix=$4 produced=0 size ts name
#   while (( produced < budget )); do
#     size=$(rand_int "$min" "$max")
#     (( produced += size ))
#     ts=$(date +%s%N)
#     name="${prefix}_${ts}_$((size/1024))KB"   # include size in name for sanity
#     printf "%s:%s\n" "$size" "$output_dir/$name"
#   done
# }

# ################################################################################
# # Build work list
# ################################################################################
# work_list="$(mktemp)"
# {
#   generate_category "$bytes_small"  "$SMALL_MIN"  "$SMALL_MAX"  "small"
#   generate_category "$bytes_medium" "$MED_MIN"    "$MED_MAX"    "medium"
#   generate_category "$bytes_large"  "$LARGE_MIN"  "$LARGE_MAX"  "large"
# } > "$work_list"

# task_total=$(wc -l <"$work_list")
# echo "Planned files: $task_total | Target size: ${target_total_gb}GB | Mode: $DATA_MODE"
# echo "Mix: small=${pct_small}% medium=${pct_medium}% large=${pct_large}%"
# echo "Output dir: $output_dir"
# echo

# ################################################################################
# # Create files with a progress indicator
# ################################################################################
# export -f make_file write_random_bytes write_zero_bytes
# export DATA_MODE

# if command -v parallel &>/dev/null; then
#   # Best: GNU parallel progress bar + ETA
#   parallel --jobs "$parallel_jobs" --bar --colsep ':' \
#            make_file {1} {2} :::: "$work_list"

# elif command -v pv &>/dev/null; then
#   # Fallback: pv line-based progress bar
#   cat "$work_list" | pv -l -s "$task_total" | \
#     xargs -n1 -P "$parallel_jobs" -I{} bash -c \
#       'IFS=: read -r sz path <<< "{}"; make_file "$sz" "$path"'
# else
#   # Minimal: print progress every 50 files
#   counter=0
#   cat "$work_list" | xargs -n1 -P "$parallel_jobs" -I{} bash -c \
#     'IFS=: read -r sz path <<< "{}"; make_file "$sz" "$path"; echo done' | \
#     while read -r _; do
#       ((counter++))
#       if (( counter % 50 == 0 )); then
#         printf " %d / %d files done (%.1f%%)\n" \
#                "$counter" "$task_total" "$(bc -l <<<"$counter*100/$task_total")"
#       fi
#     done
# fi

# rm -f "$work_list"

# echo
# echo "Done. Created $(find "$output_dir" -type f | wc -l) files."
# du -sh "$output_dir"
