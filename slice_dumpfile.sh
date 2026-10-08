#!/usr/bin/env bash
# Extract a block of lines from the middle of a large ASCII file into a smaller "subset" file.
#
# Usage: slice_dumpfile.sh INPUT OUTPUT N_LINES [N_HEADER_LINES]
#   INPUT           large ascii input file
#   OUTPUT          subset file to create
#   SIZE_MB         target size of the extracted block in MB (1 MB = 1048576 bytes), decimals allowed, e.g. 0.5
#   N_HEADER_LINES  optional: number of leading lines of INPUT to copy to the top of OUTPUT
#                   (e.g. 1 for a csv header), not counted in SIZE_MB. Default: 0
 
set -eu
 
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 INPUT OUTPUT SIZE_MB [N_HEADER_LINES]" >&2
    exit 1
fi
 
in="$1"
out="$2"
mb="$3"
hdr="${4:-0}"
 
[ -f "$in" ] || { echo "Error: input file '$in' not found." >&2; exit 1; }
[[ "$mb" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] || { echo "Error: SIZE_MB must be a positive number." >&2; exit 1; }
[[ "$hdr" =~ ^[0-9]+$ ]] || { echo "Error: N_HEADER_LINES must be a non-negative integer." >&2; exit 1; }
[ "$in" != "$out" ] || { echo "Error: input and output must differ." >&2; exit 1; }
 
target=$(awk -v m="$mb" 'BEGIN { printf "%d", m * 1048576 }')
[ "$target" -gt 0 ] || { echo "Error: SIZE_MB is too small." >&2; exit 1; }
 
total=$(wc -c < "$in")
hdr_bytes=0
if [ "$hdr" -gt 0 ]; then
    hdr_bytes=$(head -n "$hdr" "$in" | wc -c)
fi
body=$(( total - hdr_bytes ))   # bytes available after the header
 
if [ "$target" -ge "$body" ]; then
    echo "Error: requested $target bytes, but only $body bytes available after $hdr header line(s)." >&2
    exit 1
fi
 
# byte offset of the block start, centered in the non-header part of the file
start=$(( hdr_bytes + (body - target) / 2 ))
 
{
    if [ "$hdr" -gt 0 ]; then
        head -n "$hdr" "$in"
    fi
    # jump to the byte offset, drop the (probably partial) first line, then copy whole lines
    # until the target size would be exceeded. awk exiting early makes tail get SIGPIPE, which is fine.
    tail -c +"$start" "$in" | LC_ALL=C awk -v max="$target" '
        NR == 1 { next }
        { len = length($0) + 1; if (total + len > max) exit; total += len; print }
    '
} > "$out"
 
size=$(wc -c < "$out")
echo "Wrote $size bytes (~$(awk -v s="$size" 'BEGIN { printf "%.2f", s / 1048576 }') MB) from around byte offset $start of $total to '$out'"
