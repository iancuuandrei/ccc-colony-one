#!/bin/bash
for p in $(pgrep -f 'train.py --data'); do echo "$p $(tr '\0' ' ' < /proc/$p/cmdline | grep -o -- '--out [^ ]*\|--qmix [^ ]*\|--opp-aux [^ ]*' | tr '\n' ' ')"; done
