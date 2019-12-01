#!/bin/bash
gdb python3 -ex "dir ../../../../bazel-tensorflow" -ex "run $1"
