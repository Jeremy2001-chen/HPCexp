#!/bin/bash
make test
./check_output.sh ./test "1e-9" "4.5e-4"
