#!/bin/bash
## TEST VARS.sh
##   by Tim Müller
## 
## Simple script to see the output of various environment variables when run.
## 

# Parse the list of envs from ze CLI
for arg in $@; do
    echo "$arg=\"${!arg}\""
    echo ""
done

