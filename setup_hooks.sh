#!/usr/bin/env bash

# Copy hook scripts to .git/hooks
# shellcheck disable=SC2016
hook='#!/usr/bin/env bash

GREEN="\e[32m"
BOLD_LIGHT_GREEN="\e[1;\e[92m"
RED="\e[31m"
WHITE="\e[97m"
ENDCOLOR="\e[0m"

check() {
    local command="$1"
    local fix="$2"
    echo -e "\n${WHITE}Checking \"$command\"...${ENDCOLOR}"
    if ! $command; then
        echo -e "\n${RED}ERROR: ${ENDCOLOR}${WHITE}\"$command\"${ENDCOLOR} found issues. Please try $fix or fix them manually."
        exit 1
    fi
    echo -e "${GREEN}$command passed!${ENDCOLOR}"
}
check "cargo fmt -- --check" "cargo fmt"
check "cargo clippy --all-targets --all-features -- -D warnings" "cargo clippy --fix"
check "cargo check --all-targets" "cargo fix"
echo -e "\n${BOLD_LIGHT_GREEN}All checks passed!${ENDCOLOR}\n"
'

echo "$hook" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

