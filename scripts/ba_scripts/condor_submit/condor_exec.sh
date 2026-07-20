#!/bin/bash

Executable=$1
OtherArgs=("${@:2}")

RESET="\033[0m"
BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
BGBLACK='\033[0;40m'
BGRED='\033[0;41m'
BGGREEN='\033[0;42m'
BGYELLOW='\033[0;43m'
BGBLUE='\033[0;44m'
BGPRUPLE='\033[0;45m'
BGCYAN='\033[0;46m'
BGWHITE='\033[0;47m'

printf "${PURPLE}[ condor_exec.sh ]${WHITE} Start condor bash executable ${CYAN}condor_exec.sh${RESET}\n"
TIMESTAMP="`date +%Y-%m-%d_%H-%M-%S`"
printf "${PURPLE}[ condor_exec.sh ]${WHITE} Start timestamp is ${CYAN}${TIMESTAMP}${RESET}\n"
printf "${PURPLE}[ condor_exec.sh ]${WHITE} User bash executable = ${CYAN}"
echo ${Executable}
printf "${RESET}\n"
printf "${PURPLE}[ condor_exec.sh ]${WHITE} Additional arguments = ${CYAN}"
echo ${OtherArgs[@]}
printf "${RESET}\n"
printf "${PURPLE}[ condor_exec.sh ]${WHITE} Start user bash executable with the following command ${CYAN}\n"
echo "source ${Executable} ${OtherArgs[@]}"
printf "${RESET}\n"
printf "${PURPLE}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${RESET}\n"

###--- run normally:
source ${Executable} ${OtherArgs[@]}
###--- run and measure memory utilization (is printed in stderr):
# /usr/bin/time --verbose source ${Executable} ${OtherArgs[@]}

printf "${PURPLE}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${RESET}\n"
printf "${PURPLE}[ condor_exec.sh ]${WHITE} Finished user bash executable${RESET}\n"
TIMESTAMP="`date +%Y-%m-%d_%H-%M-%S`"
printf "${PURPLE}[ condor_exec.sh ]${WHITE} Stop timestamp is ${CYAN}${TIMESTAMP}${RESET}\n"
printf "${PURPLE}[ condor_exec.sh ]${WHITE} finish condor bash executable ${CYAN}condor_exec.sh${RESET}\n"

