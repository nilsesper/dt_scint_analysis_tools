#!/bin/bash

dataset_name=$1


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

CURRENT_WORKING_DIR="`pwd`"


printf "${GREEN}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${RESET}\n"
printf "${GREEN}>>>>>>${WHITEBOLD} dt_scint_analysis_tools ${RESET}\n"

printf "${GREEN}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${RESET}\n"
printf "${GREEN}>>>>>> $ {WHITE} BASIC INFO ${RESET}\n"
printf "${GREEN}>>>>>> ${WHITE}   InputFile = ${InputFile} ${RESET}\n"
printf "${GREEN}>>>>>> ${WHITE}   OutputFile = ${OutputFile} ${RESET}\n"
printf "${GREEN}>>>>>> ${WHITE}   ParamsFile = ${ParamsFile} ${RESET}\n"
printf "${GREEN}>>>>>> ${WHITE}   ClusterId = ${ClusterId} ${RESET}\n"
printf "${GREEN}>>>>>> ${WHITE}   ProcId = ${ProcId} ${RESET}\n"

printf "${GREEN}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${RESET}\n"
printf "${GREEN}>>>>>> ${WHITE} SOURCING ENVIRONMENT ${RESET}\n"

cd /home/home1/institut_3a/tacke/Bachelor_thesis/dt_scint_analysis_tools
source env.sh

printf "${GREEN}>>>>>> ${WHITE}STARTING SCRIPT AND TIMING IT ${RESET}\n"

cd /home/home1/institut_3a/tacke/Bachelor_thesis/dt_scint_analysis_tools

###--- run normally:
python scripts/ba_scripts/create_all_pcls.py --dataset_name ${dataset_name}
###--- run and measure resource utilization (is logged in stderr):
# /usr/bin/time --verbose  python scripts/ba_scripts/create_all_pcls.py --datasetname ${dataset_name}

printf "${GREEN}>>>>>> ${WHITE} DONE ${RESET}\n"
cd ${CURRENT_WORKING_DIR}
printf "${GREEN}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${RESET}\n"