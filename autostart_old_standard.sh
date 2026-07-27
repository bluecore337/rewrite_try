sleep 5
cd /home/bb/ov_develop
screen \
    -L \
    -Logfile logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog \
    bash -c "./build/ovgimbal_standard_mpc configs/standard_old.yaml "
