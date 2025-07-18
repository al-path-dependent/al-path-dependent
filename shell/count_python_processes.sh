while true
do
    ps -ef | grep Python | wc -l
    sleep 0.1
done
