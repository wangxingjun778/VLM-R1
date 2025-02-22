ps -ef | grep grpo_rec | grep -v grep | awk '{print $2}' | xargs kill
