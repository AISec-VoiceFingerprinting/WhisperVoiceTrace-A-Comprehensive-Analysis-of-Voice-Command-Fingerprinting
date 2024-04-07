#!/bin/bash

trap "clean_exit" SIGINT
clean_exit() {
  echo -e "\nStopping capture and exiting safely\n"
  sudo pkill -2 tcpdump
  exit 1
}

rasp_num=1



# Prompt for data directory paths  
datadirname=" "                         # file path of datadir
dataset_dir=$datadirname
commanddirname=" "    # file path of commands
command_dir=$commanddirname
wakeworddirname=" "          # file path of wake_word
wake_word_dir=$wakeworddirname
resultdirname=" "             # file path of result
result_dir=$resultdirname

PS3="Select the command&wakeword subdirectory to use"
command_subdirs=$command_dir/*
wake_word_file=$wake_word_dir/*

# For each voice
for voice in {301..400}; do
  # For each command subdirectory
  for result_dir in ${result_dir[@]}; do
    # sudo rm "$result_dir/sarah_${voice}" # Delete stale output files of current voice in current command subdirectory
    variant=1 # Reset voice variant counter of current voice
    
    # For each variant file of current voice in current command subdirectory
    for variant_file in ${command_dir}; do
      # Start the capture
      echo -e "\nCapturing $command_dir/sarah_$(printf '%03d' ${voice}).pcap\n"
      sudo tcpdump -U -i wlan0 -w $result_dir/sarah_$(printf '%03d' ${voice}).pcap &   
      paplay $wake_word_file
      echo -e ${variant_file}
      variant_file=${variant_file}/sarah_$(printf '%03d' ${voice}).wav
      echo -e ${variant_file}
      paplay $variant_file

      counter=0
      
      while ! timeout --foreground 60s sox -t alsa plughw:2 -d -c 1 -r 44100 $result_dir/sarah_$(printf '%03d' ${voice}).wav silence 1 0.1 5% 1 3.0 5%; do  #output 파
        echo -e "while"
        # Clean up from failed capture
        sudo pkill -2 tcpdump
        sudo rm "$result_dir/sarah_$(printf '%03d' ${voice})"*
        
        # Move on the next file if we've tried 5 times
        ((counter++))
        echo -e ${counter}
        if [[ $counter -eq 5 ]]; then
          echo -e "\nMoving on to the next file\n"
          echo "sarah_$(printf '%03d' ${voice}).pcap" >> " "  # Check error pcaps
          break
        fi

        # Start the redo capture
        echo -e "\nCapturing $command_subdir/sarah_$(printf '%03d' ${voice}).pcap\n"
        sudo tcpdump -U -i wlan0 -w $result_dir/sarah_$(printf '%03d' ${voice}).pcap & 
        paplay $wake_word_file
        paplay $variant_file
      done

      sleep 2
      sudo pkill -2 tcpdump
      ((variant++))
    done

    sudo chown $USER:$USER "$command_dir/"* # Fix ownership of files
  done
done
