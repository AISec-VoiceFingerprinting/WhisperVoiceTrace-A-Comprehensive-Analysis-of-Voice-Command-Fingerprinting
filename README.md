# WhisperVoiceTrace-A-Comprehensive-Analysis-of-Voice-Command-Fingerprinting

⚠️ Experimental - PLEASE BE CAREFUL. Intended for reasearch purposes only.⚠️
This repository contains code and data of the paper **WhisperVoiceTrace: A Comprehensive Analysis of Voice Command Fingerprinting**

## Installation
```py
pip3 install -r requirements.txt
sudo apt install graphviz
```

## Dataset
The WhiVo Alexa Dataset referenced in the paper can be accessed through the link provided : <\br>
  dataset1 : data_mon_300.npz - monitored set consisting of 300 instances per class.<\br>
  dataset2 : data_mon_1000.npz - monitored set consisting of 1000 instances per class.<\br>
  dataset3 : data_unmon_1000.npz - unmonitored set consisting of 1000 instances per class.<\br>
  dataset4 : data_location.npz - voice command traces by configuring various locations, including US, UK, Germany, India, and Canada, on five Alexa devices <\br>
The WhiVo Google Dataset referenced in the paper can be accessed through the link provided : https://drive.google.com/drive/folders/1UMYmwv4INdThN4s9c1mFNvcPEyUSpTlQ?usp=sharing<\br>
The SHAME dataset referenced in the paper can be accessed through the link provided :  https://drive.google.com/file/d/1K19SDZ3IdvAv_0rK6mG9d8WTpHg85gzV/view?usp=sharing<\br>
The DeepVC dataset referenced in the paper can be accessed through the link provided : https://drive.google.com/drive/folders/1l-fSX9VdZH5kF9z7gm82xgYX5ca0kRI0?usp=sharing<\br>

The information regarding commands used for category fingerprinting, location, dynamic/static, and human experiments can be found in the paper (Table 8, Table 9) and the provided link(https://docs.google.com/spreadsheets/d/15XVeFjGMaWQU9f6e-6OnUmUjQ9kS2anprYcihniMr6I/edit#gid=0). You can check the command number and voice_name from the array corresponding to 'label_name' in the npz file. Using the command number provided within the link, you can also retrieve the content of the respective command.<\br>

## Usage
1. Load Data and Convert the simple features to WhiVo features
```py
python3 Data_npz.py
```

2. 



