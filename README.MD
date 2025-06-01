# AI Voice Assistant

A Python-based voice assistant that uses speech recognition and text-to-speech capabilities powered by OpenAI, ElevenLabs, and Picovoice technologies.

## Features

- Wake word detection using Picovoice Porcupine
- Speech-to-text conversion 
- Natural language processing with OpenAI GPT-4
- High-quality text-to-speech synthesis via ElevenLabs
- Configurable silence detection and audio parameters
- Response caching for improved performance

## Prerequisites

- Python 3.11
- RaspberryPi Zero (or higher) with ReSpeaker 2-Mics Pi HAT
- API keys for:
  - OpenAI Platform
  - ElevenLabs
  - Picovoice Console

## Installation

1. Install ReSpeaker 2-Mics Pi HAT drivers:
   - Follow the official documentation: [ReSpeaker 2 Mics Pi HAT](https://wiki.seeedstudio.com/respeaker_2_mics_pi_hat_raspberry_v2/) - this might not be enough if you use a newer kernel version.
   - To fix this issue, you can use a script that automates the installation process: [Pi Voice Assistant Script](https://github.com/dihan/pi-voice-assistant):
   ```bash
    git clone https://github.com/dihan/pi-voice-assistant
    chmod +x install.sh
    sudo ./install.sh```
1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/piotrb9/personal-assistant.git
cd personal-assistant
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configure the application:
1. Copy config.example.json to config.json
2. Add your API keys to config.json:
   - Get OpenAI key from OpenAI Platform
   - Get ElevenLabs key from ElevenLabs
   - Get Picovoice access key from Picovoice Console

## Run
.

## Configuration
The `config.json` file contains important settings:  
#### API Configuration:  
- access_key: Picovoice access key
- openai_api_key: OpenAI API key
- elevenlabs_api_key: ElevenLabs API key
#### Model Settings:  
- openai_model_name: GPT model to use
- openai_tts_model: Text-to-speech model
- elevenlabs_model_id: ElevenLabs voice model
#### Audio Parameters:
- wav_sample_rate: Audio sample rate (default: 16000)
- silence_threshold: Silence detection threshold
- silence_duration_sec: Silence duration for input end
- porcupine_sensitivity: Wake word detection sensitivity
# Usage
Start the voice assistant:
```bash
python main.py
```

1. Say the wake word "Hey Assistant" to activate
2. Speak your query when prompted
3. Wait for the assistant's response


# Security Notice
The `config.json` file contains sensitive API keys and is excluded from version control. Never commit this file to the repository.
# Project Structure

- main.py: Core application logic
- config.example.json: Template configuration file
- config.json: Your private configuration (git-ignored)
- tts_cache/: Cached voice responses (git-ignored)

# Notes
Installing the drivers is a nightmare. Using a script made it work for me after a few hours of troubleshooting: https://github.com/dihan/pi-voice-assistant

Not tested, but suggested by many users:
This one contains different kernel versions as branches: https://github.com/HinTak/seeed-voicecard/
The main problem is there are no new releases of the drivers for the latest kernel versions, so you have to use an older kernel version. You have to use a fork.