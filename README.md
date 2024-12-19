# Voice to funcion calling

## Howto use

 - subscribe to https://huggingface.co/openai/whisper-small
 - subscribe to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
 - create an access token on https://huggingface.co/

```
docker build -t voice -f Dockerfile .
docker run --runtime nvidia --gpus all --rm -v <local folder for huggingface>:/root/.cache/huggingface --env "HF_TOKEN=<token for https://huggingface.co/>" -it --entrypoint=bash --ipc=host voice

if no nvidia gpu is available => remove '--runtime nvidia --gpus all'

then run
python3 voice2text.py --file train_1_departs_de.mp3

expected output
Der Zug 1 fährt ab => transcribed input from train_1_departs_de.mp3
The train 1 is departing => english translation
{"name": "move_train", "parameters": {"train": "1", "action": "start"}}<|eom_id|> => command to call

```

## How it works

mp3 file -> whisper -> transcribed text -> translate with llama3.2 -> english text -> function calling with llama3.2 -> json with command and arguments
