FROM pytorch/pytorch
USER root
WORKDIR /sample
RUN pip3 install --upgrade transformers accelerate huggingface_hub torch torchvision torchaudio argparse
COPY . .
