sudo apt update
sudo apt install graphviz vim openssh-client tmux git

# LLM + SD + Huggingface + simulacra-aesthetics prereqs
pip install transformers torch torchvision torchaudio accelerate diffusers ftfy huggingface_hub pandas numpy pillow graphviz pyarrow fastparquet==2023.1.0 tqdm scikit-learn

pip install git+https://github.com/openai/CLIP.git
git clone https://github.com/crowsonkb/simulacra-aesthetic-models.git

cd simulacra-aesthetic-models

# database of initial SD prompts
wget "https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts/resolve/main/data/train.parquet"

# main script
wget "https://raw.githubusercontent.com/NelsonAU/sd-lmx/main/evolve_aesthetics.py"

huggingface-cli login
export HF_HOME=/scratch
