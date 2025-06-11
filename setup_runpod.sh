#!/bin/bash

# Создаём директорию ~/workspace/ssh, если она не существует
mkdir -p ~/workspace/ssh

# Проверяем, существуют ли ключи в ~/workspace/ssh
if [ ! -f /workspace/.ssh/gitlab_key ] || [ ! -f /workspace/.ssh/gitlab_key.pub ]; then
    echo "Ключи не найдены в ~/workspace/ssh, генерируем новые..."
    ssh-keygen -t ed25519 -C "demo@demo.com" -f ~/workspace/ssh/gitlab_key -N ""
else
    echo "Ключи уже существуют в ~/workspace/ssh, пропускаем генерацию."
fi

# Создаём директорию ~/.ssh, если она не существует
mkdir -p ~/.ssh

# Создаём конфигурационный файл ~/.ssh/config
cat > ~/.ssh/config << EOF
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/gitlab_key
    IdentitiesOnly yes
EOF

# Устанавливаем права доступа для конфигурационного файла
chmod 600 ~/.ssh/config

# Устанавливаем права доступа для ключей в ~/workspace/ssh
chmod 600 /workspace/.ssh/gitlab_key
chmod 644 /workspace/.ssh/gitlab_key.pub

# Copy keys to ~/.ssh directory
cp /workspace/.ssh/gitlab_key ~/.ssh/
cp /workspace/.ssh/gitlab_key.pub ~/.ssh/
chmod 600 ~/.ssh/gitlab_key
chmod 644 ~/.ssh/gitlab_key.pub


# Display current Python version
echo "Python version:"
python --version

pip install --ignore-installed flask

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

pip uninstall numpy -y
pip install numpy==1.26.4

# Verify numpy is installed
echo "Verifying numpy installation:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Verify pytorch is installed
echo "Verifying PyTorch installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete. Start api_server.py"

MODEL_KEY="codellama" python api_server.py
