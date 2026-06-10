#模型下载
from modelscope import snapshot_download

# snapshot_download('google/flan-t5-xl', cache_dir='/root/autodl-tmp/SpaMo/data3/download_model')


# # 仅下载PyTorch权重和配置文件
# model_dir = snapshot_download(
#     'google/flan-t5-xl',
#     allow_patterns=['*.bin', 'config.json', 'generation_config.json'],
#     local_dir='/path/to/your/dir'
# )

# # 仅下载PyTorch权重和配置文件
model_dir = snapshot_download(
    'google/flan-t5-xl',
    allow_patterns=['*.safetensors'],
    local_dir='/path/to/your/dir'
)

# # 排除Flax和Safetensors格式的文件
# model_dir = snapshot_download(
#     'google/flan-t5-xl',
#     ignore_patterns=['*.msgpack', '*.safetensors', '*.h5'],
#     cache_dir='/root/autodl-tmp/SpaMo/data3/download_model'
# )