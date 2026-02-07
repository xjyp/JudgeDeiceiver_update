from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    local_dir="./meta-llama",
    local_dir_use_symlinks=False,   # 禁止软链接
    resume_download=True,           # 断点续传
)
