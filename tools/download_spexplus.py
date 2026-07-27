"""下载并校验与本项目 SpEx+ 实现匹配的 16 kHz 公开检查点。"""

import argparse
import hashlib
import json
from pathlib import Path


MODEL_FILE_ID = "10uiHjhrpzlU9WsWsfGHVdwpauXhVTX1W"
CONFIG_FILE_ID = "1h3CJnE7A0PbWeoE0a3EezhVgv97nlPG-"
EXPECTED_SHA256 = (
    "2d6a2f2b404fd18a809eb82052fd64ef0bd986f410b1043bc666b54121e44b5c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "pretrained"
            / "spex_plus"
        ),
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "缺少 gdown，请先运行: python -m pip install gdown==5.2.0"
        ) from exc

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "checkpoint.pth"
    config_path = output_dir / "upstream_config.json"

    if not model_path.is_file() or sha256(model_path) != EXPECTED_SHA256:
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
            str(model_path),
            quiet=False,
        )
    actual = sha256(model_path)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"权重 SHA256 校验失败: 期望 {EXPECTED_SHA256}, 实际 {actual}"
        )

    if not config_path.is_file():
        gdown.download(
            f"https://drive.google.com/uc?id={CONFIG_FILE_ID}",
            str(config_path),
            quiet=False,
        )

    manifest = {
        "model": "SpEx+",
        "sample_rate": 16000,
        "checkpoint": model_path.name,
        "sha256": actual,
        "source_repository": (
            "https://github.com/ex7remum/DLA_Speaker_Separation"
        ),
        "source_checkpoint_file_id": MODEL_FILE_ID,
        "license": "MIT",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"SpEx+ 权重就绪: {model_path}")
    print(f"SHA256: {actual}")


if __name__ == "__main__":
    main()
