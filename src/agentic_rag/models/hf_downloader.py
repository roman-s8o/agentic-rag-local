from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ModelKind = Literal["llm", "vlm", "embedding", "reranker", "image_embedding"]


class ModelSpec(BaseModel):
    role: str
    kind: ModelKind
    repo_id: str
    revision: str | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    enabled: bool = True
    mlx_convert: bool = False
    mlx_q_bits: int | None = 4


class ModelRegistry(BaseModel):
    models: list[ModelSpec] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> ModelRegistry:
        if not path.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})

    def get(self, key: str) -> ModelSpec | None:
        return next((m for m in self.models if key in (m.role, m.repo_id)), None)

    def enabled(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]


@dataclass(frozen=True)
class RepoInfo:
    repo_id: str
    sha: str | None
    gated: bool
    license: str | None
    pipeline_tag: str | None


@dataclass(frozen=True)
class DownloadResult:
    repo_id: str
    revision: str | None
    path: Path
    info: RepoInfo | None


@dataclass(frozen=True)
class LocalModel:
    repo_id: str
    size_bytes: int
    revisions: tuple[str, ...]


class ModelAccessError(RuntimeError):
    pass


class HFModelDownloader:
    def __init__(
        self,
        cache_dir: Path,
        token: str | None = None,
        api: HfApi | None = None,
        download_fn: Callable[..., Any] = snapshot_download,
    ) -> None:
        self.cache_dir = cache_dir
        self.token = token or None
        self.api = api or HfApi(token=self.token)
        self._download = download_fn

    def inspect(self, repo_id: str, revision: str | None = None) -> RepoInfo:
        try:
            info = self.api.model_info(repo_id, revision=revision)
        except RepositoryNotFoundError as exc:
            raise ModelAccessError(
                f"Repository '{repo_id}' not found or private. Check the id or set HF_TOKEN."
            ) from exc
        card = info.card_data
        license_ = card.get("license") if card is not None else None
        return RepoInfo(
            repo_id=repo_id,
            sha=info.sha,
            gated=bool(info.gated),
            license=str(license_) if license_ else None,
            pipeline_tag=info.pipeline_tag,
        )

    def pull(
        self,
        repo_id: str,
        revision: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        local_dir: Path | None = None,
        check: bool = True,
    ) -> DownloadResult:
        info = self.inspect(repo_id, revision) if check else None
        if info and info.gated and not self.token:
            raise ModelAccessError(
                f"'{repo_id}' is gated. Accept its license on huggingface.co and set HF_TOKEN."
            )
        if info and info.license:
            logger.info("%s is licensed under '%s'", repo_id, info.license)
        try:
            path = self._download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=None if local_dir else self.cache_dir,
                local_dir=local_dir,
                allow_patterns=include,
                ignore_patterns=exclude,
                token=self.token,
            )
        except GatedRepoError as exc:
            raise ModelAccessError(
                f"Access to '{repo_id}' denied. Accept its license on huggingface.co."
            ) from exc
        return DownloadResult(repo_id=repo_id, revision=revision, path=Path(path), info=info)

    def pull_spec(self, spec: ModelSpec) -> DownloadResult:
        result = self.pull(spec.repo_id, spec.revision, spec.include, spec.exclude)
        if spec.mlx_convert:
            self.convert_to_mlx(spec.repo_id, spec.kind, q_bits=spec.mlx_q_bits)
        return result

    def mlx_output_dir(self, repo_id: str, q_bits: int | None) -> Path:
        suffix = f"-{q_bits}bit" if q_bits else ""
        return self.cache_dir / "mlx" / f"{repo_id.replace('/', '__')}{suffix}"

    def convert_to_mlx(self, repo_id: str, kind: ModelKind, q_bits: int | None = 4) -> Path:
        if sys.platform != "darwin":
            raise ModelAccessError("MLX conversion is only supported on Apple Silicon macOS.")
        package = "mlx_vlm" if kind == "vlm" else "mlx_lm"
        out = self.mlx_output_dir(repo_id, q_bits)
        cmd = [sys.executable, "-m", package, "convert", "--hf-path", repo_id]
        cmd += ["--mlx-path", str(out)]
        if q_bits:
            cmd += ["-q", "--q-bits", str(q_bits)]
        logger.info("Running %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return out

    def list_local(self) -> list[LocalModel]:
        if not self.cache_dir.exists():
            return []
        cache = scan_cache_dir(self.cache_dir)
        return sorted(
            (
                LocalModel(
                    repo_id=repo.repo_id,
                    size_bytes=repo.size_on_disk,
                    revisions=tuple(r.commit_hash[:8] for r in repo.revisions),
                )
                for repo in cache.repos
                if repo.repo_type == "model"
            ),
            key=lambda m: m.repo_id,
        )
