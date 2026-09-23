"""Read TFT-related GitHub Actions artifacts without persisting ZIPs.

Authentication is obtained from the user's existing GitHub CLI session. Tokens
are kept in process memory and are never logged or written to disk.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any, Iterator

import pandas as pd
import requests

API_ROOT = "https://api.github.com"
TFT_BASENAMES = {
    "tft_prediction_level.csv",
    "tft_regime_analysis.csv",
    "tft_regime_tree_rules.txt",
    "tft_regime_tree_importance.csv",
}


class GitHubArtifactError(RuntimeError):
    pass


def detect_repository() -> str:
    """Detect owner/repository from the current checkout's GitHub remote."""
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], check=True,
        capture_output=True, text=True,
    )
    remote = result.stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", remote)
    if not match:
        raise GitHubArtifactError(f"origin no parece un remoto de GitHub: {remote}")
    return match.group(1)


def github_token() -> str:
    """Read a token from gh's authenticated session, without displaying it."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise GitHubArtifactError(
            "No se pudo obtener autenticación desde GitHub CLI. Ejecuta `gh auth login`."
        ) from exc
    token = result.stdout.strip()
    if not token:
        raise GitHubArtifactError("GitHub CLI no devolvió un token autenticado.")
    return token


def _session(token: str | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token or github_token()}",
    })
    return session


def list_repository_artifacts(
    repository: str | None = None, *, include_expired: bool = False,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch every Actions artifact page (the API allows up to 100 per page)."""
    repository = repository or detect_repository()
    session = _session(token)
    artifacts: list[dict[str, Any]] = []
    page = 1
    while True:
        response = session.get(
            f"{API_ROOT}/repos/{repository}/actions/artifacts",
            params={"per_page": 100, "page": page}, timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("artifacts", [])
        artifacts.extend(batch)
        if len(batch) < 100 or len(artifacts) >= payload.get("total_count", 0):
            break
        page += 1
    return [a for a in artifacts if include_expired or not a.get("expired", False)]


def artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    run = artifact.get("workflow_run") or {}
    return {
        "artifact_id": artifact.get("id"),
        "artifact_name": artifact.get("name"),
        "workflow_run_id": run.get("id"),
        "created_at": artifact.get("created_at"),
        "updated_at": artifact.get("updated_at"),
        "expires_at": artifact.get("expires_at"),
        "artifact_size_bytes": artifact.get("size_in_bytes"),
    }


def download_artifact_zip(
    artifact: dict[str, Any], *, session: requests.Session | None = None,
) -> io.BytesIO:
    """Download one artifact ZIP into memory; callers should release it promptly."""
    session = session or _session()
    response = session.get(artifact["archive_download_url"], timeout=(30, 180), stream=True)
    response.raise_for_status()
    buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            buffer.write(chunk)
    buffer.seek(0)
    return buffer


def inspect_zip(zip_buffer: io.BytesIO) -> list[str]:
    with zipfile.ZipFile(zip_buffer) as archive:
        return [name for name in archive.namelist() if not name.endswith("/")]


def _is_tft_related(name: str, filenames: list[str]) -> bool:
    basenames = {PurePosixPath(item).name.lower() for item in filenames}
    return bool(TFT_BASENAMES & basenames) or (
        "tft" in name.lower() and any(".csv" in item.lower() for item in filenames)
    )


def read_artifact_contents(
    zip_buffer: io.BytesIO, metadata: dict[str, Any],
) -> dict[str, Any]:
    """Read CSVs and text from a ZIP using member paths, without extraction."""
    result: dict[str, Any] = {"metadata": metadata, "files": {}, "dataframes": {}}
    with zipfile.ZipFile(zip_buffer) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            basename = PurePosixPath(member).name.lower()
            if basename.endswith(".csv"):
                try:
                    frame = pd.read_csv(archive.open(member))
                    for key, value in metadata.items():
                        frame[key] = value
                    result["dataframes"][member] = frame
                except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                    result["files"][member] = f"CSV no legible: {exc}"
            elif basename.endswith((".txt", ".log", ".json", ".yaml", ".yml")):
                try:
                    result["files"][member] = archive.read(member).decode("utf-8", errors="replace")
                except (KeyError, OSError) as exc:
                    result["files"][member] = f"No se pudo leer: {exc}"
            else:
                result["files"][member] = {"size": archive.getinfo(member).file_size}
    return result


def fetch_tft_artifacts(
    repository: str | None = None, *, days: int = 30,
    include_all_available: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Find and read recent TFT artifacts; ZIPs are processed one at a time.

    GitHub retains artifacts for 30 days in this repository. By default the
    requested date window is applied, while include_all_available disables it.
    """
    repository = repository or detect_repository()
    token = github_token()
    session = _session(token)
    artifacts = list_repository_artifacts(repository, token=token)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    selected = []
    for item in artifacts:
        created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
        if include_all_available or pd.isna(created) or created.to_pydatetime() >= cutoff:
            selected.append(item)

    loaded: list[dict[str, Any]] = []
    for artifact in selected:
        meta = artifact_metadata(artifact)
        zipped = download_artifact_zip(artifact, session=session)
        try:
            members = inspect_zip(zipped)
            if not _is_tft_related(str(artifact.get("name", "")), members):
                continue
            record = read_artifact_contents(zipped, meta)
            record["members"] = members
            loaded.append(record)
        except zipfile.BadZipFile:
            print(f"Warning: artifact {meta['artifact_name']} is not a readable ZIP")
        finally:
            zipped.close()
    session.close()
    return repository, loaded


def dataframe_for_basename(record: dict[str, Any], basename: str) -> pd.DataFrame | None:
    for path, frame in record.get("dataframes", {}).items():
        if PurePosixPath(path).name.lower() == basename.lower():
            return frame.copy()
    return None


def text_for_basename(record: dict[str, Any], basename: str) -> str | None:
    for path, value in record.get("files", {}).items():
        if PurePosixPath(path).name.lower() == basename.lower() and isinstance(value, str):
            return value
    return None
