# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Set

from fluxvla.engines.utils import TOKENIZERS

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


@TOKENIZERS.register_module()
class PretrainedTokenizer:

    def __init__(
        self,
        model_path: str,
        model_max_length: int = 2048,
        padding_side: str = 'right',
    ) -> None:
        """Load tokenizer from a path or repo id and record the original argument."""  # noqa: E501
        # Avoid top-level import to reduce environment constraints
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=model_max_length,
            padding_side=padding_side,
        )
        self.model_path = model_path  # Keep original argument for source resolution  # noqa: E501
        self.copy_attrs_from_obj()

        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __call__(self, *args, **kwds):
        return self.tokenizer(*args, **kwds)

    def copy_attrs_from_obj(self):
        for key, value in vars(self.tokenizer).items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.tokenizer)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    # ---------------- New implementation begins ----------------

    def _resolve_source_dir(self,
                            explicit_source: Optional[str] = None
                            ) -> Optional[Path]:
        """
        Best-effort resolution of a local source directory that contains tokenizer files:  # noqa: E501
        1) Use the explicitly provided source directory if it exists.
        2) Check tokenizer.name_or_path and the recorded model_path if they are local directories.  # noqa: E501
        3) If the value looks like a remote repo id and huggingface_hub is available,  # noqa: E501
           use snapshot_download to fetch only tokenizer-related files.
        4) Return None if no source can be resolved (caller will fall back to save_pretrained).  # noqa: E501
        """
        # 1) Explicitly provided local directory
        if explicit_source and os.path.isdir(explicit_source):
            return Path(explicit_source)

        # 2) Directly usable local directories from common candidates
        candidates: List[str] = []
        # from_pretrained's name_or_path (may be a local path or a repo id)
        if hasattr(self.tokenizer,
                   'name_or_path') and self.tokenizer.name_or_path:
            candidates.append(self.tokenizer.name_or_path)
        # The recorded init argument
        if getattr(self, 'model_path', None):
            candidates.append(self.model_path)

        for cand in candidates:
            if cand and os.path.isdir(cand):
                return Path(cand)

        # 3) Looks like a repo id -> try snapshot_download
        repo_id = None
        for cand in candidates:
            # Heuristic: contains '/', is not an existing directory, and is not an absolute path  # noqa: E501
            if cand and not os.path.isdir(
                    cand) and '/' in cand and not os.path.isabs(cand):
                repo_id = cand
                break

        if repo_id and snapshot_download is not None:
            # Download only tokenizer-related artifacts to minimize size
            allow_patterns = [
                'tokenizer.*',
                'special_tokens_map.json',
                'tokenizer_config.json',
                'added_tokens.json',
                'vocab.*',
                'merges.txt',
                'spiece.model',
                'sentencepiece.*',
                '*.bpe',
            ]
            snapshot_path = snapshot_download(
                repo_id=repo_id, allow_patterns=allow_patterns)
            # Use the returned snapshot directory as the source
            return Path(snapshot_path)

        # 4) Give up; let caller use a fallback
        return None

    def _collect_tokenizer_files(self, src_dir: Path) -> List[Path]:
        """
        Collect tokenizer-related files from a source directory.

        Inclusion rules:
        - Known names from tokenizer.vocab_files_names and common files like:
          tokenizer.json, tokenizer_config.json, special_tokens_map.json,
          added_tokens.json, merges.txt, vocab.json|txt|bpe, spiece/sentencepiece models.  # noqa: E501
        - Files whose names contain tokenizer/vocab/merge/spiece/sentencepiece/.bpe.  # noqa: E501

        Exclusion rules:
        - Explicitly exclude model weights and unrelated large artifacts
          (*.safetensors, *.bin, TF/Flax/ONNX, etc.).
        """
        # Expected filenames (from tokenizer config if available) + common defaults  # noqa: E501
        expected: Set[str] = set()
        try:
            if hasattr(
                    self.tokenizer,
                    'vocab_files_names') and self.tokenizer.vocab_files_names:
                expected |= set(self.tokenizer.vocab_files_names.values())
        except Exception:
            pass
        expected |= {
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'added_tokens.json',
            'merges.txt',
            'vocab.json',
            'vocab.txt',
            'vocab.bpe',
            'spiece.model',
            'sentencepiece.model',
        }

        # Generic keyword-based inclusion
        include_keywords = ('tokenizer', 'vocab', 'merge', 'merges', 'spiece',
                            'sentencepiece', '.bpe')

        # Explicit exclusion (weights and training/runtime artifacts)
        exclude_regex = re.compile(
            r'(pytorch_model|model\.(safetensors|bin)|tf_model|flax_model|onnx|tflite|gguf|ggml|\.ot|optimizer|scheduler|trainer)'  # noqa: E501
        )

        files: List[Path] = []
        for p in src_dir.rglob('*'):
            if not p.is_file():
                continue
            name_lower = p.name.lower()

            if exclude_regex.search(name_lower):
                continue
            if p.name in expected:
                files.append(p)
                continue
            if any(k in name_lower for k in include_keywords):
                files.append(p)

        # De-duplicate while preserving order
        seen = set()
        uniq = []
        for f in files:
            if f.as_posix() not in seen:
                uniq.append(f)
                seen.add(f.as_posix())
        return uniq

    def save_pretrained(self,
                        save_directory: str,
                        source_dir: Optional[str] = None,
                        overwrite: bool = True):
        """
        Copy tokenizer-related files from a source directory into save_directory.  # noqa: E501
        If the source directory cannot be resolved, fall back to
        self.tokenizer.save_pretrained into a temporary directory and copy the
        essential JSON files to complete metadata.
        """
        dst = Path(save_directory)
        dst.mkdir(parents=True, exist_ok=True)

        # First attempt: direct copy from a resolved source directory
        src = self._resolve_source_dir(source_dir)
        copied = []

        if src is not None:
            tok_files = self._collect_tokenizer_files(src)
            for f in tok_files:
                # Most repos keep tokenizer files in the root; flatten copy is typically fine  # noqa: E501
                rel = f.name
                target = dst / rel
                if target.exists() and not overwrite:
                    # Skip when not allowed to overwrite
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied.append(target.name)

        # Fallback: if essential JSONs are missing, generate them via save_pretrained  # noqa: E501
        must_have = ['tokenizer_config.json', 'special_tokens_map.json']
        missing = [m for m in must_have if not (dst / m).exists()]
        if missing:
            with tempfile.TemporaryDirectory(prefix='hf_tok_fallback_') as tmp:
                self.tokenizer.save_pretrained(tmp)
                for m in missing:
                    src_file = Path(tmp) / m
                    if src_file.exists():
                        shutil.copy2(src_file, dst / m)
                        copied.append(m)

        # Final safety: ensure a core vocab asset exists; if not, export everything  # noqa: E501
        if not any((dst / name).exists() for name in [
                'tokenizer.json', 'spiece.model', 'vocab.json', 'vocab.txt',
                'vocab.bpe'
        ]):
            # Neither source copy nor fallback produced a core vocabulary file;
            # export the full tokenizer to the target directory.
            self.tokenizer.save_pretrained(dst.as_posix())

        print(f'Tokenizers saved to: {dst.resolve()}')
        # Quick summary of what is present for verification
        present = sorted(p.name for p in dst.iterdir() if p.is_file())
        print('Files in target dir:', present)
        return save_directory

    # ---------------- New implementation ends ----------------
