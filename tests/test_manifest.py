import os
import hashlib
import pytest
from scripts.stats_utils import load_checkpoint_manifest, verify_checkpoint


def test_manifest_exists_and_verifies():
    """Verify that MANIFEST.sha256 exists, contains all checkpoints, and hashes match."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models")
    manifest_path = os.path.join(models_dir, "MANIFEST.sha256")

    if not os.path.exists(models_dir) or not os.path.exists(manifest_path):
        pytest.skip("Models directory or MANIFEST.sha256 not present in environment.")

    manifest = load_checkpoint_manifest(manifest_path)
    assert len(manifest) == 40, f"Expected 40 checkpoint entries, found {len(manifest)}"

    # Check that all checkpoints exist and match their expected hash
    for rel_path, expected_hash in manifest.items():
        full_path = os.path.join(models_dir, rel_path)
        assert os.path.exists(full_path), f"Checkpoint missing on disk: {full_path}"
        with open(full_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest().lower()
        assert actual_hash == expected_hash.lower(), (
            f"Hash mismatch for {rel_path}!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )
        assert verify_checkpoint(full_path, models_dir=models_dir, manifest=manifest) is True


def test_manifest_mismatch_raises_error():
    """Verify that tampering with hash triggers a ValueError in verify_checkpoint."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models")
    manifest_path = os.path.join(models_dir, "MANIFEST.sha256")

    if not os.path.exists(models_dir) or not os.path.exists(manifest_path):
        pytest.skip("Models directory or MANIFEST.sha256 not present in environment.")

    manifest = load_checkpoint_manifest(manifest_path)
    first_key = next(iter(manifest.keys()))
    corrupt_manifest = manifest.copy()
    corrupt_manifest[first_key] = "0" * 64

    test_file = os.path.join(models_dir, first_key)
    with pytest.raises(ValueError, match="SHA-256 hash mismatch"):
        verify_checkpoint(test_file, models_dir=models_dir, manifest=corrupt_manifest)
