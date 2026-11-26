"""Restore archived simulated experiment inputs next to their original scripts."""
from pathlib import Path
from zipfile import ZipFile

root = Path(__file__).resolve().parent
for name in ('double_integrator_data.zip', 'reacher_data.zip'):
    with ZipFile(root / name) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or member.is_dir():
                raise ValueError(f'Unexpected archive member: {member.filename}')
        archive.extractall(root)
    print(f'Restored {name}')
