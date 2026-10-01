"""Restore a lossless, per-file GitHub-sized SQLite backup on a fresh checkout."""
import gzip
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3


def restore_sqlite_backup(path):
    path=Path(path)
    if path.exists():
        return False
    manifest_path=path.with_name(path.name+'.restore.json')
    if not manifest_path.exists():
        return False
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    temporary=path.with_name(path.name+'.restore.tmp')
    digest=hashlib.sha256()
    try:
        with temporary.open('wb') as output:
            for item in manifest['parts']:
                name=item['name']
                if Path(name).name!=name:
                    raise ValueError('SQLite backup part must remain in its own directory')
                with gzip.open(path.parent/name,'rb') as source:
                    while chunk:=source.read(1024*1024):
                        digest.update(chunk);output.write(chunk)
            output.flush();os.fsync(output.fileno())
        if digest.hexdigest()!=manifest['sha256'] or temporary.stat().st_size!=manifest['bytes']:
            raise ValueError('SQLite backup checksum or size mismatch; original parts preserved')
        with closing(sqlite3.connect(temporary.as_uri()+'?mode=ro',uri=True)) as db:
            if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise ValueError('Restored SQLite integrity check failed')
        os.replace(temporary,path)
        return True
    finally:
        temporary.unlink(missing_ok=True)
