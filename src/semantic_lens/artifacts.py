"""Stage signatures, atomic checkpoints and content-addressed completion manifests."""
import json
from pathlib import Path
from .data import file_hash, write_json


def implementation_hash():
    from .data import digest
    root = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).parent.glob('*.py')) + [root/name for name in (
        'src/execsem/sandbox.py', 'scripts/270_semantic_lens.py',
        'src/models/loader.py', 'src/workspace_lens/adapter.py',
        'third_party/jacobian-lens/jlens/hooks.py')]
    return digest({str(p.relative_to(root)): file_hash(p) for p in files})


def verify(path, memo=None):
    path = Path(path)
    memo = {} if memo is None else memo
    if path in memo:
        return memo[path]
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['signature']['implementation_hash'] != implementation_hash():
        raise ValueError('Implementation changed; use a fresh run directory')
    for name, expected in manifest['signature']['upstream'].items():
        if verify(path.parent/name, memo) != expected:
            raise ValueError(f'Upstream stage changed: {name}')
    for name, expected in manifest['files'].items():
        if file_hash(path/name) != expected:
            raise ValueError(f'Artifact changed: {path/name}')
    memo[path] = file_hash(path/'manifest.json')
    return memo[path]


def start(root, name, config, upstream=()):
    root = Path(root)
    memo = {}
    signature = dict(config=config, implementation_hash=implementation_hash(),
                     upstream={stage: verify(root/stage, memo) for stage in upstream})
    path = root/name
    path.mkdir(parents=True, exist_ok=True)
    signature_path = path/'signature.json'
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError(f'{name} configuration changed; use a new run directory')
    else:
        if list(path.iterdir()):
            raise ValueError(f'Unrecognized partial stage: {path}')
        write_json(signature_path, signature)
    if (path/'manifest.json').exists():
        verify(path)
        return path, True
    return path, False


def finish(path):
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*'))
             if p.is_file() and p.name != 'manifest.json' and not p.name.endswith('.tmp')}
    write_json(path/'manifest.json', dict(signature=json.loads((path/'signature.json').read_text()), files=files))


def save_checkpoint(path, value):
    write_json(path, value)
    write_json(path.with_suffix('.sha256'), {'sha256': file_hash(path)})


def load_checkpoint(path):
    if file_hash(path) != json.loads(path.with_suffix('.sha256').read_text())['sha256']:
        raise ValueError(f'Checkpoint changed: {path}')
    return json.loads(path.read_text())
