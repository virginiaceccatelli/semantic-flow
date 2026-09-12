"""Container-only execution. Never falls back to running submissions on the host."""
from __future__ import annotations
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

# This wrapper is trusted; it starts the dataset program only INSIDE a container.
RUNNER = r'''
import json, os, resource, signal, subprocess, sys
seconds = float(sys.argv[1])
def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(seconds)+1), max(1, int(seconds)+1)))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with open('/case/input.txt','rb') as inp, open('/tmp/stdout','wb') as out, open('/tmp/stderr','wb') as err:
    p = subprocess.Popen([sys.executable, '-I', '-B', '/case/program.py'], stdin=inp, stdout=out, stderr=err, start_new_session=True, preexec_fn=limits)
    try:
        rc = p.wait(timeout=seconds)
        status = 'ok' if rc == 0 else 'runtime_error'
    except subprocess.TimeoutExpired:
        status = 'timeout'; rc = None
    finally:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        p.wait()
out = open('/tmp/stdout','rb').read(4097)
err = open('/tmp/stderr','rb').read(1024)
if len(out)>4096: status='output_limit'
print(json.dumps(dict(status=status,returncode=rc,stdout=out[:4096].decode('utf-8',errors='replace'),stderr=err.decode('utf-8',errors='replace'))))
'''


def resolve_image(runtime, image):
    if runtime not in ['docker', 'podman', 'singularity'] or not shutil.which(runtime):
        raise RuntimeError(f'{runtime} unavailable. Use an approved Docker/Podman/Singularity execution host; no host-execution fallback.')
    if runtime == 'singularity':
        path = Path(image).expanduser().resolve()
        if not path.is_file() or path.suffix != '.sif':
            raise RuntimeError('Singularity requires a local .sif image. Pull python:3.11-slim first and pass --image /path/python311.sif')
        return str(path)
    result = subprocess.run([runtime, 'image', 'inspect', '--format', '{{.Id}}', image], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'Cannot inspect container image. Pull it first: {runtime} pull {image}\n{result.stderr[:1000]}')
    resolved = result.stdout.strip()
    if not resolved.startswith('sha256:'): raise RuntimeError('Could not resolve immutable image ID')
    return resolved


def container_command(runtime, image, directory, name, seconds):
    if runtime == 'singularity':
        return [runtime, 'exec', '--no-oci', '--containall', '--cleanenv', '--no-eval',
                '--no-home', '--no-mount', 'home,cwd,hostfs,bind-paths,sys',
                '--net', '--network=none', '--drop-caps=ALL',
                '--memory=512m', '--memory-swap=512m', '--pids-limit=32', '--cpus=1',
                '--bind', f'{directory}:/case:ro', '--pwd', '/tmp',
                image, 'python', '-I', '-B', '-c', RUNNER, str(seconds)]
    return [runtime, 'run', '--rm', '--pull=never', '--name', name, '--network=none',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user=65534:65534', '--pids-limit=32', '--memory=512m', '--memory-swap=512m',
            '--cpus=1', '--ulimit', 'nofile=64:64', '--ulimit', 'core=0:0',
            '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m,mode=1777',
            '--mount', f'type=bind,source={directory},target=/case,readonly',
            '--workdir=/tmp', '--log-driver=none', '--entrypoint=python',
            image, '-I', '-B', '-c', RUNNER, str(seconds)]


def run_case(runtime, image, code, input_text, seconds=3):
    name = 'execsem-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='execsem-case-') as tmp:
        root = Path(tmp); case = root/'case'; case.mkdir(mode=0o755)
        for filename, text in [('program.py', code), ('input.txt', input_text)]:
            path = case/filename; path.write_text(text); path.chmod(0o444)
        command = container_command(runtime, image, str(case.resolve()), name, seconds)
        # File-backed logs bound host RAM even if an untrusted child writes to wrapper stdout.
        with (root/'stdout').open('wb') as out, (root/'stderr').open('wb') as err:
            # Discard runtime environment overrides (extra binds, overlays, GPU, etc.).
            # Preserve HOME for runtime setup only; it is not mounted into the payload.
            kwargs = {}
            if runtime == 'singularity':
                kwargs = dict(cwd=str(root), start_new_session=True,
                              env={k: os.environ[k] for k in ('PATH', 'HOME', 'USER', 'LOGNAME', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS') if k in os.environ})
            proc = subprocess.Popen(command, stdout=out, stderr=err, **kwargs)
            started = time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic()-started > seconds+30:
                        raise RuntimeError('Container startup/wrapper deadline exceeded; not labeled as a program timeout')
                    if out.tell() > 1048576 or err.tell() > 1048576:
                        raise RuntimeError('Container wrapper log limit exceeded')
                    time.sleep(.1)
            finally:
                if proc.poll() is None:
                    if runtime == 'singularity':
                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
                    try: proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        if runtime == 'singularity': os.killpg(proc.pid, signal.SIGKILL)
                        else: proc.kill()
                        proc.wait()
                # Also cleans up descendants on exceptions and interrupts.
                if runtime != 'singularity':
                    subprocess.run([runtime, 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        with (root/'stdout').open('rb') as f: stdout = f.read(1048576).decode(errors='replace')
        with (root/'stderr').open('rb') as f: stderr = f.read(4096).decode(errors='replace')
        if proc.returncode:
            raise RuntimeError(f'Container/wrapper failure ({proc.returncode}); no label saved: {stderr}')
        try: result = json.loads(stdout)
        except ValueError as exc: raise RuntimeError('Invalid container wrapper output; no label saved') from exc
        if result.get('status') not in ['ok', 'runtime_error', 'timeout', 'output_limit']:
            raise RuntimeError('Invalid execution status')
        return result
