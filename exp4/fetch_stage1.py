"""Fetch experiment records over SSH/SCP without buffering large model archives in RAM."""
import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

RUNS=['stage1_20261029_train_test','adam_extra_lr_20261029','stage45_20261029','stage4_batch_20261029','stage5_distillation_20261030']


def remote_python(code):
    payload=base64.b64encode(code.encode('utf-8')).decode('ascii')
    ps="& 'D:\\Anaconda\\envs\\letter_recognition_train\\python.exe' -c \"import base64;exec(base64.b64decode('"+payload+"'))\""
    cmd='powershell -NoProfile -EncodedCommand '+base64.b64encode(ps.encode('utf-16le')).decode('ascii')
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10',
                           '-o','ServerAliveCountMax=3','windows',cmd],capture_output=True,check=True).stdout


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--weights',action='store_true',help='also transfer model checkpoints (large)')
    p.add_argument('--run',choices=RUNS,default=RUNS[0])
    args=p.parse_args()
    extensions=['.json','.png','.md','.txt','.log']+(['.pt'] if args.weights else [])
    # The Windows Python owns a unique system temporary directory for the transfer archive.
    code=r'''
import hashlib,json,tempfile,zipfile
from pathlib import Path
root=Path(r'E:\机器视觉\homework\exp4\outputs')/RUN_NAME
assert root.is_dir(),root
scratch=Path(tempfile.mkdtemp(prefix='exp4-fetch-'))
try:
    archive=scratch/'records.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in EXTENSIONS:
                if f.suffix=='.pt':z.write(f,f.relative_to(root).as_posix())
                else:z.writestr(f.relative_to(root).as_posix(),f.read_bytes())
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    print(json.dumps({'scratch':str(scratch),'archive':str(archive),'sha256':h.hexdigest()}))
except BaseException:
    import shutil
    shutil.rmtree(scratch)
    raise
'''.replace('RUN_NAME',repr(args.run)).replace('EXTENSIONS',repr(extensions))
    meta=json.loads(remote_python(code).decode('utf-8').strip())
    scratch=None
    try:
        # Follow the repository's mktemp convention on the local device.
        scratch=Path(subprocess.check_output(['mktemp','-d'],text=True).strip())
        archive=scratch/'records.zip'
        remote_path=meta['archive'].replace(chr(92),'/')
        subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10',
                        '-o','ServerAliveCountMax=3','windows:'+remote_path,str(archive)],check=True)
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        assert h.hexdigest()==meta['sha256'],'Transfer archive hash mismatch'
        out=Path(__file__).resolve().parent/'outputs'/args.run
        out.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('Unsafe archive member')
            z.extractall(out)
            print(f'Fetched {len(z.namelist())} files to {out}')
    finally:
        if scratch is not None:shutil.rmtree(scratch)
        cleanup="import shutil,tempfile;from pathlib import Path;p=Path("+repr(meta['scratch'])+");assert p.parent==Path(tempfile.gettempdir()) and p.name.startswith('exp4-fetch-');shutil.rmtree(p)"
        try:remote_python(cleanup)
        except Exception as error:
            print('Remote temporary directory remains: '+meta['scratch']+' ('+str(error)+')',file=sys.stderr)


if __name__=='__main__':main()
