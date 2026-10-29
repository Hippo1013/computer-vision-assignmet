"""Fetch experiment records from Windows over SSH, including Unicode paths safely."""
import argparse
import base64
import io
import subprocess
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--weights',action='store_true',help='also transfer model checkpoints (large)')
    args=p.parse_args()
    remote = r'''
import base64,io,zipfile
from pathlib import Path
root=Path(r'E:\机器视觉\homework\exp4\outputs\stage1_20261029_train_test')
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for f in root.rglob('*'):
        if f.is_file() and f.suffix in EXTENSIONS:
            z.write(f,f.relative_to(root).as_posix())
print(base64.b64encode(buffer.getvalue()).decode('ascii'))
'''.replace('EXTENSIONS',repr(['.json','.png','.md','.txt','.log']+(['.pt'] if args.weights else [])))
    pycode="import base64;exec(base64.b64decode('"+base64.b64encode(remote.encode()).decode()+"'))"
    ps="& 'D:\\Anaconda\\envs\\letter_recognition_train\\python.exe' -c \""+pycode+"\""
    cmd='powershell -NoProfile -EncodedCommand '+base64.b64encode(ps.encode('utf-16le')).decode()
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','windows',cmd],capture_output=True,check=True)
    data=base64.b64decode(result.stdout.strip(),validate=True)
    out=Path(__file__).resolve().parent/'outputs/stage1_20261029_train_test'
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe archive member')
        z.extractall(out)
        print(f'Fetched {len(z.namelist())} files to {out}')


if __name__=='__main__':main()
