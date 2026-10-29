"""Detached Windows runner: stage four completes, summarizes, and waits for user."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    base=Path(__file__).resolve().parent
    out=base/'outputs/stage45_20261029';out.mkdir(parents=True,exist_ok=True)
    (out/'environment.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    with open(out/'runner.log','ab',buffering=0) as log,open(out/'runner-error.log','ab',buffering=0) as error:
        code=subprocess.call([sys.executable,'-u',str(base/'train_stage45.py')],cwd=base.parent,stdout=log,stderr=error)
        if code==0:
            code=subprocess.call([sys.executable,str(base/'summarize_stage45.py')],cwd=base.parent,stdout=log,stderr=error)
            state='stage4_complete_awaiting_user' if code==0 else 'summary_failed'
        else:state='paused' if code==75 else 'runner_failed'
    (out/'runner_status.json').write_text(json.dumps({'state':state,'exit_code':code,'time':datetime.now().astimezone().isoformat()},indent=2),encoding='utf-8')
    return code


if __name__=='__main__':sys.exit(main())
