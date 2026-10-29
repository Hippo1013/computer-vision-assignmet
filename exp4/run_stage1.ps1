$ErrorActionPreference = 'Stop'
$python = 'D:\Anaconda\envs\letter_recognition_train\python.exe'
& $python (Join-Path $PSScriptRoot 'run_stage1.py')
exit $LASTEXITCODE
