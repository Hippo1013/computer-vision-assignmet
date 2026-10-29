$ErrorActionPreference = 'Stop'
$python = 'D:\Anaconda\envs\letter_recognition_train\python.exe'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$out = Join-Path $PSScriptRoot 'outputs\stage1_20261029'
New-Item -ItemType Directory -Force $out | Out-Null
& $python -m pip freeze | Out-File -Encoding utf8 (Join-Path $out 'environment.txt')
& $python -u (Join-Path $PSScriptRoot 'train_stage1.py') 1>> (Join-Path $out 'runner.log') 2>> (Join-Path $out 'runner-error.log')
if ($LASTEXITCODE -ne 0) {
    @{state='runner_failed';exit_code=$LASTEXITCODE;time=(Get-Date -Format o)} | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $out 'runner_status.json')
    exit $LASTEXITCODE
}
& $python (Join-Path $PSScriptRoot 'summarize_stage1.py') $out 1>> (Join-Path $out 'runner.log') 2>> (Join-Path $out 'runner-error.log')
@{state= $(if($LASTEXITCODE -eq 0){'complete'}else{'summary_failed'});exit_code=$LASTEXITCODE;time=(Get-Date -Format o)} | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $out 'runner_status.json')
exit $LASTEXITCODE
