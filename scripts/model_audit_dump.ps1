Set-Location 'C:\Computing\Aether-eye'
$ErrorActionPreference = 'Continue'

$report = 'C:\Computing\Aether-eye\output\model_audit_full_output.txt'
New-Item -ItemType Directory -Force -Path (Split-Path $report) | Out-Null
Start-Transcript -Path $report -Force | Out-Null

function Get-RepoFiles {
    param(
        [string[]]$Include
    )

    Get-ChildItem -Recurse -File -Include $Include |
      Where-Object { $_.FullName -notmatch '\\\\.venv\\\\|\\\\node_modules\\\\|\\\\__pycache__\\\\|\\\\.git\\\\|\\\\output\\\\' }
}

Write-Host '=== MODEL ARCHITECTURE REFERENCES ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py' |
  Select-String -Pattern 'ConvNeXt|convnext|ViT|vit_|SiameseUNet|siamese|YOLO|yolo|ResNet|resnet|EfficientNet|SegFormer|UNet|unet|transformer|Transformer' |
  Select-Object Path, LineNumber, Line |
  Format-Table -AutoSize -Wrap

Write-Host '=== LOSS FUNCTIONS ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py' |
  Select-String -Pattern 'loss|Loss|criterion|Criterion|tversky|Tversky|dice|Dice|focal|Focal|BCE|CrossEntropy' |
  Where-Object { $_.Path -notmatch '__pycache__' } |
  Select-Object Path, LineNumber, Line |
  Format-Table -AutoSize -Wrap

Write-Host '=== DATASETS REFERENCED ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py','*.yaml','*.json','*.md' |
  Select-String -Pattern 'LEVIR|WHU|xView|DOTA|FGVC|Stanford|SpaceNet|Building.change|building_change|change_detection' |
  Where-Object { $_.Path -notmatch '__pycache__' } |
  Select-Object Path, LineNumber, Line |
  Format-Table -AutoSize -Wrap

Write-Host '=== ONNX MODELS LOADED AT RUNTIME ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py','*.yaml' |
  Select-String -Pattern '\.onnx|onnx_path|ort\.InferenceSession|InferenceSession' |
  Where-Object { $_.Path -notmatch '__pycache__' } |
  Select-Object Path, LineNumber, Line |
  Format-Table -AutoSize -Wrap

Write-Host '=== MODEL CONFIG FILES ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.yaml','*.json' |
  Where-Object { $_.FullName -match 'config|inference|model|classifier|detector' } |
  Select-Object FullName |
  Format-Table -AutoSize -Wrap

Write-Host '=== INFERENCE CONFIG CONTENTS ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.yaml' |
  Where-Object { $_.FullName -match 'inference|classifier|detector|change' } |
  ForEach-Object {
    Write-Host "--- $($_.FullName) ---" -ForegroundColor Yellow
    Get-Content $_.FullName
    Write-Host ''
  }

Write-Host '=== PYTORCH MODEL CLASS DEFINITIONS ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py' |
  Select-String -Pattern 'class.*\(nn\.Module\)|class.*\(.*Module\)' |
  Where-Object { $_.Path -notmatch '__pycache__' } |
  Select-Object Path, LineNumber, Line |
  Format-Table -AutoSize -Wrap

Write-Host '=== TRAINING CONFIGS FOUND ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py','*.yaml' |
  Where-Object { $_.Name -match 'train|config|cfg' } |
  Select-Object FullName |
  Format-Table -AutoSize -Wrap

Write-Host '=== MODEL CARDS / METRICS FILES ===' -ForegroundColor Cyan
Get-RepoFiles -Include 'model_card.json','metrics.json','run_meta.json' |
  ForEach-Object {
    Write-Host "--- $($_.FullName) ---" -ForegroundColor Yellow
    Get-Content $_.FullName
    Write-Host ''
  }

Write-Host '=== VIT/CLASSIFIER SERVICE ACTUAL IMPLEMENTATION ===' -ForegroundColor Cyan
Get-Content backend\app\services\vit_service.py
Write-Host ''
Get-Content backend\app\services\change_service.py -ErrorAction SilentlyContinue
Write-Host ''

Write-Host '=== ONNX MODEL SERVICE ===' -ForegroundColor Cyan
Get-Content backend\app\services\onnx_model_service.py -ErrorAction SilentlyContinue

Write-Host '=== ACTUAL MODEL ARTIFACTS ON DISK ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.pt','*.onnx','*.pth' |
  Where-Object { $_.FullName -notmatch '__pycache__|node_modules' } |
  Select-Object FullName, @{n='SizeMB';e={[math]::Round($_.Length/1MB,1)}} |
  Format-Table -AutoSize -Wrap

Write-Host '=== OPTIMIZERS AND SCHEDULERS ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py' |
  Select-String -Pattern 'optim\.|Adam|SGD|AdamW|lr_scheduler|CosineAnnealing|StepLR|ReduceLROnPlateau' |
  Where-Object { $_.Path -notmatch '__pycache__' } |
  Select-Object Path, LineNumber, Line |
  Format-Table -AutoSize -Wrap

Write-Host '=== DATA AUGMENTATION ===' -ForegroundColor Cyan
Get-RepoFiles -Include '*.py' |
  Select-String -Pattern 'transforms\.|RandomCrop|RandomFlip|Normalize|ColorJitter|RandomResized|albumentations|Compose' |
  Where-Object { $_.Path -notmatch '__pycache__' } |
  Select-Object Path, LineNumber, Line |
  Format-Table -AutoSize -Wrap

Stop-Transcript | Out-Null
