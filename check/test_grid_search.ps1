# Test script for experiment grid search

Write-Host "Testing experiment grid search with small test suite..." -ForegroundColor Cyan
Write-Host ""

python launch.py `
    --experiments-json experiments/test_small.json `
    --multi-gpu `
    --wandb-group "grid_test" `
    --wandb-mode disabled

Write-Host ""
Write-Host "Test completed!" -ForegroundColor Green
