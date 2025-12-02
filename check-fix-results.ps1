Write-Host "Checking ESLint results after fixes..."

# Check vite-js
Set-Location "vite-js"
Write-Host "Checking vite-js..."
$npx_enabled = Get-Command npx -ErrorAction SilentlyContinue
if($npx_enabled) {
    npx eslint "src/**/*.{js,jsx,ts,tsx}" --format=compact --config .eslintrc.cjs -o eslint-results.txt
    Write-Host "vite-js results saved to eslint-results.txt"
} else {
    Write-Host "npx not available, skipping vite-js check"
}

# Go back and check minimal-api-dev-v6
Set-Location "..\minimal-api-dev-v6"
Write-Host "Checking minimal-api-dev-v6..."
if($npx_enabled) {
    npx eslint "src/**/*.{js,ts,tsx}" --config .eslintrc.js --format=compact -o eslint-results.txt
    Write-Host "minimal-api-dev-v6 results saved to eslint-results.txt"
} else {
    Write-Host "npx not available, skipping minimal-api-dev-v6 check"
}

Set-Location ".."
Write-Host "Check completed."
