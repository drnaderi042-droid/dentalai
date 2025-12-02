# Script to run ESLint on vite-js frontend and output to file
Set-Location "vite-js"
Write-Host "Checking ESLint errors on vite-js frontend..."

# Run ESLint with quiet mode to show only errors
npx eslint src --quiet > frontend-errors.txt 2>&1

Write-Host "ESLint completed. Results saved to frontend-errors.txt"
Write-Host "Sample output:"
Get-Content frontend-errors.txt | Select-Object -First 50

Set-Location ".."
