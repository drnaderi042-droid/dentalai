# PowerShell script to fix ESLint issues
Write-Host "Starting ESLint fix process..."

# Change to vite-js directory and run eslint --fix
Set-Location "vite-js"
Write-Host "Running ESLint fix for vite-js..."
npx eslint src --ext .js,.jsx,.ts,.tsx --fix
Write-Host "vite-js ESLint fix completed."

# Change to minimal-api-dev-v6 directory and run eslint --fix
Set-Location "..\minimal-api-dev-v6"
Write-Host "Running ESLint fix for minimal-api-dev-v6..."
npx eslint src --ext .js,.ts,.tsx --fix
Write-Host "minimal-api-dev-v6 ESLint fix completed."

# Go back to root
Set-Location ".."
Write-Host "ESLint fix process completed."
