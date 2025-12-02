# Run ESLint and capture output to file
Write-Host "Running ESLint check on vite-js..."

# Change to vite-js directory
Set-Location "vite-js"

# Run ESLint and redirect output to file
npx eslint src --max-warnings 0 > eslint-output.txt 2>&1

# Display the output
Write-Host "ESLint output saved to eslint-output.txt"
Write-Host "Contents of output file:"
Get-Content eslint-output.txt

# Go back to root
Set-Location ".."
Write-Host "ESLint check completed."
