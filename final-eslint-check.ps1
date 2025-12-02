# Final ESLint check to show remaining errors
Set-Location "vite-js"
Write-Host "Running final ESLint check on vite-js..."
$viteOutput = npx eslint "src/**/*.{js,jsx,ts,tsx}" --format=compact --max-warnings=200 2>&1
$viteOutput | Out-File -FilePath ..\vite-js-eslint-remaining.txt -Encoding UTF8
$viteOutput
Write-Host "`nChecking minimal-api-dev-v6..."
Set-Location "..\minimal-api-dev-v6"
$apiOutput = npx eslint "src/**/*.{js,ts,tsx}" --format=compact --max-warnings=200 2>&1
$apiOutput | Out-File -FilePath ..\minimal-api-dev-v6-eslint-remaining.txt -Encoding UTF8
$apiOutput
Write-Host "`nESLint check complete."
Set-Location ".."
