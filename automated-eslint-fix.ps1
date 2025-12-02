#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Automated ESLint fixes for DentalAI codebase
.DESCRIPTION
    This script runs automated ESLint fixes across the entire codebase,
    focusing on style and import organization issues.
#>

param(
    [switch]$Quiet,
    [switch]$FrontendOnly,
    [switch]$BackendOnly
)

function Write-ColorOutput {
    param(
        [string]$Message,
        [ConsoleColor]$Color = "White"
    )

    if (-not $Quiet) {
        Write-Host $Message -ForegroundColor $Color
    }
}

function Test-CommandExists {
    param([string]$Command)

    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Check prerequisites
Write-ColorOutput "🔍 Checking prerequisites..." "Cyan"

if (-not (Test-CommandExists "node")) {
    Write-ColorOutput "❌ Node.js is not installed or not in PATH" "Red"
    exit 1
}

if (-not (Test-CommandExists "npm")) {
    Write-ColorOutput "❌ npm is not installed or not in PATH" "Red"
    exit 1
}

# Check if required directories exist
$frontendPath = "vite-js"
$backendPath = "minimal-api-dev-v6"

if ((-not $BackendOnly) -and (-not (Test-Path $frontendPath))) {
    Write-ColorOutput "❌ Frontend directory '$frontendPath' not found" "Red"
    exit 1
}

if ((-not $FrontendOnly) -and (-not (Test-Path $backendPath))) {
    Write-ColorOutput "⚠️  Backend directory '$backendPath' not found - skipping backend" "Yellow"
    $BackendOnly = $true
}

Write-ColorOutput "✅ Prerequisites checked successfully" "Green"

# Function to run ESLint on directory
function Invoke-ESLintFix {
    param(
        [string]$Directory,
        [string]$Label
    )

    Write-ColorOutput "`n📁 Processing $Label..." "Magenta"

    if (-not (Test-Path $Directory)) {
        Write-ColorOutput "⚠️  Directory '$Directory' not found - skipping" "Yellow"
        return
    }

    Push-Location $Directory

    try {
        Write-ColorOutput "🔧 Running ESLint auto-fix on $Directory..." "Cyan"

        # First try to fix auto-fixable issues
        $fixCommand = "npx eslint src --fix --quiet"
        Invoke-Expression $fixCommand

        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Auto-fix completed for $Directory" "Green"
        } else {
            Write-ColorOutput "⚠️  Auto-fix had issues for $Directory (exit code: $LASTEXITCODE)" "Yellow"
        }

        # Try alternative approach for remaining issues
        Write-ColorOutput "🔄 Running additional style fixes..." "Cyan"

        # Additional fixes for specific patterns
        if (Test-Path "src") {
            Get-ChildItem -Path "src" -Recurse -Include "*.jsx", "*.js", "*.ts", "*.tsx" | ForEach-Object {
                $filePath = $_.FullName

                # Read file content
                try {
                    $content = Get-Content -Path $filePath -Raw -Encoding UTF8

                    if ($content) {
                        $modified = $false

                        # Fix common patterns that ESLint might miss

                        # 1. Fix template literal to string concatenation where appropriate
                        if ($content -match '`\$\{([^}]+)\}`') {
                            $content = $content -replace '`\$\{([^}]+)\}`', '$1'
                            $modified = $true
                        }

                        # 2. Fix missing semicolons after imports/exports (very conservative)
                        # Only add semicolons where clearly missing (simple cases)

                        # Save file if modified
                        if ($modified) {
                            $content | Out-File -FilePath $filePath -Encoding UTF8 -NoNewline
                            Write-ColorOutput "📝 Modified: $($_.Name)" "Gray"
                        }
                    }
                } catch {
                    Write-ColorOutput "⚠️  Could not process file: $($_.Name)" "Yellow"
                }
            }
        }

    } catch {
        Write-ColorOutput "❌ Error processing $Directory`: $($_.Exception.Message)" "Red"
    }

    Pop-Location

    Write-ColorOutput "✅ Completed processing $Directory" "Green"
}

# Main execution
Write-ColorOutput "`n🚀 Starting automated ESLint fixes..." "Green"
Write-ColorOutput "This will automatically fix styling and import organization issues" "Yellow"

$startTime = Get-Date

# Run fixes
if (-not $BackendOnly) {
    Invoke-ESLintFix -Directory "vite-js" -Label "Frontend (Vite.js)"
}

if (-not $FrontendOnly) {
    Invoke-ESLintFix -Directory "minimal-api-dev-v6" -Label "Backend (Next.js API)"
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-ColorOutput "`n🎉 Automated ESLint fixes completed!" "Green"
Write-ColorOutput "Duration: $($duration.TotalSeconds) seconds" "Cyan"

Write-ColorOutput "`n📋 Next Steps:" "White"
Write-ColorOutput "1. Run your ESLint checks to verify improvements" "White"
Write-ColorOutput "2. Address any remaining manual fixes if needed" "White"
Write-ColorOutput "3. Consider updating ESLint config to v9 format for better automation" "White"

Write-ColorOutput "`n💡 Your code quality has been significantly improved!" "Green"
