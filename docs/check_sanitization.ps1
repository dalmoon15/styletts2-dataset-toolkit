# Repository Sanitization Checker
# Scans for personalized paths and references that should be removed before public release

param(
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Repository Sanitization Checker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Define patterns to search for (personal references that should be removed/generalized)
$patterns = @{
    "E: Drive References" = @(
        "E:\\AI\\",
        "E:/AI/",
        "E:\\styletts2-dataset-toolkit",
        "E:/styletts2-dataset-toolkit"
    )
    "Personal Usernames" = @(
        "Lostenergydrink",
        "JinwooSung",
        "\\Lost\\",
        "/Lost/"
    )
    "Personal Paths" = @(
        "C:\\Users\\Lost",
        "my_voice",
        "my-voice",
        "your_model\.pth"
    )
    "Suspicious Placeholders" = @(
        "/path/to/",
        "your_voice",
        "finetuned/your"
    )
}

# Exclude patterns (files/directories to skip)
$excludePatterns = @(
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "check_sanitization.ps1"  # Don't check this script itself
)

$totalIssues = 0
$fileResults = @{}

# Get all text files (code, docs, config)
$extensions = @("*.md", "*.py", "*.ps1", "*.bat", "*.yml", "*.yaml", "*.txt", "*.json")
$allFiles = Get-ChildItem -Path . -Include $extensions -Recurse -File | Where-Object {
    $file = $_
    $shouldExclude = $false
    foreach ($exclude in $excludePatterns) {
        if ($file.FullName -match [regex]::Escape($exclude)) {
            $shouldExclude = $true
            break
        }
    }
    -not $shouldExclude
}

Write-Host "Scanning $($allFiles.Count) files..." -ForegroundColor Yellow
Write-Host ""

foreach ($category in $patterns.Keys) {
    $categoryIssues = 0
    $categoryFiles = @{}
    
    foreach ($pattern in $patterns[$category]) {
        foreach ($file in $allFiles) {
            try {
                $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
                if ($content -match $pattern) {
                    $matches = Select-String -Path $file.FullName -Pattern $pattern -AllMatches
                    
                    if ($matches) {
                        $relativePath = $file.FullName.Replace((Get-Location).Path + "\", "")
                        
                        if (-not $categoryFiles.ContainsKey($relativePath)) {
                            $categoryFiles[$relativePath] = @()
                        }
                        
                        foreach ($match in $matches) {
                            $categoryFiles[$relativePath] += @{
                                Pattern = $pattern
                                Line = $match.LineNumber
                                Text = $match.Line.Trim()
                            }
                            $categoryIssues++
                        }
                    }
                }
            }
            catch {
                # Skip binary files or files that can't be read
            }
        }
    }
    
    if ($categoryIssues -gt 0) {
        Write-Host "❌ $category" -ForegroundColor Red
        Write-Host "   Found $categoryIssues issue(s) in $($categoryFiles.Keys.Count) file(s)" -ForegroundColor Yellow
        
        if ($Verbose) {
            foreach ($filePath in $categoryFiles.Keys) {
                Write-Host "   📄 $filePath" -ForegroundColor Gray
                foreach ($issue in $categoryFiles[$filePath]) {
                    Write-Host "      Line $($issue.Line): $($issue.Pattern)" -ForegroundColor DarkGray
                    Write-Host "         $($issue.Text.Substring(0, [Math]::Min(80, $issue.Text.Length)))" -ForegroundColor DarkGray
                }
            }
            Write-Host ""
        }
        else {
            foreach ($filePath in $categoryFiles.Keys) {
                Write-Host "   📄 $filePath ($($categoryFiles[$filePath].Count) issue(s))" -ForegroundColor Gray
            }
            Write-Host ""
        }
        
        $totalIssues += $categoryIssues
    }
    else {
        Write-Host "✅ $category" -ForegroundColor Green
        Write-Host "   No issues found" -ForegroundColor Gray
        Write-Host ""
    }
}

Write-Host "========================================" -ForegroundColor Cyan
if ($totalIssues -eq 0) {
    Write-Host "✅ REPOSITORY IS CLEAN" -ForegroundColor Green
    Write-Host "No personalization issues found!" -ForegroundColor Green
}
else {
    Write-Host "❌ ISSUES FOUND: $totalIssues" -ForegroundColor Red
    Write-Host "Run with -Verbose flag for detailed line-by-line output" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($totalIssues -gt 0) {
    exit 1
}
exit 0
