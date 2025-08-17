# Verifies presence of key spec-required modules and approximate file count granularity.

$required = @(
  'nexus/core/engine.py',
  'nexus/catalog/ingest.py',
  'nexus/payouts/fetch.py',
  'nexus/intelligence/exploration.py',
  'nexus/intelligence/fitness.py',
  'nexus/intelligence/promotion.py',
  'tests/test_catalog.py',
  'tests/test_payouts.py',
  'tests/test_fitness_promotion.py'
)

$missing = @()
foreach ($r in $required) {
  if (-not (Test-Path $r)) { $missing += $r }
}

if ($missing.Count -gt 0) {
  Write-Error "Missing required files: $($missing -join ', ')"; exit 1
}

# Rough granularity check: ensure at least 40 python source files (small modules style)
$pyFiles = Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '__pycache__' }
if ($pyFiles.Count -lt 40) {
  Write-Warning "Project has only $($pyFiles.Count) .py files (<40). Granularity target not met."; exit 2
}
Write-Host "[verify_project_layout] OK ($($pyFiles.Count) python files)" -ForegroundColor Green
exit 0

