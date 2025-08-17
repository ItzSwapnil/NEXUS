Param(
  [string]$CatalogJson = "market_catalog.json",
  [double]$Threshold = 80.0
)

if (-not (Test-Path $CatalogJson)) {
  Write-Error "Catalog file '$CatalogJson' not found. Run verify_catalog.ps1 first."; exit 1
}

try {
  $data = Get-Content $CatalogJson -Raw | ConvertFrom-Json
} catch {
  Write-Error "Failed to parse $CatalogJson: $_"; exit 1
}

if (-not $data) { Write-Error "Empty catalog"; exit 1 }

$below = @()
foreach ($m in $data) {
  $p = $m.display_payout
  if (-not ($p -is [double] -or $p -is [int])) { Write-Error "Non-numeric payout for $($m.symbol)"; exit 1 }
  if ($p -lt $Threshold) { $below += $m.symbol }
}

Write-Host "[verify_payouts] Count=$($data.Count) BelowThreshold=$($below.Count)" -ForegroundColor Cyan
if ($below.Count -gt 0) { Write-Host "Markets below threshold: $($below -join ', ')" }
exit 0

