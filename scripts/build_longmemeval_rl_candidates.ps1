[CmdletBinding()]
param(
    [ValidateSet("nsp_text_tiling", "bert_mlp_text_tiling")]
    [string]$Method = "nsp_text_tiling",

    [ValidateSet("small", "medium", "large")]
    [string[]]$Tier = @("small", "medium", "large"),

    [string]$RunPrefix = "longmemeval_full",

    [string]$CampaignId = "",

    [int]$StartIndex = 0,

    [int]$Limit = 0,

    [switch]$RetryTerminal,

    [switch]$ContinueOnError,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$samplesRoot = Join-Path $projectRoot "datasets\segmented\longmemeval\full\$Method\samples"
$candidateScript = Join-Path $projectRoot "scripts\build_rl_candidates.py"
$campaignScript = Join-Path $projectRoot "scripts\manage_extraction_campaign.py"
$runsRoot = Join-Path $projectRoot "outputs\rl_router\runs"
$resolvedCampaignId = if ($CampaignId) { $CampaignId } else { "${RunPrefix}_${Method}" }

if (-not (Test-Path -LiteralPath $samplesRoot -PathType Container)) {
    throw "LongMemEval segmented samples directory is missing: $samplesRoot"
}
if (-not (Test-Path -LiteralPath $candidateScript -PathType Leaf)) {
    throw "Candidate extraction script is missing: $candidateScript"
}
if (-not (Test-Path -LiteralPath $campaignScript -PathType Leaf)) {
    throw "Extraction campaign script is missing: $campaignScript"
}
if ($StartIndex -lt 0 -or $Limit -lt 0) {
    throw "StartIndex and Limit must be non-negative."
}

$tierOrder = @("small", "medium", "large")
$selectedTiers = @($tierOrder | Where-Object { $Tier -contains $_ })
if ($selectedTiers.Count -eq 0) {
    throw "At least one tier must be selected."
}

# Fail before the first sample creates a run. The Python entry point repeats this
# validation so direct single-sample invocations have the same guarantee.
$keyByTier = @{
    small = "SILICONFLOW_API_KEY"
    medium = "MEDIUM_MODEL_API_KEY"
    large = "LARGE_MODEL_API_KEY"
}
$missingKeys = @(
    foreach ($selectedTier in $selectedTiers) {
        $keyName = $keyByTier[$selectedTier]
        if (-not [Environment]::GetEnvironmentVariable($keyName)) {
            "$selectedTier=$keyName"
        }
    }
)
if ($missingKeys.Count -gt 0 -and -not $DryRun) {
    throw "Missing API key environment variables: $($missingKeys -join ', ')"
}

$segmentFiles = @(
    Get-ChildItem -LiteralPath $samplesRoot -Filter "segments.jsonl" -File -Recurse |
        Sort-Object FullName
)
if ($StartIndex -ge $segmentFiles.Count) {
    throw "StartIndex $StartIndex is outside the $($segmentFiles.Count) discovered samples."
}
$remaining = @($segmentFiles | Select-Object -Skip $StartIndex)
if ($Limit -gt 0) {
    $remaining = @($remaining | Select-Object -First $Limit)
}

$campaignArguments = @(
    "run", "python", $campaignScript, "init",
    "--campaign-id", $resolvedCampaignId,
    "--dataset", "longmemeval",
    "--split", "full",
    "--method", $Method,
    "--run-prefix", $RunPrefix
)
if ($DryRun) {
    Write-Host "uv $($campaignArguments -join ' ')"
}
else {
    & uv @campaignArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize extraction campaign: $resolvedCampaignId"
    }
}

$failures = [System.Collections.Generic.List[object]]::new()
$openProviderCircuits = [System.Collections.Generic.HashSet[string]]::new()
$processed = 0
foreach ($segmentFile in $remaining) {
    $sampleId = $segmentFile.Directory.Name
    $runId = "${RunPrefix}_${Method}_${sampleId}"
    $manifestPath = Join-Path $runsRoot "$runId\manifest.json"
    $runExists = Test-Path -LiteralPath $manifestPath -PathType Leaf

    foreach ($selectedTier in $selectedTiers) {
        if ($openProviderCircuits.Contains($selectedTier)) {
            Write-Warning "Skipping sample=$sampleId tier=$selectedTier because its provider circuit is open."
            continue
        }
        $modeArguments = if ($runExists) {
            @("--resume", $runId)
        }
        else {
            @("--extraction-run-id", $runId)
        }
        $arguments = @(
            "run",
            "python",
            $candidateScript,
            $segmentFile.FullName
        ) + $modeArguments + @(
            "--tier", $selectedTier,
            "--campaign-id", $resolvedCampaignId
        )
        if ($RetryTerminal -and $runExists) {
            $arguments += "--retry-terminal"
        }

        Write-Host "[$($processed + 1)/$($remaining.Count)] sample=$sampleId tier=$selectedTier run=$runId"
        if ($DryRun) {
            Write-Host "uv $($arguments -join ' ')"
            $runExists = $true
            continue
        }

        & uv @arguments
        if ($LASTEXITCODE -ne 0) {
            $failure = [pscustomobject]@{
                sample_id = $sampleId
                tier = $selectedTier
                run_id = $runId
                exit_code = $LASTEXITCODE
            }
            $failures.Add($failure)
            if ($LASTEXITCODE -eq 10) {
                [void]$openProviderCircuits.Add($selectedTier)
            }
            if (-not $ContinueOnError) {
                throw "Candidate extraction failed: sample=$sampleId tier=$selectedTier exit=$LASTEXITCODE"
            }
            continue
        }
        $runExists = $true
    }
    $processed += 1
}

$summary = [pscustomobject]@{
    campaign_id = $resolvedCampaignId
    method = $Method
    selected_tiers = $selectedTiers
    discovered_samples = $segmentFiles.Count
    scheduled_samples = $remaining.Count
    processed_samples = $processed
    failures = @($failures)
    open_provider_circuits = @($openProviderCircuits)
}
if (-not $DryRun) {
    $refreshArguments = @(
        "run", "python", $campaignScript, "refresh",
        "--campaign-id", $resolvedCampaignId,
        "--method", $Method,
        "--run-prefix", $RunPrefix
    )
    & uv @refreshArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to refresh extraction campaign: $resolvedCampaignId"
    }
}
$summary | ConvertTo-Json -Depth 5
if ($failures.Count -gt 0) {
    exit 2
}
