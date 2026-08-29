[CmdletBinding()]
param(
    [ValidateSet("locomo", "longmemeval")]
    [string]$Dataset = "longmemeval",

    [ValidateSet("nsp_text_tiling", "bert_mlp_text_tiling")]
    [string]$Method = "nsp_text_tiling",

    [double]$Alpha = 0.5,

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
if ($Alpha -lt 0) {
    throw "Alpha must be non-negative."
}
$alphaText = $Alpha.ToString(
    "0.############################",
    [System.Globalization.CultureInfo]::InvariantCulture
)
$alphaToken = $alphaText.Replace(".", "p")
$segmentationMethod = "${Method}_alpha_${alphaToken}"
$samplesRoot = Join-Path $projectRoot "datasets\segmented\$Dataset\full\$segmentationMethod\samples"
$candidateScript = Join-Path $projectRoot "scripts\build_rl_candidates.py"
$campaignScript = Join-Path $projectRoot "scripts\manage_extraction_campaign.py"
$runsRoot = Join-Path $projectRoot "outputs\rl_router\runs"
$resolvedCampaignId = if ($CampaignId) { $CampaignId } else { "${RunPrefix}_${segmentationMethod}" }

if (-not (Test-Path -LiteralPath $samplesRoot -PathType Container)) {
    throw "$Dataset segmented samples directory is missing: $samplesRoot"
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
    "--dataset", $Dataset,
    "--split", "full",
    "--method", $segmentationMethod,
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
$failedSamples = [System.Collections.Generic.HashSet[string]]::new()
$openProviderCircuits = [System.Collections.Generic.HashSet[string]]::new()
$processed = 0
$completedUnits = 0
$totalUnits = $remaining.Count * $selectedTiers.Count
$overallStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$overallCalls = 0
$overallInputTokens = 0
$overallOutputTokens = 0
$overallCost = 0.0
$overallFacts = 0
foreach ($segmentFile in $remaining) {
    $sampleId = $segmentFile.Directory.Name
    $runId = "${RunPrefix}_${segmentationMethod}_${sampleId}"
    $manifestPath = Join-Path $runsRoot "$runId\manifest.json"
    $runExists = Test-Path -LiteralPath $manifestPath -PathType Leaf
    $sampleHadFailure = $false

    foreach ($selectedTier in $selectedTiers) {
        $percent = if ($totalUnits -gt 0) {
            [Math]::Floor(100 * $completedUnits / $totalUnits)
        }
        else { 100 }
        Write-Progress `
            -Activity "$Dataset candidate extraction" `
            -Status "sample=$sampleId tier=$selectedTier ($completedUnits/$totalUnits tier jobs)" `
            -PercentComplete $percent
        if ($openProviderCircuits.Contains($selectedTier)) {
            Write-Warning "Skipping sample=$sampleId tier=$selectedTier because its provider circuit is open."
            $sampleHadFailure = $true
            [void]$failedSamples.Add($sampleId)
            $completedUnits += 1
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
            $completedUnits += 1
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
            $sampleHadFailure = $true
            [void]$failedSamples.Add($sampleId)
            if ($LASTEXITCODE -eq 10) {
                [void]$openProviderCircuits.Add($selectedTier)
            }
            if (-not $ContinueOnError) {
                throw "Candidate extraction failed: sample=$sampleId tier=$selectedTier exit=$LASTEXITCODE"
            }
            $completedUnits += 1
            continue
        }
        $runExists = $true
        $completedUnits += 1
    }
    $processed += 1
    if (-not $DryRun -and (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        $runManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $extractionSummary = $runManifest.extraction_summary
        if ($null -ne $extractionSummary) {
            $sampleFacts = @(
                $extractionSummary.fact_counts.PSObject.Properties |
                    ForEach-Object { [int]$_.Value }
            ) | Measure-Object -Sum | Select-Object -ExpandProperty Sum
            $attempts = $extractionSummary.attempt_summary
            $sampleCalls = [int]$attempts.logical_api_calls
            $sampleInputTokens = [int]$attempts.provider_input_tokens
            $sampleOutputTokens = [int]$attempts.provider_output_tokens
            $sampleCost = [double]$extractionSummary.known_cost
            $overallFacts += [int]$sampleFacts
            $overallCalls += $sampleCalls
            $overallInputTokens += $sampleInputTokens
            $overallOutputTokens += $sampleOutputTokens
            $overallCost += $sampleCost
            $sampleSummaryLine = (
                "SAMPLE SUMMARY sample={0} status={1} facts={2} calls={3} " +
                "input_tokens={4} output_tokens={5} total_tokens={6} cost={7:N6}"
            ) -f @(
                $sampleId,
                $(if ($sampleHadFailure) { "failed" } else { $extractionSummary.status }),
                $sampleFacts,
                $sampleCalls,
                $sampleInputTokens,
                $sampleOutputTokens,
                ($sampleInputTokens + $sampleOutputTokens),
                $sampleCost
            )
            Write-Host $sampleSummaryLine
        }
    }
}
$overallStopwatch.Stop()
Write-Progress -Activity "$Dataset candidate extraction" -Completed

$failedSampleCount = $failedSamples.Count

$summary = [pscustomobject]@{
    campaign_id = $resolvedCampaignId
    dataset = $Dataset
    segmentation_algorithm = $Method
    adaptive_alpha = $Alpha
    segmentation_method = $segmentationMethod
    selected_tiers = $selectedTiers
    discovered_samples = $segmentFiles.Count
    scheduled_samples = $remaining.Count
    processed_samples = $processed
    successful_samples = $processed - $failedSampleCount
    failed_samples = $failedSampleCount
    wall_time_seconds = $overallStopwatch.Elapsed.TotalSeconds
    total_facts = $overallFacts
    total_api_calls = $overallCalls
    total_input_tokens = $overallInputTokens
    total_output_tokens = $overallOutputTokens
    total_tokens = $overallInputTokens + $overallOutputTokens
    total_known_cost = $overallCost
    failures = @($failures)
    open_provider_circuits = @($openProviderCircuits)
}
if (-not $DryRun) {
    $refreshArguments = @(
        "run", "python", $campaignScript, "refresh",
        "--campaign-id", $resolvedCampaignId,
        "--method", $segmentationMethod,
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
