[CmdletBinding()]
param(
    [ValidateSet("nsp_text_tiling", "bert_mlp_text_tiling")]
    [string]$Method = "nsp_text_tiling",

    [double]$Alpha = 0.5,

    [ValidateSet("small", "medium", "large")]
    [string[]]$Tier = @("small", "medium", "large"),

    [string]$RunPrefix = "locomo_full",

    [string]$CampaignId = "",

    [int]$StartIndex = 0,

    [int]$Limit = 0,

    [switch]$RetryTerminal,

    [switch]$ContinueOnError,

    [switch]$DryRun
)

$shared = Join-Path $PSScriptRoot "build_longmemeval_rl_candidates.ps1"
& $shared -Dataset locomo @PSBoundParameters
exit $LASTEXITCODE
