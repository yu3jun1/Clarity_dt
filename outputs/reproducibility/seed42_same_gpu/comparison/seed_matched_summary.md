# Seed-matched E versus B

- B seeds43/44 are existing historical controls; deterministic environment metadata is unavailable. This is seed-matched, not fully execution-protocol-matched; causal attribution needs matched B43/44 reruns.
- Patient means average windows within each patient, then give each patient equal weight; windows are not independent patients.
- Eight test patients and three seeds are exploratory; no confirmatory significance claim.

|Seed|Group|H1 MSE|H2 MSE|H3 MSE|H3/H1|
|---|---|---|---|---|---|
|42|B|0.179597|0.163246|0.167899|0.9349|
|42|E|0.157861|0.169643|0.219361|1.3896|
|43|B|0.173392|0.161456|0.157044|0.9057|
|43|E|0.157823|0.144282|0.157464|0.9977|
|44|B|0.175624|0.168855|0.157240|0.8953|
|44|E|0.146768|0.130677|0.126816|0.8641|

Patient-level paired differences are in `patient_level_differences.csv`; all horizon metrics and seed-paired differences are in `seed_matched_summary.json`.

Recovery scheduling: unchanged source/configuration/DataLoader protocol; E42 rep01/rep02 restarted from scratch sequentially on original GPU0. Non-resumable incomplete runs were deleted at the user request, not selected/excluded based on test scores. E43 restarted on GPU1; healthy E44 continued on GPU2. The SHM governor limits combined E/F concurrency to three and starts only with >=80 GiB free SHM on an idle assigned GPU.
