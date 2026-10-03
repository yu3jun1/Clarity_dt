# 同卡 seed42 reproducibility audit

旧/新历史结果保留为独立 replicate，不覆盖、不自动剔除。

|Variant|Replicate|H1 MSE|H2 MSE|H3 MSE|Checkpoint step|
|---|---|---|---|---|---|
|A|historical_rep01|0.173009|0.294697|0.626819|432|
|A|historical_rep02|0.166148|0.164044|0.183409|456|
|B|historical_rep01|0.171643|0.154493|0.144453|456|
|B|historical_rep02|0.178738|0.163402|0.155669|432|
|C|historical_rep01|0.149051|0.131929|0.126237|552|
|C|historical_rep02|0.149993|0.135826|0.132754|720|
|D|historical_rep01|0.166711|0.150733|0.142094|432|
|D|historical_rep02|0.173696|0.160152|0.152653|384|
|E|historical_rep01|0.153635|0.152099|0.164183|408|
|A|deterministic_rep01|0.157548|0.144917|0.143666|696|
|A|deterministic_rep02|0.157548|0.144917|0.143666|696|

## 稳定性 gates

- A: stable
- B: pending
- E: pending

旧 A 0.6268: candidate_anomalous_historical_replicate_requires_adjudication；保留原始记录，不能仅因性能差而剔除。

正式 seed42 采用预先指定的 rep01；rep02 用于复现性检查，不把两个 replicate 当成两个独立 seed。
E43/44 仅在 A/B/E 全部通过 gate 后启动。B43/44 暂用历史对照，最终比较注明执行协议差异。
