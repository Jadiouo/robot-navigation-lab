# v2 決策規則（人類可讀版）

權威版本是 `protocol_addendum_v2.json`（canonical-JSON SHA-256 存於其 `hash` 欄）；本檔與其一致，若有出入以 JSON 為準。
附錄鏈接協定 r2（`test_protocol_v2.json`，hash `aba7a45c7972a260ba47de1c834281e697ed93f70ca4a924b580bcec653c6e47`），
先於任何 PPO v2 正式訓練與任何測試 episode 凍結（git tag `v2-protocol-addendum`）。附錄不改動協定 r2、場景集、baseline 與 P3/P4 凍結。

## 1. 主線
`ppo2_s0..2` = 完整配方：新 reward + 行人速度特徵 + hall curriculum（level <= 3.5），3 seed，約 6 h。

## 2. 消融臂
`ppo2nv_s0..2`：與主線完全相同的配方、checkpoint 選擇與 headline 規則，**唯一差別 `use_agent_velocity=False`**。
- 預算：相同 env 步數 = 主線三個 seed 最終訓練步數（`train_meta.json` 的 `steps`）的中位數（以 `--max-steps` 設定；`--steps` 的 curriculum／lr 排程沿用主線；不以牆鐘為預算）。
- 順序：主線訓練、選擇並凍結（`ppo2_frozen.json`）之後，**依序**訓練（不同時）。
- 凍結：`ppo2nv_frozen.json` 鏈接 `ppo2_frozen.json` 的 hash 與附錄 hash；ObsSpec2 須 `use_agent_velocity=False` 且其餘欄位與主線相同。
- 評估：`ppo2nv` 套件 = 與 ppo2 相同的套件（nominal、壓力軸、mclbreak；11520 jobs）加上其 warehouse 部分（3 x 300 = 900）= 12420 jobs。
  主線、baseline、v1、warehouse 只需 `ppo2_frozen.json`，不必等消融臂。

## 3. 失敗分支（事先寫死）
- 若 G1 中「ppo2 headline @MCL vs dwa @MCL」（nominal，240 對）在 Holm 校正（G1 家族內）後**顯著較差**（Holm p < 0.05 且成功率差 < 0）：
  **收攤**。不再於此測試集上迭代任何 PPO 版本，v1 + v2 寫成負面結果；任何後續方法（BC 暖啟動、residual RL 等）必須使用全新測試 seed（>= 400000）與新協定。
- 若不顯著較差或顯著較好：照實報告；同樣不得在此測試集上再迭代任何 PPO 版本。

## 4. pilot 的角色
僅作 bug／退化解（停住、全 timeout、速度塌陷）閘門，**不用於選臂或調係數**（樣本與預算太小）。
若揭露 bug 可修，但修正須在附錄凍結前，或以附錄修訂記錄（帶日期、鏈接本附錄 hash）。

## 5. 預先揭露
1. v2 是整包改動（reward + 速度特徵 + curriculum）：主線結論歸因於配方；速度特徵的效果只由消融臂（G6）回答。
2. baseline 參數維持 P3 凍結，PPO 經兩輪設計迭代：調參不對稱。
3. PPO 訓練時 reward 使用真值淨空塑形（policy 輸入無真值）。
4. warehouse 協定 r1 -> r2 因目視 QA 修訂（在任何 episode 之前）。
5. 速度特徵門檻以 8 個訓練範圍場景診斷調整。
6. mcl_aug 為探索性。

## 新增預先宣告檢定族（方法同協定：exact McNemar、家族內 Holm、paired bootstrap 10000 次 seed 0）
| 家族 | 性質 | 內容 | 檢定數 |
|---|---|---|---|
| G6 | 確認性 | ppo2 headline vs ppo2nv headline，nominal，240 對，GT 與 MCL 各一 | 2 |
| G7 | 探索性 | ppo2nv headline vs dwa、pp_stop，nominal，GT／MCL | 4 |

缺資料的檢定以 p = 1 計入 Holm，報告標註「尚未執行」。報表的 `loc_induced` 分類對 `mcl` 與 `mcl_aug` 都生效。
