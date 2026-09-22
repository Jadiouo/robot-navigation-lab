# 離線互動展示

展示器把封存的 result archives 整理為一個可以單獨拷貝、以 `file://`
開啟的 HTML 資料夾。它只讀取既有的 JSON、CSV 和 zip；不訓練、不建立新
rollout，也不讀取課程作業目錄或選配的 PyTorch。

在 repository 根目錄執行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/navlab showcase --results docs/results --output outputs/showcase
```

把 `outputs/showcase/` 整個資料夾複製到任何位置後，直接開啟
`index.html` 即可。頁面內嵌 catalog、CSS 與 JavaScript，不使用 CDN、
`fetch` 或外部字體；`raw/` 保留兩份原始 archive 及 SHA-256 sidecar 供
下載核對。`build-manifest.json` 另保存兩 archive、三份 web source resource
及生成 `index.html` 的 SHA-256，讓展示輸出版本可追溯。若使用 localhost
瀏覽，可從 repository 根目錄執行：

```bash
python3 -m http.server 8000
```

再開啟 `http://localhost:8000/outputs/showcase/`。

catalog schema 是 v1，固定包含所有案例的地圖、reference trajectory、
實際 trace、run manifest 與原始文字內容。展示的時間是 post-step
`(step + 1) * dt`；t=0 只顯示 manifest 的初始狀態。比較表固定以 20 個
held-out route 的全部回合計算，失敗回合不會被隱藏。

資料來源缺少 archive、checksum 不相符、catalog schema 不符，或 web
template marker 損壞時，`navlab showcase` 會失敗並指出來源路徑，而不會
產生看似完整的頁面。
