# 來源與素材狀態

本作品從零實作並僅使用程式生成的地圖與路線；執行時不會 import 課程作業目錄或 `rewrite-reference`。

課程 HW1、HW2、HW3 是設計脈絡：規劃、運動學追蹤與一維 PPO steering 的問題設定啟發本作品，但舊程式、既有模型、圖表與訓練數字不當作本作品的實驗證據。新數字只來自此 package 的 `trace.csv` 和 manifest。

HW3-2 的 Proly/Unity 遊戲未包含在此作品。其公開散布權限在本 workspace 沒有可驗證證據，因此它既不是依賴也不是展示素材；checkpoint progress、死亡 transition 和 reward-hacking 防護等觀察僅可能在日後、自建環境且有明確研究問題時重新驗證。

本次重製採多 agent 分工：planning/trajectory、simulation/control、delivery/evaluation。所有公開發布前仍由擁有者確認要上傳的素材與來源狀態；本 repo 以 [MIT License](../LICENSE) 發布。
