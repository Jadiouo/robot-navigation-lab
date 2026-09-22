# 方法、資料契約與限制

`robot-navigation-lab` 是二維靜態已知地圖上的導航模擬與實驗平台。車輛參考點是後軸中心；座標使用公尺、角度使用弧度，x 向右、y 向上。它不包含感知、SLAM、動態障礙物、輪胎模型或實車安全保證。

每次 demo 依序執行：A* 或 RRT* 幾何規劃、受車體碰撞與曲率限制驗證的 trajectory construction、曲率限速與終點煞停、共用 bicycle actuator 下的追蹤控制。成功同時要求終點位置、低速、有效進度與無碰撞；規劃失敗、軌跡不可行、collision、timeout 都會保留為結果，而非從表格移除。

控制比較只在相同場景、起始條件、reference trajectory、車模、縱向控制、轉角/轉角速率限制與評測 seed 下成立。PPO 與古典控制器都只使用同源的 `PolicyInput` 有限預覽資訊：PPO 將它 encode 成 clipped/normalized 40 維向量，古典控制器直接讀取未正規化欄位。它不會沿用課程 HW3 的 14 維 observation 或 checkpoint。PPO 的 entropy bonus 是未經 tanh squash 的 Normal distribution entropy，作為探索正則項，不能視為已執行動作分布的精確 entropy。

## 如何檢查一筆結果

每個 run 的 `trace.csv` 是原始逐步資料；`trajectory.csv` 是不可變的 reference；`run.json` 保存命令、seed、車輛設定、套件版本、程式 digest 與終態原因。所有 summary 指標都由 trace 加上 manifest 的終態重新計算。`overview.png` 和 `replay.gif` 使用同一份 trace，不含人工挑選的軌跡。

`benchmark` 的三個群組分別是完整漏斗、共同軌跡的控制器比較，以及 backward speed pass × lookahead braking 的 2×2 消融。`--quick` 刻意縮小矩陣，並在 `benchmark.json` 揭露；它是 smoke test，不應視為完整實驗。
