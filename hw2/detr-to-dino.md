從 DETR 到 DINO，模型架構與訓練機制經歷了數個重大的革命性變化，主要為了解決原始 DETR 收斂速度極慢以及對小物件偵測能力不佳的痛點。以下為你詳細梳理這五大核心變化：
1. 混合查詢選擇 (Mixed Query Selection)
DETR 的做法： Object Queries（物件查詢）是一組完全從零開始的靜態可學習參數（Learnable queries），模型在訓練初期完全沒有任何先驗知識，只能盲目地在全圖尋找物件。
DINO 的變化： 引入了「混合查詢選擇」。模型會先讓 Encoder 掃描整張圖片的特徵，並從中挑選出 Top-K（例如 50 或 900 個）特徵最強的像素位置，直接用這些位置來動態初始化 Decoder 的「位置查詢（即 4D 錨框 Anchor Boxes）」
。而「內容查詢（Content queries）」則保持為可學習的參數
。這讓模型「贏在起跑點」，一開始就能聚焦在最可能有物件的區域。
 以下為你詳細拆解 Object Queries 的運作機制：
一、 Transformer Decoder 與 Object Queries 的建立與特徵抓取
1. Object Queries 是怎麼建立的？
在原始 DETR 中（從零學習）： Object Queries 是一組數量固定（例如你設定的 N=30 或 N=50）的可學習參數 (Learnable positional encodings)
。一開始，它們沒有任何關於影像的先驗知識，是透過在訓練資料中不斷迭代，慢慢學習到在畫面中不同位置、不同大小尋找物件的「統計模式」（例如某些 Query 專門負責抓畫面中間的大物件，某些負責抓左下角的小物件）
。
在 DINO 中（混合查詢選擇 Mixed Query Selection）： 為了改善 DETR 盲目瞎猜導致收斂極慢的問題，DINO 改變了 Queries 的建立方式
。DINO 會先讓 Encoder 掃描整張圖片，並從 Encoder 的特徵中挑選出 Top-K（例如 Top-50）特徵最強、最像物件的位置
。這些位置會被用來動態初始化 Queries 的「位置資訊（即 4D 錨框 Anchor Boxes）」，而 Queries 的「內容特徵 (Content Queries)」則保持為可學習的參數
。這就像是 Encoder 先幫探照燈（Queries）指出了大概的位置，探照燈一打開就已經照在物件附近了。
2. Queries 如何透過 Position Encoding 知道要抓哪些 Pixels 的 Feature？
Self-Attention（避免重複抓取）： 在去抓影像特徵前，Queries 之間會先透過多頭自注意力機制 (Multi-Head Self-Attention) 互相溝通
。這讓它們能交換彼此負責的空間位置，進而抑制對同一個物件產生重複的預測框，這正是 DETR 架構能夠捨棄傳統 NMS (非極大值抑制) 演算法的關鍵原因
。
Cross-Attention（精準抽取特徵）： 接著，Queries 會與 Encoder 傳過來的 H×W 個影像 Tokens 進行交互。影像 Tokens 本身帶有二維的空間位置編碼 (Spatial Positional Encodings)
。
在運算時，Queries 內含的位置資訊會去和影像 Tokens 的空間位置編碼進行內積（配對計算 Attention 分數）
。
DINO 的形變注意力加持： 由於 DINO 採用了形變注意力機制 (Deformable Attention)，Queries 不會盲目地去計算整張圖片所有像素的注意力，而是將自己初始化的 4D 錨框當作參考點 (Reference Points)，只針對參考點周圍的少數幾個關鍵採樣點 (Sampling points) 進行特徵抽取
。這使得模型能以極高的效率聚焦在數字的邊緣與輪廓上
。

2. 對比去噪訓練 (Contrastive DeNoising Training, CDN)
DETR 的做法： 高度依賴二分圖匹配（Bipartite Matching）來分配預測框與真實框。但在訓練初期，預測框是隨機的，導致匹配目標不斷變動（匹配不穩定），使得收斂極慢。
DINO 的變化： 引入了 CDN 機制，在訓練時把「帶有雜訊的真實框 (Noised GT boxes)」當作額外的 Queries 餵給 Decoder，進行「開卷考試」
。DINO 更進一步將雜訊分為兩種：
正樣本（小雜訊）： 教導模型將稍微偏移的框微調回精準的真實位置
。
負樣本（大雜訊）： 刻意給予偏移極大的框，並強迫模型將其預測為「背景（無物件）」，以此教導模型拒絕無用的錨框並消除重複預測
。
3. 向前看兩次 (Look Forward Twice) 的邊界框迭代微調
DETR 的做法： Decoder 的每一層直接預測出最終的絕對座標，層與層之間的梯度傳遞較為基本。
DINO 的變化： Decoder 的每一層都會以上一層輸出的框為基礎，預測一個「偏移量 (Offset)」來進行邊界框迭代微調 (Iterative box refinement)
。同時，DINO 改變了梯度回傳的路徑，第 i 層的參數更新不僅會受到第 i 層本身 Loss 的監督，還會接收來自第 i+1 層預測偏移量的梯度回傳
。這確保了每一層的框更新都朝著更精準的方向優化。
4. 多尺度特徵與形變注意力 (Multi-Scale Features & Deformable Attention)
DETR 的做法： 僅使用 Backbone 的單一尺度特徵（通常是 Layer 4，解析度極低），且使用標準的全局自注意力機制 (Global Attention)，計算量龐大且容易丟失小物件細節
。
DINO 的變化： 吸收了 Deformable DETR 的優點，提取 Backbone 的多尺度特徵（通常是 3 到 4 個尺度，包含高解析度特徵），並改用 MSDeformAttn (多尺度形變注意力)
。模型不再關注全圖所有像素，而是只針對參考點周圍的少數關鍵採樣點進行特徵抽取
，這大幅降低了運算成本並極大化了對小物件的偵測能力。
5. 損失函數與分類頭的重構 (Sigmoid Focal Loss vs. Cross Entropy)
DETR 的做法： 使用傳統的 Softmax 交叉熵損失。如果預測框沒有對應到物件，必須明確預測為一個額外的**「背景類別 (No-object class, ∅)」**（例如類別數 10 + 1 背景 = 11 維）
。
DINO 的變化： 取消了獨立的背景類別設定。DINO 改用 Sigmoid Focal Loss，將每個類別視為獨立的二元分類任務
。所謂的「背景」，就是模型對所有真實類別的預測機率都趨近於 0
。Focal Loss 同時也完美解決了大量無用預測框與少量真實目標之間的類別不平衡問題
。


