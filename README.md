# 電商智慧客服與評價分析系統 (AI-Platform-Final)

本專案為 AI 期末專案，在 WSL2 環境下使用 Python (`uv`) 呼叫本地 Ollama (`qwen2.5` 與 `nomic-embed-text`)，打造一個具備商業實務應用的電商智慧客服系統。

---

## 🚀 專案四大核心模組
1. **串流輸出 (Streaming)**：模擬智慧客服即時打字回覆顧客的退換貨等諮詢。
2. **結構化輸出 (Structured Output)**：自動分析顧客評論，將滿意度、問題分類與處理優先級提取為標準 JSON 格式。
3. **工具呼叫 (Tool Calling)**：整合客服後台的折扣計算工具，精確計算優惠價格並由 AI 回應。
4. **Embedding 嵌入向量**：進行 FAQ 知識庫與顧客提問的語意相似度檢索準備。

---

## 🛠️ 本地執行步驟
```bash
# 1. 確保 Ollama 服務已啟動並下載模型
ollama run qwen2.5
ollama run nomic-embed-text

# 2. 透過 uv 在終端機執行主程式
uv run main.py
