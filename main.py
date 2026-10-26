import json
from ollama import chat, embed

MODEL_NAME = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"

# --------------------------------------------------
# 功能 1：串流輸出 - 智慧客服即時回覆
# --------------------------------------------------
def demo_streaming():
    print("=== 功能 1：串流輸出 (智慧客服即時回覆) ===")
    prompt = "顧客詢問：請問你們的退換貨政策是什麼？請用親切的語氣簡短說明。"
    print(f"顧客提問：{prompt}\n客服回覆：", end="", flush=True)
    
    stream = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")

# --------------------------------------------------
# 功能 2：結構化輸出 - 顧客評價情感與分類分析
# --------------------------------------------------
def demo_structured_output():
    print("=== 功能 2：結構化輸出 (顧客評價自動分類) ===")
    review = "這款外套品質很好，布料摸起來很舒服，但物流太慢了等了足足一週！"
    prompt = f"""請分析以下顧客評價，並嚴格只輸出 JSON 格式：
評價內容："{review}"

JSON 欄位規範：
- sentiment: 情感傾向 (正向/負向/混合)
- category: 問題分類 (商品品質/物流服務/售後客服)
- priority: 處理優先級 (高/中/低)
"""
    response = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        format="json"
    )
    print(f"原始評價：{review}")
    print("分析結果 (JSON)：")
    print(response["message"]["content"])
    print()

# --------------------------------------------------
# 功能 3：工具呼叫 - 客服後台計算折扣工具
# --------------------------------------------------
def calculate_discount(price: float, discount_rate: float) -> float:
    """計算折扣後的商品最終金額"""
    return price * discount_rate

def demo_tool_calling():
    print("=== 功能 3：工具呼叫 (客服折扣計算器) ===")
    print("情境：顧客詢問原價 1500 元的衣服打 85 折是多少錢？")
    
    # 手動模擬 Tool Call 流程
    price = 1500
    rate = 0.85
    result = calculate_discount(price, rate)
    
    prompt = f"原價 {price} 元，打 {rate} 折後的計算結果為 {result} 元。請以客服身份簡短回覆顧客。"
    response = chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    print("AI 呼叫工具後的回答：")
    print(response["message"]["content"])
    print()

# --------------------------------------------------
# 功能 4：Embedding - 客服 FAQ 知識庫相似度比對
# --------------------------------------------------
def demo_embedding():
    print("=== 功能 4：Embedding (FAQ 相似問題檢索) ===")
    faq_question = "如何申請退款與退貨？"
    user_query = "我想把買的東西退掉要怎麼做？"
    
    res1 = embed(model=EMBED_MODEL, input=faq_question)
    res2 = embed(model=EMBED_MODEL, input=user_query)
    
    vec1 = res1["embeddings"][0]
    vec2 = res2["embeddings"][0]
    
    print(f"FAQ 標準問題：{faq_question}")
    print(f"顧客實際提問：{user_query}")
    print(f"FAQ 向量維度：{len(vec1)}")
    print(f"提問向量維度：{len(vec2)}")
    print(f"向量數值範例 (前 5 維)：{vec1[:5]}")
    print()

# --------------------------------------------------
# 主執行入口
# --------------------------------------------------
def main():
    print("========================================")
    print("  電商智慧客服與評價分析系統 (AI 平台期末專案)")
    print("========================================\n")
    demo_streaming()
    demo_structured_output()
    demo_tool_calling()
    demo_embedding()
    print("----------------------------------------")
    print("系統執行完畢！")

if __name__ == "__main__":
    main()