import json
import ollama

MODEL_NAME = 'qwen2.5'

# ---------------------------------------------------------
# 1. 串流輸出 (Streaming)
# ---------------------------------------------------------
def demo_streaming():
    print("-" * 50)
    print("【1. 串流輸出 Streaming 示範】")
    print("-" * 50)
    prompt = "請用繁體中文以 100 字左右介紹什麼是大語言模型 (LLM)。"
    print(f"提示詞: {prompt}\n回應: ")
    
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")

# ---------------------------------------------------------
# 2. 結構化輸出 (Structured Output)
# ---------------------------------------------------------
def demo_structured_output():
    print("-" * 50)
    print("【2. 結構化輸出 Structured Output 示範】")
    print("-" * 50)
    user_review = "這間餐廳的牛肉麵湯頭濃郁超好吃，但出餐速度慢了點，等了快半小時。"
    print(f"待分析評論: {user_review}\n")

    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["正向", "負向", "中立"]},
            "summary": {"type": "string"},
            "score": {"type": "integer"}
        },
        "required": ["sentiment", "summary", "score"]
    }

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': '你是一個專業的文本分析專家，請分析使用者評論並嚴格輸出指定 JSON 格式。'},
            {'role': 'user', 'content': user_review}
        ],
        format=schema,
        stream=False
    )
    
    print("解析出的 JSON 結果:")
    print(response['message']['content'])

# ---------------------------------------------------------
# 3. 工具呼叫 (Tool Calling)
# ---------------------------------------------------------
def add_numbers(a: int, b: int) -> int:
    """計算兩個整數相加"""
    return a + b

def demo_tool_calling():
    print("-" * 50)
    print("【3. 工具呼叫 Tool Calling 示範】")
    print("-" * 50)
    prompt = "請幫我計算 3527 加 4819 等於多少？"
    print(f"使用者問題: {prompt}")

    tools = [{
        'type': 'function',
        'function': {
            'name': 'add_numbers',
            'description': '計算兩個整數相加',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'integer', 'description': '第一個數'},
                    'b': {'type': 'integer', 'description': '第二個數'}
                },
                'required': ['a', 'b']
            }
        }
    }]

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        tools=tools
    )

    tool_calls = response['message'].get('tool_calls')
    if tool_calls:
        for tool in tool_calls:
            if tool['function']['name'] == 'add_numbers':
                args = tool['function']['arguments']
                result = add_numbers(args['a'], args['b'])
                print(f"自動辨識並呼叫函式 `add_numbers({args['a']}, {args['b']})`")
                print(f"計算結果: {result}")
    else:
        print("未觸發工具呼叫，回應:", response['message']['content'])

# ---------------------------------------------------------
# 4. Embedding 向量計算 (Embedding)
# ---------------------------------------------------------
def demo_embedding():
    print("-" * 50)
    print("【4. Embedding 嵌入向量示範】")
    print("-" * 50)
    text = "機器學習是人工智慧的一個分支"
    print(f"輸入文字: '{text}'")
    
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    embedding = response['embedding']
    print(f"向量維度長度: {len(embedding)}")
    print(f"向量前 5 個數值示範: {embedding[:5]}...")

# ---------------------------------------------------------
# 主執行入口
# ---------------------------------------------------------
def main():
    demo_streaming()
    demo_structured_output()
    demo_tool_calling()
    demo_embedding()
    print("-" * 50)
    print("程式執行完畢")

if __name__ == '__main__':
    main()