import boto3
import json
import re
import time

MODEL_ID = "qwen.qwen3-next-80b-a3b"
REGION = "us-east-1"

SYSTEM_PROMPT = """
You are a financial assistant.

You have access to two tools:

1. get_stock_prices(tickers: list[str])
2. compare_prices(prices: dict[str, float])

Rules:
1. If the user asks for stock prices or a stock comparison, do not answer immediately.
2. First call get_stock_prices.
3. After you receive the stock prices, call compare_prices.
4. After you receive the comparison result, give the final answer.
5. When calling a tool, respond ONLY with valid JSON.
6. Do not output markdown code fences.
7. Do not output explanations around the JSON.

Tool call format:
{"action":"tool_name","arguments":{...}}

If no tool is needed, answer normally.
"""

client = boto3.client("bedrock-runtime", region_name=REGION)


def get_stock_prices(tickers: list[str]) -> dict:
    fake_prices = {
        "AAPL": 185.25,
        "MSFT": 412.80,
        "NVDA": 903.10,
        "AMZN": 178.40,
    }
    return {ticker.upper(): fake_prices.get(ticker.upper(), None) for ticker in tickers}


def compare_prices(prices: dict[str, float]) -> dict:
    valid_prices = {k: v for k, v in prices.items() if v is not None}

    if len(valid_prices) < 2:
        return {
            "winner": None,
            "winner_price": None,
            "loser": None,
            "loser_price": None,
            "difference": None,
            "message": "Need at least two valid prices to compare."
        }

    sorted_prices = sorted(valid_prices.items(), key=lambda x: x[1], reverse=True)
    winner, winner_price = sorted_prices[0]
    loser, loser_price = sorted_prices[-1]

    return {
        "winner": winner,
        "winner_price": winner_price,
        "loser": loser,
        "loser_price": loser_price,
        "difference": round(winner_price - loser_price, 2),
        "message": f"{winner} is higher priced than {loser}."
    }


def print_metrics(response, label):
    usage = response.get("usage", {})
    metrics = response.get("metrics", {})

    print(f"\n--- {label} Metrics ---")
    print(f"Input tokens:  {usage.get('inputTokens')}")
    print(f"Output tokens: {usage.get('outputTokens')}")
    print(f"Total tokens:  {usage.get('totalTokens')}")
    print(f"Latency (ms):  {metrics.get('latencyMs')}")


def parse_tool_call(text: str) -> dict:
    # Try JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback 1: get_stock_prices(["AAPL", "MSFT"])
    match = re.search(r'get_stock_prices\(\[(.*?)\]\)', text)
    if match:
        tickers = [t.strip().strip('"').strip("'") for t in match.group(1).split(",")]
        return {
            "action": "get_stock_prices",
            "arguments": {"tickers": tickers}
        }

    raise ValueError(f"Could not parse tool call: {text}")


user_request = input("Ask your agent: ")

messages = [
    {
        "role": "user",
        "content": [
            {"text": SYSTEM_PROMPT + "\n\nUser request: " + user_request}
        ],
    }
]

max_steps = 5

for step in range(1, max_steps + 1):
    start = time.time()
    response = client.converse(
        modelId=MODEL_ID,
        messages=messages
    )
    end = time.time()

    model_text = response["output"]["message"]["content"][0]["text"]

    print(f"\n=== Agent Step {step} ===")
    print(f"\n--- Local latency --- {round((end - start) * 1000)} ms")
    print_metrics(response, f"Step {step}")

    print("\n--- Model output ---\n")
    print(model_text)

    # Try to parse a tool call
    try:
        parsed = parse_tool_call(model_text)
    except Exception:
        print("\n--- Final answer ---\n")
        print(model_text)
        break

    action = parsed["action"]
    arguments = parsed["arguments"]

    print("\n--- Parsed tool call ---\n")
    print(json.dumps(parsed, indent=2))

    # Execute the requested tool
    if action == "get_stock_prices":
        tool_result = get_stock_prices(**arguments)
    elif action == "compare_prices":
        tool_result = compare_prices(**arguments)
    else:
        raise ValueError(f"Unknown tool requested: {action}")

    print("\n--- Tool result ---\n")
    print(json.dumps(tool_result, indent=2))

    # Feed tool call + tool result back into conversation
    messages.append({
        "role": "assistant",
        "content": [{"text": model_text}]
    })

    messages.append({
        "role": "user",
        "content": [{"text": f"Tool result: {json.dumps(tool_result)}"}]
    })
else:
    print("\nReached max_steps without a final answer.")