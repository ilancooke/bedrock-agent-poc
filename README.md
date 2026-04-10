# Bedrock Agent POC

Simple agent built using AWS Bedrock Converse API.

## Features
- Tool calling (get_stock_prices, compare_prices)
- Multi-step agent loop
- Token + latency tracking
- CLI input


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run

```bash
source .venv/bin/activate
python test_converse.py
