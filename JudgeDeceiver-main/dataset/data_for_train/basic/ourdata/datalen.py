import json
with open("arena_hard_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))   # 条数
