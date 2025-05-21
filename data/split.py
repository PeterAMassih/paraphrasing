import json
import random
import os

with open("evaluation/paraphrase_dataset.json", "r") as f:
    dataset = json.load(f)

all_items = []
for category, items in dataset.items():
    for item in items:
        item["category"] = category
        all_items.append(item)

random.seed(42)
random.shuffle(all_items)

split_index = int(len(all_items) * 0.9)
train_items = all_items[:split_index]
test_items = all_items[split_index:]

train_data = {}
test_data = {}

for item in train_items:
    category = item.pop("category")
    if category not in train_data:
        train_data[category] = []
    train_data[category].append(item)

for item in test_items:
    category = item.pop("category")
    if category not in test_data:
        test_data[category] = []
    test_data[category].append(item)

os.makedirs("evaluation/splits", exist_ok=True)

with open("evaluation/splits/paraphrase_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("evaluation/splits/paraphrase_test.json", "w") as f:
    json.dump(test_data, f, indent=2)

total_train = sum(len(items) for items in train_data.values())
total_test = sum(len(items) for items in test_data.values())
total = total_train + total_test

print(f"Original dataset: {total} items")
print(f"Train split: {total_train} items ({total_train/total*100:.1f}%)")
print(f"Test split: {total_test} items ({total_test/total*100:.1f}%)")