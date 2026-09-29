# Dataset

The project uses **[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)**, identified in the original notebook as `Dingdong-Inc/FreshRetailNet-50K` on Hugging Face. The notebook's saved output reports 4,500,000 training rows and 350,000 evaluation rows across approximately 50,000 product–store series. Preserve the provided chronological splits.

## Download

Install the optional dataset loader with `python -m pip install datasets`, then run the following from the repository root:

```python
from pathlib import Path
from datasets import load_dataset

output = Path("data")
output.mkdir(exist_ok=True)
dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
for split in ("train", "eval"):
    dataset[split].to_parquet(str(output / f"{split}.parquet"))
```

Existing `data/train.parquet` and `data/eval.parquet` files can be used directly.

## Fields used by the standalone pipeline

| Raw field | Purpose |
| --- | --- |
| `product_id`, `store_id` | Series identifiers |
| `dt` | Daily date |
| `sale_amount` | Observed sales target, internally called `units_ordered` |
| `first_category_id`, `second_category_id` | Product categories |
| `city_id` | Location |
| `discount` | Discount value; values below 1 generate `is_promotion=1` |
| `holiday_flag`, `activity_flag` | Calendar and activity indicators |
| `avg_temperature`, `avg_humidity`, `precpt`, `avg_wind_level` | Weather covariates |
