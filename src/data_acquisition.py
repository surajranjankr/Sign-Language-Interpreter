import fiftyone as fo
import fiftyone.utils.huggingface as fouh
from collections import Counter
import os

# 1. Load the full WLASL index
print("Scanning WLASL Library...")
dataset = fouh.load_from_hub("Voxel51/WLASL")

# 2. Identify Top 20 words by density
all_labels = [s.gloss.label for s in dataset]
top_20 = [word for word, count in Counter(all_labels).most_common(20)]

# 3. Filter for only these 20 words
view = dataset.match(fo.ViewField("gloss.label").is_in(top_20))

# 4. Export videos to our DVC-tracked folder
view.export(
    export_dir="./data/raw_videos",
    dataset_type=fo.types.FiftyOneDataset,
    export_media=True
)
print(f"✅ Found {len(view)} total videos for: {top_20}")