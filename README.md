# Photo AI ⚽📸

Automated soccer photo processing toolkit. Complete 4-stage workflow to organize your game photos by player with only the best quality images.

## Features

⚽ **Complete Soccer Photo Processing Workflow:**
1. **Remove Bad-Quality Photos** - Filter out blurry, out-of-focus images
2. **Remove Duplicate Photos** - Keep only the best version of similar shots  
3. **Group Photos by Player** - Automatically organize by detected players
4. **Select Best Photos per Player** - Choose 1-2 highest quality photos per player

## Quick Start

### Installation

```bash
git clone <repository-url>
cd photo-ai
pip install -e .
```

### 🖥️ **Desktop GUI (Recommended)**

```bash
photo-ai-gui
```

### ⚽ **Soccer Photo Processing**

**Setup:**
1. Create a `players/` folder in your game photos directory
2. Add one reference photo per player (e.g., `Messi.jpg`, `Ronaldo.jpg`)
3. Run the complete workflow

**Command Line:**
```bash
# Complete 4-stage soccer photo processing  
photo-ai soccer game_photos/

# With custom options
photo-ai soccer game_photos/ --players-dir custom_players/ --max-photos 3
```

**Expected Output:**
```
game_photos/output/
├── Messi/
│   ├── best_photo_1.jpg
│   └── best_photo_2.jpg  
├── Ronaldo/
│   ├── best_photo_1.jpg
│   └── best_photo_2.jpg
└── unknown/
    └── unrecognized_photos.jpg
```

## Python API

```python
from photo_ai import PhotoProcessor

# Complete soccer photo processing
processor = PhotoProcessor()
result = processor.process_soccer_photos_complete(
    input_dir="game_photos/",
    players_dir="game_photos/players/", 
    output_dir="game_photos/output/"
)

print(f"✅ Processed {result['final_summary']['input_photos']} photos")
print(f"🏆 Selected {result['final_summary']['final_selected']} best photos")
print(f"👥 Found {result['final_summary']['players_found']} players")
```

## Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: Photos are copied, originals preserved
- **Supported Formats**: JPG, JPEG, PNG, WebP, BMP, TIFF

## License

This project is licensed under the MIT License.

---

**Photo AI** ⚽ - Automated soccer photo organization made simple!
