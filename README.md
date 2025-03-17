# Pattern Variation Generator

An AI-powered tool for generating creative variations of pattern designs while preserving key structural and stylistic elements.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RyOmmPLFvlXmBsiyeIDbEMWjtao1lNh6?usp=sharing)

## Features

- Geometric pattern detection and preservation
- Smart color scheme variation
- Abstract pattern handling
- Opacity and transparency support
- Multi-scale pattern analysis
- Enhanced pattern structure preservation

## Requirements

- Python 3.8+
- CUDA-enabled GPU (Recommended)
- 8GB+ GPU Memory
- Linux/Windows/MacOS

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CarpetsVariations
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Usage

1. Place your input pattern images in the `Input` folder
2. Run the web interface:
```bash
python gradio_app.py
```
Or use the pattern generation script directly:
```bash
python phase_2/run_model.py
```
3. Find generated variations in the `Output` folder
4. View grid comparisons in the `Grids` folder

## Project Structure

```
CarpetsVariations/
├── Input/          # Input pattern images
├── Output/         # Generated variations
├── Grids/         # Comparison grids
├── phase-1/       # Initial implementation
├── phase-2/       # Enhanced implementation
├── gradio_app.py  # Web interface
└── requirements.txt
```

## Parameters

- `num_variations`: Number of variations to generate (default: 5)
- `color_variation_frequency`: Frequency of color scheme changes (default: 0.6)
- `strength`: Generation strength (default: dynamic based on pattern)
- `guidance_scale`: Model guidance scale (default: dynamic based on pattern)

## Notes

- Supports PNG and JPEG images
- Preserves transparency in PNG files
- Automatically analyzes pattern characteristics
- Maintains pattern coherence while introducing creative variations

## Web Interface

The Gradio web interface provides:
- Interactive parameter adjustment
- Real-time pattern analysis
- Live generation preview
- Custom prompt input
- Progress tracking
- Suggested parameter optimization

## Web Interface Features

The Gradio interface provides an easy way to:
- Upload and analyze patterns
- Adjust generation parameters with real-time feedback
- Preview variations in a gallery view
- Add custom prompts to guide generation
- Track generation progress
- Use AI-suggested parameters or customize your own
- Generate 1-10 variations per run
- Save results automatically

## Jupyter Notebook

You can explore and run the code interactively through our Jupyter notebook. We provide two options:

1. **Google Colab**: Click the badge below to open the notebook in Google Colab:
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RyOmmPLFvlXmBsiyeIDbEMWjtao1lNh6?usp=sharing)

2. **Local Jupyter**: You can also run the notebook locally by opening the `carpets_variations.ipynb` file in Jupyter Lab or Jupyter Notebook.

The notebook contains interactive examples and visualizations of the carpet variations.
