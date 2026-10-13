"""Relabel text on an existing plot image.

This utility is kept separate from model training. Usage:
    python relabel_plot.py input.png output.png --title "New title"
Requires Pillow and Matplotlib.
"""
import argparse
from PIL import Image, ImageDraw, ImageFont

def relabel(input_path, output_path, title=None, xlabel=None, ylabel=None):
    image = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()
    # This is a general-purpose simple relabel utility; unlike the original
    # pixel-coordinate notebook function, it does not attempt to reconstruct
    # pixels obscured by the old legend.
    if title:
        draw.rectangle((int(width*.2), 0, int(width*.8), int(height*.08)), fill="white")
        draw.text((width//2, int(height*.02)), title, fill="black", anchor="mm", font=font)
    if xlabel:
        draw.rectangle((int(width*.4), int(height*.93), int(width*.6), height), fill="white")
        draw.text((width//2, int(height*.96)), xlabel, fill="black", anchor="mm", font=font)
    if ylabel:
        draw.rectangle((0, int(height*.2), int(width*.04), int(height*.8)), fill="white")
        draw.text((int(width*.02), height//2), ylabel, fill="black", anchor="mm", font=font)
    image.save(output_path)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--title")
    parser.add_argument("--xlabel")
    parser.add_argument("--ylabel")
    args = parser.parse_args()
    relabel(args.input, args.output, args.title, args.xlabel, args.ylabel)

if __name__ == "__main__":
    main()
