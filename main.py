from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import os
import random
from hypermetrices import modeul
from tree import main

image_path = "./image.1.png"
out=modeul (image_path)
main(out)