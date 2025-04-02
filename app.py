import spaces
import gradio as gr
from PIL import Image
from transparent_background import Remover
import numpy as np

# Initialize the model globally
remover = Remover(jit=False)

@spaces.GPU
def process_image(input_image, output_type):
    global remover
    
    if output_type == "Mask only":
        # Process the image and get only the mask
        output = remover.process(input_image, type='map')
        if isinstance(output, Image.Image):
            # If output is already a PIL Image, convert to grayscale
            mask = output.convert('L')
        else:
            # If output is a numpy array, convert to PIL Image
            mask = Image.fromarray((output * 255).astype(np.uint8), mode='L')
        return mask
    else:
        # Process the image and return the RGBA result
        output = remover.process(input_image, type='rgba')
        return output
        
description = """<h1 align="center">InSPyReNet Background Remover</h1>
<p><center>
<a href="https://github.com/plemeri/InSPyReNet" target="_blank">[Github]</a>
</center></p>
"""

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Input Image", height=512),
        gr.Radio(["Default", "Mask only"], label="Output Type", value="Default")
    ],
    outputs=gr.Image(type="pil", label="Output Image", height=512),
    description=description,
    theme='bethecloud/storj_theme',
    examples=[
        ["1.png", "Default"],
        ["2.png", "Default"],
        ["3.jfif", "Default"],
        ["4.webp", "Default"]
    ],
    cache_examples=True
)

if __name__ == "__main__":
    iface.launch(share=True)
