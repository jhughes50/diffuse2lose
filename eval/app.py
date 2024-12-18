import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pipeline():
    try:
        logger.info("Starting pipeline loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Using dtype: {dtype}")
        
        logger.info("Loading base pipeline...")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=dtype
        )
        pipeline = pipeline.to(device)
        logger.info("Base pipeline loaded successfully")
        
        unet_folder = "/mnt/extra-dtc/diffusion/unet_epoch_10"
        logger.info(f"Loading UNet from: {unet_folder}")
        unet = UNet2DConditionModel.from_pretrained(
            unet_folder,
            torch_dtype=dtype,
            local_files_only=True,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
        unet = unet.to(device)
        logger.info("UNet loaded successfully")
        
        pipeline.unet = unet
        logger.info("Pipeline setup completed")
        return pipeline
    
    except Exception as e:
        logger.error(f"Error in load_pipeline: {str(e)}")
        raise

pipeline = load_pipeline()  # Load the pipeline once at the start

def inpaint(prompt, input_image, mask_image):
    try:
        logger.info("Starting inpainting process...")
        logger.info(f"Received prompt: {prompt}")
        
        # Input validation
        if input_image is None:
            logger.error("Input image is None")
            return "Please provide an input image", None
        if mask_image is None:
            logger.error("Mask image is None")
            return "Please provide a mask image", None
            
        logger.info("Converting images to correct format...")
        input_image = input_image.convert("RGB")
        mask_image = mask_image.convert("L")
        
        logger.info(f"Input image size: {input_image.size}")
        logger.info(f"Mask image size: {mask_image.size}")
        
        # Ensure images are the right size (multiple of 8)
        width, height = input_image.size
        if width % 8 != 0 or height % 8 != 0:
            logger.info("Resizing images to be multiple of 8...")
            new_width = (width // 8) * 8
            new_height = (height // 8) * 8
            input_image = input_image.resize((new_width, new_height))
            mask_image = mask_image.resize((new_width, new_height))
        
        logger.info("Starting inference...")
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                image=input_image,
                mask_image=mask_image
            ).images[0]
        
        logger.info("Inference completed successfully")
        return None, result
    
    except Exception as e:
        logger.error(f"Error in inpaint function: {str(e)}")
        return f"Error during inpainting: {str(e)}", None

def app():
    logger.info("Starting Gradio app...")
    with gr.Blocks() as demo:
        gr.Markdown("## Stable Diffusion Inpainting")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Input Image", type="pil")
                mask_image = gr.Image(label="Upload Mask", type="pil")


            with gr.Column():
                prompt = gr.Textbox(label="Prompt",  value="Occupancy top down view map, white is free space, black is occupied")

                generate_button = gr.Button("Generate Inpainting")
                output_image = gr.Image(label="Output Image")
                
                # Add a textbox for error messages
                error_box = gr.Textbox(label="Status/Error Messages", interactive=False)

        generate_button.click(
            inpaint,
            inputs=[prompt, input_image, mask_image],
            outputs=[error_box, output_image]
        )

    logger.info("Launching Gradio interface...")
    demo.launch(share=True)

if __name__ == "__main__":
    logger.info("Starting application...")
    app()
