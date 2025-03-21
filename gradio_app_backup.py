# Standard library imports
import os
import tempfile
from pathlib import Path

# Third-party imports
import gradio as gr
import numpy as np
from PIL import Image

# Local imports
from phase_2.run_model import ImprovedImageVariationGenerator

# Initialize the generator
generator = ImprovedImageVariationGenerator()

def analyze_on_upload(image):
    """Analyze uploaded image and provide parameter suggestions."""
    if image is None:
        return {}, ""
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        input_path = temp_file.name
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(input_path)
        else:
            image.save(input_path)

    try:
        # Use the enhanced analyzer from phase-2
        analysis_results = generator.analyze_image(input_path)
        complexity_analysis = analysis_results['grouped_chars'].get('complexity', {})
        suggested_params = generator._adjust_generation_params(
            complexity_analysis, 
            analysis_results['geo_properties']
        )
        
        # Create detailed analysis report
        analysis_report = f"""
        Pattern Analysis Results:
        - Geometric Pattern: {'Yes' if analysis_results['is_geometric'] else 'No'}
        - Abstract Pattern: {'Yes' if analysis_results['is_abstract'] else 'No'}
        - Pattern Density: {complexity_analysis.get('pattern_density', 0):.2f}
        - Curve Complexity: {complexity_analysis.get('curve_complexity', 0):.2f}
        - Layer Count: {complexity_analysis.get('layer_count', 1)}
        
        Suggested Parameters:
        - Strength: {suggested_params['strength']:.2f}
        - Guidance Scale: {suggested_params['guidance_scale']:.2f}
        - Inference Steps: {suggested_params['num_inference_steps']}
        """
        
        return suggested_params, analysis_report
    
    except Exception as e:
        return {}, f"Error in analysis: {str(e)}"
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

def generate_single_variation(image, params, variation_index, use_suggested=True, user_prompt=""):
    """Generate a single variation with custom parameters and prompt."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            input_path = temp_file.name
            if isinstance(image, np.ndarray):
                Image.fromarray(image).save(input_path)
            else:
                image.save(input_path)

            # Extract parameters based on whether using suggested or custom
            strength = None if use_suggested else params.get('strength')
            guidance_scale = None if use_suggested else params.get('guidance_scale')
            num_inference_steps = None if use_suggested else params.get('num_inference_steps')
            
            variation = generator.generate_variations(
                input_path,
                output_dir=str(Path(tempfile.gettempdir())),
                num_variations=1,
                color_variation_frequency=0.6,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                user_prompt=user_prompt
            )[0]
            
            variation_array = np.array(variation)
            return variation_array, f"Variation {variation_index + 1}"
            
    except Exception as e:
        print(f"Error generating variation {variation_index}: {str(e)}")
        return None
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

# ...existing code...

def create_interface():
    with gr.Blocks(css="""
        .spinner {
            width: 40px;
            height: 40px;
            margin: 10px auto;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .progress-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 10px;
        }
    """) as demo:
        gr.Markdown("# Pattern Variation Generator")
        
        # Store parameters in state
        suggested_params = gr.State({})
        current_params = gr.State({
            'strength': 0.61,
            'guidance_scale': 6.5,
            'num_inference_steps': 25
        })
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Pattern Image", type="pil")
                analysis_text = gr.Textbox(label="Analysis Results", lines=8, interactive=False)
                user_prompt = gr.Textbox(label="Additional Prompt (optional)", lines=2)
            
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Generated Variations",
                    show_label=True,
                    columns=3,
                    rows=2,
                    height=500,
                    object_fit="contain",
                    preview=True
                )
            use_suggested = gr.Checkbox(label="Use Suggested Parameters", value=True)
            
        with gr.Row():
            with gr.Column():
                strength_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.61, step=0.05,
                    label="Strength (Higher values create more varied results)"
                )
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=20.0, value=6.5, step=0.5,
                    label="Guidance Scale"
                )
                steps_slider = gr.Slider(
                    minimum=5, maximum=50, value=25, step=5,
                    label="Inference Steps"
                )
                variations_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Number of Variations"
                )
        
        with gr.Row():
            generate_btn = gr.Button("Generate Variations", variant="primary")
            cancel_btn = gr.Button("Cancel Generation")
        
        # Define loading icon
        loading_icon = gr.HTML("<div class='spinner'></div>")
        progress = gr.Textbox(label="Progress", interactive=False)
        
        # Event handlers
        def on_upload(image, current):
            if image is None:
                return {}, current, "", gr.update(), gr.update(), gr.update()
            
            suggested, analysis = analyze_on_upload(image)
            current.update(suggested)
            
            return (
                suggested,
                current,
                analysis,
                gr.update(value=suggested.get('strength', 0.61)),
                gr.update(value=suggested.get('guidance_scale', 6.5)),
                gr.update(value=suggested.get('num_inference_steps', 25))
            )
        
        input_image.change(
            on_upload,
            inputs=[input_image, current_params],
            outputs=[
                suggested_params, current_params, analysis_text,
                strength_slider, guidance_slider, steps_slider
            ]
        )
        
        # Parameter update handlers
        def update_current_params(strength, guidance, steps, current):
            current.update({
                'strength': strength,
                'guidance_scale': guidance,
                'num_inference_steps': steps
            })
            return current

        for slider in [strength_slider, guidance_slider, steps_slider]:
            slider.change(
                update_current_params,
                inputs=[strength_slider, guidance_slider, steps_slider, current_params],
                outputs=[current_params]
            )
        
        # Handle checkbox changes
        def update_sliders(use_suggested, suggested, current):
            params = suggested if use_suggested else current
            return {
                strength_slider: gr.update(
                    value=params.get('strength', 0.61),
                    interactive=not use_suggested
                ),
                guidance_slider: gr.update(
                    value=params.get('guidance_scale', 6.5),
                    interactive=not use_suggested
                ),
                steps_slider: gr.update(
                    value=params.get('num_inference_steps', 25),
                    interactive=not use_suggested
                )
            }
        
        use_suggested.change(
            update_sliders,
            inputs=[use_suggested, suggested_params, current_params],
            outputs=[strength_slider, guidance_slider, steps_slider]
        )
        
        # Generation with progress updates
        def generate_with_progress(
            image, use_suggested, suggested_params, current_params,
            strength, guidance, steps, num_variations, user_prompt
        ):
            if image is None:
                return None, gr.update(visible=False), "Please upload an image first"
            
            params = suggested_params if use_suggested else current_params
            variations = []
            
            yield variations, gr.update(visible=True), "Starting generation..."
            
            for i in range(num_variations):
                result = generate_single_variation(
                    image, params, i,
                    use_suggested=use_suggested,
                    user_prompt=user_prompt
                )
                if result is not None:
                    variations.append(result)
                    yield (
                        variations, 
                        gr.update(visible=True if i < num_variations - 1 else False),
                        f"Generated {i + 1}/{num_variations} variations"
                    )
            
            yield variations, gr.update(visible=False), "Generation complete!"
        
        generate_btn.click(
            generate_with_progress,
            inputs=[
                input_image, use_suggested, suggested_params, current_params,
                strength_slider, guidance_slider, steps_slider,
                variations_slider, user_prompt
            ],
            outputs=[gallery, loading_icon, progress]
        )
        
        # Cancel button logic
        def cancel_generation():
            # Implement cancellation logic here if needed
            return "Generation cancelled"
        
        cancel_btn.click(lambda: "Generation cancelled", outputs=[progress])
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch(share=True)
   