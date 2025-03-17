import torch
import clip
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import StableDiffusionXLImg2ImgPipeline, LCMScheduler
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import numpy as np
import cv2
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
import random
from sklearn.cluster import KMeans
import colorsys
from scipy.signal import find_peaks
import pandas as pd
import fnmatch

def create_variation_grid(original_image, variations, filename, grid_dir):
    """Create and save a 2x3 grid of the original image and its variations."""
    # Create figure with original aspect ratio consideration
    aspect_ratio = original_image.size[0] / original_image.size[1]
    plt.figure(figsize=(20, 15))

    # Plot original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title('Original', fontsize=12)
    plt.axis('off')

    # Plot variations
    for idx, var_image in enumerate(variations):
        plt.subplot(2, 3, idx + 2)
        plt.imshow(var_image)
        plt.title(f'Variation {chr(97+idx)}', fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    grid_path = grid_dir / f"{filename}_grid.png"
    plt.savefig(grid_path, bbox_inches='tight', dpi=300)
    plt.close()

    return grid_path

class GeometricVariationHandler:
    def __init__(self):
        self.dimension_modifiers = [
            "with varied circle sizes from 0.7x to 1.3x original size",
            "with dynamic distribution of circle sizes",
            "with alternating large and small circles",
            "with gradual size progression of circles",
            "featuring multi-scale circular elements"
        ]

        self.arrangement_modifiers = [
            "with varied spacing between circles",
            "with dynamic circular arrangement",
            "featuring shifted circle positions",
            "with alternating circle density",
            "with varied circle overlap"
        ]

    def _is_circular_pattern(self, image: np.ndarray) -> bool:
        """
        Detect if the pattern is primarily composed of circles.

        Args:
            image: numpy array of the image in BGR format
        Returns:
            bool: True if the pattern is primarily circular
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze each contour
            total_shapes = 0
            circular_shapes = 0

            for contour in contours:
                # Filter out very small contours
                if cv2.contourArea(contour) < 100:
                    continue

                total_shapes += 1

                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)

                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Check if the shape is approximately circular
                if 0.75 <= circularity <= 1.25:
                    circular_shapes += 1

            # Calculate the proportion of circular shapes
            if total_shapes == 0:
                return False

            circular_ratio = circular_shapes / total_shapes
            return circular_ratio > 0.6  # If more than 60% shapes are circular

        except Exception as e:
            print(f"Error in circular pattern detection: {e}")
            return False

    def get_geometric_parameters(self, variation_index: int, total_variations: int, image) -> dict:
        """Generate variation-specific parameters for geometric patterns."""
        # Check if the pattern is circular
        is_circular = self._is_circular_pattern(np.array(image))

        # Use higher strength specifically for circular patterns
        base_strength = 0.7 if is_circular else 0.6

        # Vary strength more aggressively for circles
        strength_variation = 0.2 * (variation_index / total_variations)
        strength = min(8.0, base_strength + strength_variation)

        # Adjust parameters based on pattern type
        if is_circular:
            return {
                'strength': strength,
                'guidance_scale':8.0,  # Higher guidance for geometric precision
                'num_inference_steps': 30  # More steps for detail
            }
        else:
            return {
                'strength': 0.6,
                'guidance_scale': 7.0,
                'num_inference_steps': 25
            }

    def enhance_geometric_prompt(self, base_prompt: str, variation_index: int, image) -> str:
        """Enhance the prompt specifically for geometric patterns."""
        is_circular = self._is_circular_pattern(np.array(image))

        if is_circular:
            # Select modifiers based on variation index
            dimension_modifier = self.dimension_modifiers[variation_index % len(self.dimension_modifiers)]
            arrangement_modifier = self.arrangement_modifiers[variation_index % len(self.arrangement_modifiers)]
            new_prompt = ""
            if variation_index > 3:  # For later variations
                  new_prompt = ",dramatic structural variation"

            circle_specific_instructions = [
                "vary circle distribution while maintaining pattern",
                "transform circle scale relationships"
            ]

            enhanced_prompt = (
                f"{base_prompt}, {new_prompt},{dimension_modifier}, {arrangement_modifier}, "
                f"{circle_specific_instructions[variation_index % len(circle_specific_instructions)]}, "
                "emphasizing geometric precision and circular form"
            )
            return enhanced_prompt
        else:
            # For non-circular geometric patterns, use original enhancement
            dimension_modifier = (
                "with varied shape scales ranging from 0.5x to 1.5x original size",
                "with dynamic size distribution of elements",
                "with alternating large and small geometric elements",
                "with gradual size progression of shapes",
                "featuring multi-scale geometric elements"
            )[variation_index % 5]

            arrangement_modifier = (
                "with varied spacing between elements",
                "with dynamic spatial arrangement",
                "featuring shifted geometric alignments",
                "with alternating density of elements",
                "with varied pattern density"
            )[variation_index % 5]

            enhanced_prompt = f"{base_prompt}, {dimension_modifier}, {arrangement_modifier}, maintaining geometric precision"
            return enhanced_prompt

class GeometricPatternHandler:
    def __init__(self):
        self.shape_thresholds = {
            'circularity': 0.85,
            'rectangularity': 0.95,
            'triangularity': 0.75
        }

    def detect_geometric_pattern(self, image: Image.Image) -> Tuple[bool, Dict]:
        """
        Detect if an image is purely geometric and analyze its properties.
        Returns (is_geometric, properties_dict)
        """
        # Convert PIL to numpy array
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Enhanced edge detection for geometric shapes
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        shape_metrics = []
        total_area = gray.shape[0] * gray.shape[1]
        geometric_area = 0

        for contour in contours:
            if cv2.contourArea(contour) < total_area * 0.001:  # Filter out noise
                continue

            # Calculate shape metrics
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter == 0:
                continue

            geometric_area += area

            # Calculate shape characteristics
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            rect = cv2.minAreaRect(contour)
            rect_area = rect[1][0] * rect[1][1]
            rectangularity = area / rect_area if rect_area > 0 else 0

            # Approximate the contour for polygon detection
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            vertices = len(approx)

            shape_metrics.append({
                'circularity': circularity,
                'rectangularity': rectangularity,
                'vertices': vertices,
                'area': area
            })

        # Analyze overall geometric properties
        is_geometric = False
        properties = {
            'coverage_ratio': geometric_area / total_area,
            'shape_count': len(shape_metrics),
            'regular_shapes': 0,
            'dominant_shape': 'mixed'
        }

        if shape_metrics:
            # Count regular shapes
            regular_shapes = sum(1 for m in shape_metrics if
                               (m['circularity'] > self.shape_thresholds['circularity'] or
                                m['rectangularity'] > self.shape_thresholds['rectangularity'] or
                                m['vertices'] in [3, 4, 6, 8]))

            properties['regular_shapes'] = regular_shapes
            properties['regularity_ratio'] = regular_shapes / len(shape_metrics)
            properties['pattern_type'] = 'squares' if rectangularity > 0.9 else 'geometric'
            #properties['regularity_score'] = regularity_score
            #properties['size_distribution'] = self._analyze_size_distribution(contours)

            # Determine if the pattern is purely geometric
            is_geometric = (properties['regularity_ratio'] > 0.85 and
                          properties['coverage_ratio'] > 0.4)

            # Identify dominant shape type
            circles = sum(1 for m in shape_metrics if m['circularity'] > self.shape_thresholds['circularity'])
            rectangles = sum(1 for m in shape_metrics if m['rectangularity'] > self.shape_thresholds['rectangularity'])
            polygons = sum(1 for m in shape_metrics if m['vertices'] in [3, 6, 8])

            max_count = max(circles, rectangles, polygons)
            if max_count == circles:
                properties['dominant_shape'] = 'circular'
            elif max_count == rectangles:
                properties['dominant_shape'] = 'rectangular'
            elif max_count == polygons:
                properties['dominant_shape'] = 'polygonal'

        return is_geometric, properties


class ColorVariationHandler:
    def __init__(self):
        # Color scheme variations
        self.color_schemes = [
            "warm earth tones with aesthetic accents",
            "cool blues with orange highlights",
            "monochromatic dark with light accents",
            "rich terracotta with dark outlines",
            "deep burgundy with copper elements",
            "navy blue with silver highlights",
            "charcoal with beautiful geometric elements",
            "olive green with bronze details",
            "deep purple with metallic accents",
            "muted pastels with a soft, faded effect",
            "black and white with bold, sharp contrasts",
            "seafoam green with coral highlights",
            "dusty rose with light gray undertones",
            "sunset hues with gradient blending",
            "rich mahogany with deep gold accents",
            "soft taupe with crisp white elements",
            "midnight blue with subtle metallic sheen",
            "warm amber with dark walnut accents"
        ]

        # Pattern enhancement modifiers
        self.pattern_modifiers = [
            "with varied line weights",
            "with alternating pattern density",
            "featuring layered geometric elements",
            "with dynamic scale variation",
            "emphasizing pattern rhythm",
            "with balanced negative space",
            "featuring intricate detail work",
            "with precise geometric alignment",
            "emphasizing pattern flow",
            "with careful element spacing"
        ]


    def enhance_variation_prompt(self, base_prompt: str, variation_index: int) -> str:
        """Create specific prompt for each variation."""
        # Select color scheme and pattern modifier for this variation
        color_scheme = self.color_schemes[variation_index % len(self.color_schemes)]
        pattern_modifier = self.pattern_modifiers[variation_index % len(self.pattern_modifiers)]

        # Build enhanced prompt
        enhanced_prompt = (
            f"{base_prompt}, {color_scheme}, {pattern_modifier}, "
            "maintaining geometric precision and pattern structure, "
            "professional pattern design, high-quality commercial pattern, "
            "preserving line quality and shape definition"
        )

        return enhanced_prompt

    def get_negative_prompt(self) -> str:
        """Generate negative prompt to maintain pattern quality."""
        return (
            "blurry, distorted, warped, irregular shapes, broken lines, "
            "messy, cluttered, unclear pattern, photographic, realistic, "
            "inconsistent spacing, poor line quality, rough edges, "
            "missing pattern elements, uneven pattern distribution"
        )

class AbstractColorHandler:
    def __init__(self):
        # Color-focused modifiers for abstract patterns
        self.color_modifiers = [
            "fluid color transitions",
            "smooth color gradients",
            "abstract color composition",
            "pure color expression",
            "color field composition"
        ]

        # Abstract composition modifiers
        self.composition_modifiers = [
            "with soft color boundaries",
            "with gentle color blending",
            "with atmospheric color diffusion",
            "with subtle color interactions",
            "with harmonious color flow"
        ]

    def detect_abstract_color_pattern(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect if the image is primarily an abstract color composition.

        Args:
            image: numpy array of the image
        Returns:
            bool: True if abstract color pattern
            Dict: Properties of the color composition
        """
        try:
            # Convert to RGB if not already
            if len(image.shape) == 2:
                return False, {}

            # Calculate edge density
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)  # Lower thresholds
            edge_density = np.mean(edges) / 255.0

            # Calculate color gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            smooth_transitions = np.mean(gradient_magnitude) < 30

            # Check for lack of distinct shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            significant_contours = sum(1 for c in contours if cv2.contourArea(c) > 100)

            # Calculate color statistics
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            color_variance = np.std(hsv[:,:,0])
            saturation_variance = np.std(hsv[:,:,1])

            is_abstract = (
                edge_density < 0.1 and  # Few distinct edges
                smooth_transitions and   # Smooth color transitions
                significant_contours < 5 and  # Few distinct shapes
                color_variance > 10  # Some color variation
            )

            properties = {
                'edge_density': float(edge_density),
                'color_variance': float(color_variance),
                'saturation_variance': float(saturation_variance),
                'smooth_transitions': smooth_transitions,
                'significant_contours': significant_contours
            }

            return is_abstract, properties

        except Exception as e:
            print(f"Error in abstract color detection: {e}")
            return False, {}

    def get_abstract_parameters(self, variation_index: int, properties: Dict) -> Dict:
        """Generate parameters optimized for abstract color patterns."""
        # Use lower strength to preserve color composition
        base_strength = 0.4

        # Adjust strength based on color properties
        color_factor = min(1.2, max(0.8, properties.get('color_variance', 0) / 50))
        strength = base_strength * color_factor

        return {
            'strength': strength,
            'guidance_scale': 4.5,  # Lower guidance for more natural color flow
            'num_inference_steps': 20
        }

    def enhance_abstract_prompt(self, base_prompt: str, variation_index: int, properties: Dict) -> str:
        """Enhance the prompt specifically for abstract color patterns."""
        # Select modifiers based on variation index
        color_modifier = self.color_modifiers[variation_index % len(self.color_modifiers)]
        composition_modifier = self.composition_modifiers[variation_index % len(self.composition_modifiers)]

        # Add specific instructions based on color properties
        color_instructions = []
        if properties.get('smooth_transitions', True):
            color_instructions.append("maintaining smooth color transitions")
        if properties.get('color_variance', 0) > 30:
            color_instructions.append("preserving dynamic color relationships")
        if properties.get('saturation_variance', 0) > 20:
            color_instructions.append("retaining color intensity variation")

        enhanced_prompt = (
            f"{base_prompt}, abstract color composition, {color_modifier}, {composition_modifier}, "
            f"{', '.join(color_instructions)}, "
            "minimalist color field composition, "
            "no recognizable shapes or patterns, "
            "without distinct shapes or patterns, pure color expression"
        )

        return enhanced_prompt

    def get_abstract_negative_prompt(self) -> str:
        """Generate negative prompt for abstract color patterns."""
        return (
            "distinct shapes, hard edges, defined patterns, geometric forms, "
            "floral elements, organic shapes, representational elements, "
            "figurative elements, textures, lines, borders, outlined shapes, "
            "recognizable objects, structured patterns"
        )

class EnhancedImageCharacteristics:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None

        # Enhanced characteristic groups
        self.characteristic_groups = {
            'color': [
                "monochromatic", "colorful", "vibrant", "pastel",
                "warm colors", "cool colors", "high contrast", "low contrast",
                "analogous colors", "complementary colors", "neutral colors",
                "muted colors", "bold colors", "harmonious palette"
            ],
            'style': [
                "abstract", "geometric", "minimal", "symmetric", "organic",
                "flowing", "natural", "floral", "botanical", "sharp",
                "structured", "angular", "architectural", "transparent",
                "decorative", "ornamental", "traditional", "modern",
                "delicate", "intricate", "refined"
            ],
            'pattern': [
                "textured", "detailed", "complex", "layered", "repeating",
                "regular pattern", "irregular pattern", "border pattern",
                "all-over pattern", "motif-based", "symmetrical pattern",
                "directional pattern", "scattered pattern", "central pattern",
                "medallion pattern", "interlocking pattern", "nested elements"
            ],
            'composition': [
                "bold", "subtle", "rhythmic", "chaotic", "balanced",
                "asymmetrical", "repetitive", "random", "centered",
                "edge-to-edge", "bordered", "framed", "continuous",
                "hierarchical", "structured layout"
            ],
            'technique': [
                "precise lines", "hand-drawn", "digital", "etched",
                "printed", "woven", "embroidered", "block-printed",
                "screen-printed", "vector-based", "rasterized",
                "clean edges", "smooth transitions"
            ],
            'ornamental_style': [
                "arabesque", "baroque", "rococo", "art nouveau",
                "celtic", "damascene", "filigree", "scrollwork",
                "mandala-like", "calligraphic", "oriental",
                "victorian", "gothic", "byzantine"
            ],
            'texture_depth': [
                "layered opacity", "translucent layers", "gradient overlay",
                "metallic effect", "embossed pattern", "etched detail",
                "burnished texture", "depth variation", "shadow detail"
            ],
            'pattern_movement': [
                "flowing curves", "spiral motion", "radial flow",
                "interweaving", "overlapping elements", "undulating",
                "serpentine", "circular motion", "meandering"
            ]
        }

    def analyze_opacity_guidance(self, image: Image.Image) -> Dict:
        """Analyze image opacity and calculate guidance scale."""
        try:
            # Convert to RGBA if not already
            image_rgba = image.convert('RGBA')
            alpha_channel = np.array(image_rgba.getchannel('A'))

            # Calculate opacity metrics
            mean_opacity = np.mean(alpha_channel) / 255.0
            opacity_variance = np.var(alpha_channel) / (255.0 ** 2)

            # Calculate dynamic guidance scale
            base_guidance = 7.5
            opacity_factor = 1.0 + (1.0 - mean_opacity) * 2.0
            variance_factor = 1.0 + opacity_variance * 3.0

            # Calculate final guidance scale
            adjusted_guidance = base_guidance * opacity_factor * variance_factor
            final_guidance = np.clip(adjusted_guidance, 5.0, 20.0)

            return {
                'mean_opacity': float(mean_opacity),
                'opacity_variance': float(opacity_variance),
                'recommended_guidance': float(final_guidance)
            }
        except Exception as e:
            print(f"Error in opacity guidance analysis: {e}")
            return {
                'mean_opacity': 1.0,
                'opacity_variance': 0.0,
                'recommended_guidance': 7.5
            }

    def enhance_low_opacity_regions(self, image: Image.Image) -> Image.Image:
        """Enhance regions with low opacity while preserving structure."""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Handle images without alpha channel
            if img_array.shape[-1] < 4:
                return image

            alpha_channel = img_array[:, :, 3]

            # Create mask for low opacity regions
            low_opacity_mask = alpha_channel < 204  # Less than 80% opacity

            # Enhanced array with preserved structure
            enhanced_array = img_array.copy()

            # Process each RGB channel
            for channel in range(3):
                channel_data = enhanced_array[:, :, channel]

                # Apply contrast enhancement to low opacity regions
                channel_data[low_opacity_mask] = np.clip(
                    ((channel_data[low_opacity_mask] - 128) * 1.2 + 128),
                    0,
                    255
                ).astype(np.uint8)

                # Apply edge preservation
                if np.any(low_opacity_mask):
                    kernel = np.ones((3,3), np.uint8)
                    edges = cv2.Canny(channel_data, 100, 200)
                    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                    channel_data[dilated_edges > 0] = img_array[:, :, channel][dilated_edges > 0]

                enhanced_array[:, :, channel] = channel_data

            return Image.fromarray(enhanced_array)

        except Exception as e:
            print(f"Error in opacity enhancement: {e}")
            return image

    def get_recommended_parameters(self, opacity_metrics: Dict) -> Dict:
        """Get recommended processing parameters based on opacity analysis."""
        return {
            'guidance_scale': opacity_metrics['recommended_guidance'],
            'opacity_compensation': max(1.0, 2.0 - opacity_metrics['mean_opacity']),
            'detail_enhancement': min(2.0, 1.0 + opacity_metrics['opacity_variance'] * 4.0),
            'suggested_preprocessing': [
                'enhance_contrast' if opacity_metrics['mean_opacity'] < 0.8 else None,
                'sharpen_edges' if opacity_metrics['opacity_variance'] > 0.2 else None,
                'preserve_structure' if opacity_metrics['mean_opacity'] < 0.9 else None
            ]
        }


    def analyze_floral_patterns(self, image: Image.Image) -> Dict:
        """Enhanced analysis specifically for floral patterns with stripe pattern exclusion."""
        try:
            # Convert to numpy array
            img_array = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # First, check for dominant horizontal lines (stripes)
            def detect_stripes(gray_img):
                # Apply horizontal Sobel operator
                sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate magnitude of gradients
                mag_x = np.abs(sobelx)
                mag_y = np.abs(sobely)

                # Calculate ratio of vertical to horizontal gradients
                total_x = np.sum(mag_x)
                total_y = np.sum(mag_y)

                if total_x > 0:
                    gradient_ratio = total_y / total_x
                    # If horizontal gradients dominate significantly
                    return gradient_ratio < 0.5
                return False

            # Check for regular spacing of horizontal lines
            def check_line_regularity(gray_img):
                # Project to vertical axis to detect horizontal lines
                projection = np.sum(gray_img, axis=1)
                # Calculate standard deviation of distances between peaks
                peaks, _ = find_peaks(projection, distance=5)
                if len(peaks) > 1:
                    peak_distances = np.diff(peaks)
                    std_distance = np.std(peak_distances)
                    mean_distance = np.mean(peak_distances)
                    # Check if lines are regularly spaced
                    return std_distance / mean_distance < 0.3
                return False

            # If stripes are detected, return minimal floral metrics
            if detect_stripes(gray) or check_line_regularity(gray):
                return {
                    'petal_like_curves': 0,
                    'organic_shapes': 0,
                    'symmetry_score': 0,
                    'confidence': 0,
                    'confidence_adjustment': {
                        'base_threshold': 0.25,
                        'boost_factor': 0
                    },
                    'pattern_type': 'stripes'
                }

            # Multi-scale edge detection with even stricter thresholds
            edges_fine = cv2.Canny(gray, 100, 200)  # Further increased thresholds
            edges_medium = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 75, 150)
            edges_coarse = cv2.Canny(cv2.GaussianBlur(gray, (9,9), 0), 50, 100)

            combined_edges = cv2.addWeighted(
                edges_fine, 0.6,  # Increased weight for fine details
                cv2.addWeighted(edges_medium, 0.25, edges_coarse, 0.15, 0),
                0.4, 0
            )

            # Initialize metrics
            floral_metrics = {
                'petal_like_curves': 0,
                'organic_shapes': 0,
                'symmetry_score': 0,
                'confidence': 0,
                'pattern_type': 'unknown'
            }

            # Analyze contours with stricter criteria
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = 0
            min_contour_area = 150  # Increased minimum area

            for contour in contours:
                # Skip elongated shapes (likely stripes)
                x, y, w, h = cv2.boundingRect(contour)
                if w/h > 3 or h/w > 3:  # Skip highly elongated shapes
                    continue

                if len(contour) > 10 and cv2.contourArea(contour) > min_contour_area:  # Increased minimum points
                    try:
                        # Stricter ellipse fitting
                        ellipse = cv2.fitEllipse(contour)
                        eccentricity = np.abs(1 - ellipse[1][0]/ellipse[1][1])

                        # Even narrower eccentricity range
                        if 0.48 <= eccentricity <= 0.62:
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            contour_area = cv2.contourArea(contour)

                            if hull_area > 0:
                                solidity = contour_area / hull_area
                                # Stricter solidity range
                                if 0.8 <= solidity <= 0.85:
                                    perimeter = cv2.arcLength(contour, True)
                                    if perimeter > 0:
                                        circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                                        # Stricter circularity range
                                        if 0.4 <= circularity <= 0.6:
                                            floral_metrics['petal_like_curves'] += 1
                                            valid_contours += 1

                    except:
                        continue

            # Normalize metrics with higher requirements
            total_contours = max(valid_contours, 1)
            floral_metrics['petal_like_curves'] /= total_contours

            # Require multiple valid contours for floral pattern
            if valid_contours < 3:
                floral_metrics['confidence'] = 0
            else:
                # Calculate confidence with stricter threshold
                confidence_score = floral_metrics['petal_like_curves']
                floral_metrics['confidence'] = confidence_score if confidence_score > 0.7 else 0

            floral_metrics['confidence_adjustment'] = {
                'base_threshold': 0.3,  # Further increased base threshold
                'boost_factor': 1.2 if floral_metrics['confidence'] > 0.7 else 1.0
            }

            return floral_metrics

        except Exception as e:
            print(f"Error in floral pattern analysis: {e}")
            return {
                'petal_like_curves': 0,
                'organic_shapes': 0,
                'symmetry_score': 0,
                'confidence': 0,
                'pattern_type': 'error',
                'confidence_adjustment': {'base_threshold': 0.3, 'boost_factor': 1.0}
            }

    def analyze_pattern_complexity(self, image: Image.Image) -> Dict:
      """Analyze advanced pattern characteristics."""
      try:
          img_array = np.array(image.convert('RGB'))

          # Analyze pattern density variation
          density_map = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
          density_variation = np.std(density_map)

          # Analyze curvature characteristics
          edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 50, 150)
          contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

          # Calculate curve metrics
          curve_metrics = []
          for contour in contours:
              if len(contour) > 5:
                  _, radius = cv2.minEnclosingCircle(contour)
                  area = cv2.contourArea(contour)
                  if area > 0:
                      curve_metrics.append(radius / area)

          return {
              'pattern_density': float(density_variation),
              'curve_complexity': float(np.mean(curve_metrics)) if curve_metrics else 0.0,
              'layer_count': self._estimate_layer_count(img_array)
          }
      except Exception as e:
          print(f"Error in pattern complexity analysis: {e}")
          return {'pattern_density': 0.5, 'curve_complexity': 0.5, 'layer_count': 1}


    def analyze_pattern_structure(self, image: Image.Image) -> Dict:
      """Analyze the structural characteristics of the pattern with memory optimization."""
      try:
          print("Converting image to numpy array...")
          # Resize image to a manageable size for analysis
          max_size = 512
          img_resized = image.copy()
          if max(image.size) > max_size:
              ratio = max_size / max(image.size)
              new_size = tuple(int(dim * ratio) for dim in image.size)
              img_resized = image.resize(new_size, Image.Resampling.LANCZOS)

          # Convert to numpy array
          img_array = np.array(img_resized.convert('RGB'))
          print("Converting to grayscale...")
          gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

          print("Performing edge detection...")
          # Use lower threshold values for edge detection to reduce computation
          edges = cv2.Canny(gray, 30, 100)
          edge_density = float(np.mean(edges) / 255.0)

          print("Calculating symmetry...")
          # Calculate symmetry on downsampled edge image for efficiency
          height, width = edges.shape
          left_half = edges[:, :width//2]
          right_half = np.fliplr(edges[:, width//2:])
          symmetry_score = float(1.0 - np.mean(np.abs(left_half - right_half)) / 255.0)

          print("Calculating pattern regularity...")
          # Calculate regularity on a smaller sample
          sample_size = min(1000, edges.size)
          sample_indices = np.random.choice(edges.size, sample_size, replace=False)
          edge_sample = edges.flatten()[sample_indices]
          correlation = np.correlate(edge_sample, edge_sample, mode='full')
          regularity_score = float(np.max(correlation[1:]) / correlation[0] if correlation[0] != 0 else 0)

          result = {
              'symmetry_score': symmetry_score,
              'regularity_score': regularity_score,
              'edge_density': edge_density
          }

          # Clean up
          del img_array, gray, edges, left_half, right_half, edge_sample, correlation

          print("Pattern structure analysis completed successfully")
          return result

      except Exception as e:
          print(f"Error in pattern structure analysis: {e}")
          import traceback
          traceback.print_exc()
          # Return default values in case of error
          return {
              'symmetry_score': 0.5,
              'regularity_score': 0.5,
              'edge_density': 0.5
          }

      except Exception as e:
          print(f"Error in pattern structure analysis: {e}")
          import traceback
          traceback.print_exc()
          # Return default values in case of error
          return {
              'symmetry_score': 0.5,
              'regularity_score': 0.5,
              'edge_density': 0.5
          }

    def _estimate_layer_count(self, img_array: np.ndarray) -> int:
      """Estimate number of overlapping pattern layers."""
      gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
      blur = cv2.GaussianBlur(gray, (5,5), 0)
      ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      # Use connected components analysis
      num_labels, labels = cv2.connectedComponents(thresh)

      # Analyze depth through gradient magnitude
      sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
      sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
      gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

      # Estimate layers based on gradient peaks
      significant_gradients = np.sum(gradient_magnitude > np.mean(gradient_magnitude))
      estimated_layers = max(1, int(significant_gradients / (gray.size * 0.05)))

      return min(estimated_layers, 5)

    def analyze_color_harmony(self, image: Image.Image) -> Dict:
        """Analyze color harmony and relationships."""
        # Convert image to RGB array
        img_array = np.array(image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)

        # Extract dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_

        # Convert to HSV for better color analysis
        dominant_hsv = cv2.cvtColor(
            dominant_colors.reshape(1, -1, 3).astype(np.uint8),
            cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)

        return {
            'dominant_colors': dominant_colors,
            'color_variance': np.std(dominant_hsv[:, 1]),  # Saturation variance
            'value_range': np.ptp(dominant_hsv[:, 2]),    # Value range
            'is_monochromatic': np.std(dominant_hsv[:, 0]) < 20  # Hue variance threshold
        }

    def analyze_image(self, image_path: str) -> Tuple[List[Dict], Dict]:
      """Enhanced image analysis with pattern complexity, floral detection, and opacity guidance."""
      self.initialize_clip()

      try:
          print("Starting image analysis...")
          # Load image with alpha channel preserved for opacity analysis
          image = Image.open(image_path)

          # Get opacity guidance parameters
          opacity_metrics = self.analyze_opacity_guidance(image)
          recommended_params = self.get_recommended_parameters(opacity_metrics)

          # Apply opacity-based enhancements if needed
          if opacity_metrics['mean_opacity'] < 0.8:
              enhanced_image = self.enhance_low_opacity_regions(image)
          else:
              enhanced_image = image

          # Convert to RGB for further processing
          image_rgb = enhanced_image.convert("RGB")

          # Resize image
          max_size = 336
          width, height = image_rgb.size
          if width > height:
              new_width = min(max_size, width)
              new_height = int((height / width) * new_width)
          else:
              new_height = min(max_size, height)
              new_width = int((width / height) * new_height)

          image_rgb = image_rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)

          # Prepare image for CLIP
          clip_image = self.preprocess(image_rgb).unsqueeze(0).to(self.device)

          # Perform all analyses
          floral_metrics = self.analyze_floral_patterns(image_rgb)
          complexity_analysis = self.analyze_pattern_complexity(image_rgb)
          structure_features = self.analyze_pattern_structure(image_rgb)
          color_features = self.analyze_color_harmony(image_rgb)

          characteristics = []
          grouped_chars = {}

          with torch.no_grad():
              image_features = self.model.encode_image(clip_image)
              del clip_image
              torch.cuda.empty_cache()

              for group_name, terms in self.characteristic_groups.items():
                  print(f"Processing {group_name} characteristics...")

                  # Determine thresholds and boosts based on both floral and opacity characteristics
                  base_threshold = (
                      floral_metrics['confidence_adjustment']['base_threshold']
                      if 'floral' in group_name.lower()
                      else 0.2
                  )

                  # Adjust threshold based on opacity
                  threshold = base_threshold * (1.0 / max(0.3, opacity_metrics['mean_opacity']))

                  # Calculate combined boost factor
                  floral_boost = (
                      floral_metrics['confidence_adjustment']['boost_factor']
                      if 'floral' in group_name.lower()
                      else 1.0
                  )
                  opacity_boost = recommended_params['opacity_compensation']
                  combined_boost = floral_boost * opacity_boost

                  batch_size = 10
                  grouped_chars[group_name] = []

                  for i in range(0, len(terms), batch_size):
                      batch_terms = terms[i:i + batch_size]
                      text_inputs = clip.tokenize(batch_terms).to(self.device)
                      text_features = self.model.encode_text(text_inputs)

                      # Apply combined guidance and boosting
                      similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                      similarity *= combined_boost
                      similarity *= recommended_params['guidance_scale'] / 7.5  # Normalize to base guidance

                      values, indices = similarity[0].topk(min(3, len(batch_terms)))

                      for value, idx in zip(values, indices):
                          score = value.item()
                          if score > threshold:
                              char_dict = {
                                  "characteristic": batch_terms[idx],
                                  "score": score,
                                  "group": group_name
                              }
                              characteristics.append(char_dict)
                              grouped_chars[group_name].append(char_dict)

                      del text_inputs, text_features
                      torch.cuda.empty_cache()


              # Add all analysis results to grouped_chars
              grouped_chars.update({
                  'structure': structure_features,
                  'complexity': complexity_analysis,
                  'color_harmony': color_features,
                  'floral_metrics': floral_metrics,
                  'opacity_analysis': {
                      'metrics': opacity_metrics,
                      'recommended_parameters': recommended_params
                  }
              })

              return characteristics, grouped_chars

      except Exception as e:
          print(f"Error in image analysis: {e}")
          import traceback
          traceback.print_exc()
          return [], {}

    def initialize_clip(self):
      """Initialize CLIP model with proper error handling and memory management."""
      if self.model is None:
          try:
              print("Initializing CLIP model...")
              self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
              print("CLIP model initialized successfully")
          except Exception as e:
              print(f"Error initializing CLIP model: {e}")
              raise
class ImprovedImageVariationGenerator:
    def __init__(self):
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
        self.pipe = None
        self.characteristics_analyzer = EnhancedImageCharacteristics()
        self.geometric_pattern_handler = GeometricPatternHandler()
        self.color_variation_handler = ColorVariationHandler()
        self.abstract_color_handler = AbstractColorHandler()



    def _adjust_generation_params(self, complexity_analysis: Dict, geometric_properties: Dict) -> Dict:
        """Adjust generation parameters based on pattern complexity."""
        params = {
            'strength': 0.61,  # Base strength
            'guidance_scale': 6.5,
            'num_inference_steps': 25
        }

        # Adjust for pattern density
        density = complexity_analysis.get('pattern_density', 0.5)
        params['strength'] *= max(0.84, min(1.3, 1 - (density * 0.2)))  # More permissive range

        # Adjust for curve complexity
        curve_complex = complexity_analysis.get('curve_complexity', 0.5)
        params['guidance_scale'] *= max(0.84, min(1.2, 1 + (curve_complex * 0.2)))

        # Adjust for layer count
        layer_count = complexity_analysis.get('layer_count', 1)
        params['num_inference_steps'] += (layer_count - 1) * 2
        params['num_inference_steps'] = min(params['num_inference_steps'], 25)

        # New: Adjust strength based on coverage ratio and shape count
        coverage_ratio = geometric_properties.get('coverage_ratio', 0.0001)
        shape_count = geometric_properties.get('shape_count', 0.0001)

        if coverage_ratio == 0 and shape_count == 0:
            # Decrease strength for images with no detected geometric patterns
            params['strength'] *= 0.9
            params['guidance_scale'] *= 1.1

        elif coverage_ratio > 0.3 and shape_count > 100:
            # Increase strength for images with good coverage and many shapes
            params['strength'] *= 1.3

        return params

    def get_blended_parameters(self, characteristics: List[Dict], grouped_chars: Dict) -> Dict:
        """Enhanced parameter blending based on pattern complexity."""
        structure_features = grouped_chars.get('structure', {})
        color_features = grouped_chars.get('color_harmony', {})

        # Get key metrics
        edge_density = structure_features.get('edge_density', 0)
        symmetry_score = structure_features.get('symmetry_score', 0)
        regularity_score = structure_features.get('regularity_score', 0)

        # Base parameters with dynamic strength
        base_params = {
            'strength': 0.61,
            'guidance_scale': 6.5,
            'num_inference_steps': 15,
        }

        # Adjust strength based on pattern complexity
        if edge_density > 0.5:
            base_params['strength'] = 0.5  # Conservative for complex patterns

        # Increase steps for regular patterns
        if regularity_score > 0.8:
            base_params['strength'] *= 1.1  # Instead of increasing steps
            base_params['guidance_scale'] *= 0.9

        # Further reduce strength for highly symmetric patterns
        if symmetry_score > 0.95:
            base_params['strength'] *= 0.9

        # Adjust for color preservation
        if color_features.get('is_monochromatic', False):
            base_params['strength'] *= 0.9  # More conservative for monochromatic patterns

        return base_params

    def _build_prompt(self, characteristics: List[Dict], grouped_chars: Dict, preserve_colors: bool) -> str:
        """Enhanced prompt building with better pattern preservation."""
        structure_features = grouped_chars.get('structure', {})
        color_features = grouped_chars.get('color_harmony', {})
        floral_metrics = grouped_chars.get('floral_metrics', {})

        # Start with base prompt parts
        prompt_parts = ["high-quality commercial pattern"]

        pattern_type = floral_metrics.get('pattern_type', 'unknown')
        if pattern_type == 'floral':
            prompt_parts.extend([
                "floral pattern",
                "organic curved elements",
                "natural flowing design"
            ])

        elif pattern_type == 'unknown':
            prompt_parts.append("maintain original pattern style")


        # Add structural preservation hints
        if structure_features.get('regularity_score', 0) > 0.7:
            prompt_parts.extend([
                "pattern elements preservation",
                "preserve pattern spacing and alignment"
            ])

        if structure_features.get('symmetry_score', 0) > 0.9:
            prompt_parts.append("perfectly symmetrical composition")


        # Add edge preservation for complex patterns
        if structure_features.get('edge_density', 0) > 0.6:
            prompt_parts.extend([
                "precise edge definition",
                "crisp pattern boundaries",
                "maintain intricate details"
            ])

        # Add style characteristics
        style_chars = [c['characteristic'] for c in grouped_chars.get('style', [])]
        if style_chars:
            prompt_parts.append(f"with {', '.join(style_chars[:2])} style")

        # Add pattern characteristics
        #pattern_chars = [c['characteristic'] for c in grouped_chars.get('pattern', [])]
        #if pattern_chars:
        #    prompt_parts.append(f"featuring {', '.join(pattern_chars[:2])} patterns")

        # Add color guidance if not preserving colors
        if not preserve_colors and not color_features.get('is_monochromatic', False):
            color_chars = [c['characteristic'] for c in grouped_chars.get('color', [])]
            if color_chars:
                prompt_parts.append(f"in {', '.join(color_chars[:2])} color scheme")

        prompt_parts.extend([
            "creative reinterpretation of pattern elements",
            "dynamic rearrangement of components",
            "varied element positions and scales",
            "innovative structural composition",
            "thematically consistent novel arrangements",
            "manufacturing-ready textile design",

        ])

        if any('geometric' in c['characteristic'] for c in grouped_chars.get('style', [])):
            prompt_parts.extend([
                "dynamic spatial arrangement",
                "dynamic shapes dimensions",
                "creative shapes position",

            ])

        return ", ".join(prompt_parts)

    def _generate_negative_prompt(self, is_geometric: bool) -> str:
        """Generate appropriate negative prompt based on pattern type."""
        base_negative = "photographic, realistic, photograph, distorted patterns, broken symmetry, inconsistent details, noisy, pixelated, blurry, poor quality"

        if is_geometric:
            return f"{base_negative}, organic shapes, curved lines, natural elements, irregular shapes, distorted geometry, incomplete shapes, rough edges, imprecise geometry, non-geometric elements, freeform patterns"
        return base_negative

    def _generate_reproducible_seed(self, image_path: str, variation_index: int) -> int:
        """Generate reproducible seed while maintaining pattern consistency."""
        filename = Path(image_path).stem
        filename_seed = sum(ord(c) * (i + 1) for i, c in enumerate(filename))
        combined_seed = filename_seed + (variation_index * 10000)
        return combined_seed % 2147483647

    def analyze_image(self, input_path: str):
        """Perform comprehensive image analysis once."""
        # Load and analyze image
        characteristics, grouped_chars = self.characteristics_analyzer.analyze_image(input_path)
        original_image = Image.open(input_path).convert("RGBA")
        process_image = original_image.convert("RGB")
        is_geometric, geo_properties = self.geometric_pattern_handler.detect_geometric_pattern(process_image)

        # Process image dimensions
        width, height = process_image.size
        max_size = 1024
        if width > height:
            new_width = min(max_size, width)
            new_height = int((height / width) * new_width)
        else:
            new_height = min(max_size, height)
            new_width = int((width / height) * new_height)

        # Ensure dimensions are multiples of 8
        new_width = new_width - (new_width % 8)
        new_height = new_height - (new_height % 8)
        process_image = process_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Extract alpha channel if present
        alpha_channel = original_image.split()[3] if len(original_image.split()) == 4 else None
        is_abstract, abstract_properties = self.abstract_color_handler.detect_abstract_color_pattern(
            np.array(process_image)
        )

        return {
            'characteristics': characteristics,
            'grouped_chars': grouped_chars,
            'is_geometric': is_geometric,
            'geo_properties': geo_properties,
            'original_image': original_image,
            'process_image': process_image,
            'alpha_channel': alpha_channel,
            'is_abstract': is_abstract,
            'abstract_properties': abstract_properties
        }

    def get_generation_parameters(self, analysis_results: dict, variation_index: int, num_variations: int):
        """Calculate all parameters needed for generation."""
        complexity_analysis = analysis_results['grouped_chars'].get('complexity', {})
        base_params = self._adjust_generation_params(complexity_analysis, analysis_results['geo_properties'])

        if analysis_results['is_abstract']:
            # Use abstract color handling
            base_params = self.abstract_color_handler.get_abstract_parameters(
                variation_index,
                analysis_results['abstract_properties']
            )
            prompt = self.abstract_color_handler.enhance_abstract_prompt(
                "high-quality artistic composition",
                variation_index,
                analysis_results['abstract_properties']
            )
            negative_prompt = self.abstract_color_handler.get_abstract_negative_prompt()

        if analysis_results['is_geometric']:
            geometric_variation_handler = GeometricVariationHandler()
            geo_params = geometric_variation_handler.get_geometric_parameters(
                variation_index,
                num_variations,
                analysis_results['original_image']
            )
            base_params.update(geo_params)

        preserve_colors = random.random() < 0.75

        # Build prompts
        prompt = self._build_prompt(
            analysis_results['characteristics'],
            analysis_results['grouped_chars'],
            preserve_colors
        )

        if analysis_results['is_geometric']:
            prompt = geometric_variation_handler.enhance_geometric_prompt(
                prompt,
                variation_index,
                analysis_results['original_image']
            )

        prompt = self.color_variation_handler.enhance_variation_prompt(prompt, variation_index)
        negative_prompt = self._generate_negative_prompt(analysis_results['is_geometric'])

        return {
            'base_params': base_params,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'preserve_colors': preserve_colors
        }

    def generate_variations(self, input_path: str, output_dir: str, num_variations: int = 5,
                       color_variation_frequency: float = 0.6,
                       strength: float = None,
                       guidance_scale: float = None,
                       num_inference_steps: int = None, user_prompt: str = "") -> List[Image.Image]:
        """Generate variations with integrated parameter handling."""
        self.initialize_pipeline()

        # Perform analysis once
        analysis_results = self.analyze_image(input_path)

        variations = []
        for i in range(num_variations):
            # Get base parameters for this variation
            gen_params = self.get_generation_parameters(analysis_results, i, num_variations)

            # Override with custom parameters if provided
            if strength is not None:
                gen_params['base_params']['strength'] = strength
            if guidance_scale is not None:
                gen_params['base_params']['guidance_scale'] = guidance_scale
            if num_inference_steps is not None:
                gen_params['base_params']['num_inference_steps'] = num_inference_steps

            # Generate seed
            seed = self._generate_reproducible_seed(input_path, i)
            generator = torch.Generator(device="cuda").manual_seed(seed)

            print(f"\nGenerating variation {i+1}/{num_variations}")
            print(f"Parameters: {gen_params['base_params']}")
            print(f"Prompt: {gen_params['prompt']}")

            prompt_parts = gen_params['prompt'].split(",")
            if len(prompt_parts) > 0:
                first_part = prompt_parts[0].split()
                insertion_index = min(3, len(first_part)) # insert after 3 words, or less if the first part is shorter
                modified_first_part = " ".join(first_part[:insertion_index]) + ", " + user_prompt + " " + " ".join(first_part[insertion_index:])
                combined_prompt = ", ".join([modified_first_part] + prompt_parts[1:])
            else:
                combined_prompt = user_prompt + ", " + gen_params['prompt']

            result = self.pipe(
                prompt=combined_prompt,  # Use the combined prompt
                negative_prompt=gen_params['negative_prompt'],
                image=analysis_results['process_image'],
                strength=gen_params['base_params']['strength'],
                guidance_scale=gen_params['base_params']['guidance_scale'],
                num_inference_steps=gen_params['base_params']['num_inference_steps'],
                generator=generator,
            ).images[0]


            # Post-process result
            result = result.resize(analysis_results['original_image'].size, Image.Resampling.LANCZOS)
            if analysis_results['alpha_channel']:
                result = result.convert("RGBA")
                result.putalpha(analysis_results['alpha_channel'])

            variations.append(result)

        return variations



    def initialize_pipeline(self):
        """Initialize the pipeline with specific configurations."""
        if self.pipe is None:
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            self.pipe.load_lora_weights(self.lcm_lora_id)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_attention_slicing()
            self.pipe.to("cuda")

def process_images(input_folder, num_variations=5, color_variation_frequency=0.6):
    """Process images from Google Drive folder and save results back to Drive."""

    # Setup directories in Drive
    input_dir = Path('../Input')
    output_dir = Path('../Output')
    grid_dir = Path('../Grids')

    for dir_path in [input_dir, output_dir, grid_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize the generator
    generator = ImprovedImageVariationGenerator()

    # Get all image files from the input folder
    input_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # Track progress
    total_files = len(input_files)
    processed_count = 0
    skipped_count = 0

    for idx, input_file in enumerate(input_files, 1):
        input_path = Path(input_folder) / input_file
        base_name = os.path.splitext(input_file)[0]
        extension = input_path.suffix

        # Check if any variations exist for this image
        variation_pattern = f"{base_name}[a-{chr(96 + num_variations)}]{extension}"
        existing_variations = [
            f for f in os.listdir(output_dir)
            if fnmatch.fnmatch(f, variation_pattern)
        ]

        # Check if grid exists
        grid_exists = any(
            fnmatch.fnmatch(f, f"{base_name}_grid.*")
            for f in os.listdir(grid_dir)
        )

        # Skip if all variations and grid exist
        if existing_variations and len(existing_variations) == num_variations and grid_exists:
            print(f"[{idx}/{total_files}] Skipping {input_file} (already processed)")
            skipped_count += 1
            continue

        # Process the image
        print(f"\n[{idx}/{total_files}] Processing {input_file}")
        original_image = Image.open(input_path)
        print(f"Original dimensions: {original_image.size}")

        # Copy input file to organized input directory
        drive_input_path = input_dir / input_file
        shutil.copy2(input_path, drive_input_path)

        # Generate variations
        variations = generator.generate_variations(
            input_path,
            output_dir,
            num_variations,
            color_variation_frequency
        )

        # Save variations to Drive
        for var_idx, var_image in enumerate(variations):
            output_filename = f"{base_name}{chr(97 + var_idx)}{extension}"
            output_path = output_dir / output_filename

            # Convert RGBA to RGB if saving as JPEG
            if var_image.mode == 'RGBA' and extension.lower() in ('.jpg', '.jpeg'):
                var_image = var_image.convert('RGB')

            var_image.save(output_path)
            print(f"  Saved variation {chr(97 + var_idx)} to: {var_image.size}")

        # Create and save grid visualization
        grid_path = create_variation_grid(
            original_image,
            variations,
            base_name,
            grid_dir
        )
        print(f"  Saved grid visualization to Drive: {grid_path}")
        processed_count += 1

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total images: {total_files}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")


input_folder = "../Input"
process_images(input_folder, num_variations=5)











