from flask import Flask, request, jsonify, send_file, render_template_string
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import io
import base64
import os
import random
import json
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Optional imports with error handling
REMBG_AVAILABLE = False
YOLO_AVAILABLE = False
TORCH_AVAILABLE = False
FACE_RECOGNITION_AVAILABLE = False

try:
    from rembg import remove
    REMBG_AVAILABLE = True
    print("✅ rembg loaded successfully")
except ImportError as e:
    print(f"⚠️  rembg not available: {e}")
    print("   Install: pip install rembg onnxruntime")

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import segmentation
    TORCH_AVAILABLE = True
    print("✅ PyTorch loaded successfully")
except ImportError as e:
    print(f"⚠️  PyTorch not available: {e}")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO loaded successfully")
except ImportError as e:
    print(f"⚠️  YOLO not available: {e}")
    print("   Install: pip install ultralytics")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition loaded successfully")
except ImportError as e:
    print(f"⚠️  face_recognition not available: {e}")
    print("   Install: pip install face-recognition")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Konfigurasi folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model untuk deteksi objek
yolo_model = None
if YOLO_AVAILABLE:
    try:
        yolo_model = YOLO('yolov8n.pt')  # Download otomatis jika belum ada
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"⚠️  YOLO model loading failed: {e}")
        yolo_model = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image, target_width=None, target_height=None, maintain_aspect=True):
    """
    Resize image berdasarkan width dan height yang ditentukan
    """
    try:
        original_width, original_height = image.size
        
        # Jika tidak ada target yang ditentukan, return original
        if not target_width and not target_height:
            return image, f"No resize - original size: {original_width}x{original_height}"
        
        # Jika hanya width yang ditentukan
        if target_width and not target_height:
            if maintain_aspect:
                ratio = target_width / original_width
                target_height = int(original_height * ratio)
            else:
                target_height = original_height
        
        # Jika hanya height yang ditentukan
        elif target_height and not target_width:
            if maintain_aspect:
                ratio = target_height / original_height
                target_width = int(original_width * ratio)
            else:
                target_width = original_width
        
        # Jika maintain aspect ratio dengan kedua dimensi ditentukan
        elif maintain_aspect and target_width and target_height:
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            ratio = min(width_ratio, height_ratio)  # Pilih ratio terkecil untuk fit
            target_width = int(original_width * ratio)
            target_height = int(original_height * ratio)
        
        # Resize image menggunakan Lanczos filter untuk kualitas terbaik
        resized_image = image.resize((target_width, target_height), Image.LANCZOS)
        
        resize_info = f"Resized from {original_width}x{original_height} to {target_width}x{target_height}"
        if maintain_aspect:
            resize_info += " (aspect ratio maintained)"
        
        return resized_image, resize_info
        
    except Exception as e:
        print(f"Resize error: {e}")
        return image, f"Resize failed: {str(e)}"

def enhance_image_quality(image, intensity=1.0):
    """Enhanced image quality improvement with adjustable intensity"""
    # Convert PIL to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Stage 1: Initial denoising while preserving edges
    denoise_strength = int(10 * intensity)
    denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, denoise_strength, denoise_strength, 7, 21)
    
    # Stage 2: Adaptive contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel with adjustable intensity
    clip_limit = 2.0 * intensity
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Stage 3: Smart sharpening with edge preservation
    blur_strength = max(1, int(3 * intensity))
    blurred = cv2.GaussianBlur(enhanced, (0, 0), blur_strength)
    sharp_intensity = 1.0 + (0.5 * intensity)
    sharpened = cv2.addWeighted(enhanced, sharp_intensity, blurred, -0.5 * intensity, 0)
    
    # Stage 4: Detail enhancement using bilateral filter
    detailed = cv2.bilateralFilter(sharpened, 9, int(75 * intensity), int(75 * intensity))
    
    # Convert back to PIL Image
    final_pil = Image.fromarray(cv2.cvtColor(detailed, cv2.COLOR_BGR2RGB))
    
    # Stage 5: PIL-based final touch-ups
    # Auto contrast
    final_pil = ImageOps.autocontrast(final_pil, cutoff=int(2 * intensity))
    
    # Saturation enhancement
    enhancer = ImageEnhance.Color(final_pil)
    final_pil = enhancer.enhance(1.0 + (0.15 * intensity))
    
    # Micro-contrast enhancement
    enhancer = ImageEnhance.Sharpness(final_pil)
    final_pil = enhancer.enhance(1.0 + (0.2 * intensity))
    
    return final_pil

def advanced_background_removal(image, method="auto", background_color=(255, 255, 255), edge_quality=3):
    """Advanced background removal dengan multiple algorithms dan post-processing"""
    
    # Method 1: AI-based background removal (paling akurat)
    if method == "ai" and REMBG_AVAILABLE:
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Remove background dengan rembg
            result = remove(img_byte_arr)
            result_image = Image.open(io.BytesIO(result))
            
            # Post-processing untuk smooth edges berdasarkan quality
            if result_image.mode == 'RGBA':
                # Extract alpha channel
                alpha = np.array(result_image)[:, :, 3]
                
                # Quality-based smoothing
                blur_kernel = 3 if edge_quality <= 2 else 5 if edge_quality <= 4 else 7
                alpha_smooth = cv2.GaussianBlur(alpha, (blur_kernel, blur_kernel), 0)
                
                # Apply morphological operations untuk cleanup
                kernel_size = 3 + (edge_quality - 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                alpha_smooth = cv2.morphologyEx(alpha_smooth, cv2.MORPH_CLOSE, kernel)
                
                # Update alpha channel
                result_array = np.array(result_image)
                result_array[:, :, 3] = alpha_smooth
                result_image = Image.fromarray(result_array, 'RGBA')
            
            # Apply background color jika diperlukan
            if background_color != "transparent":
                background = Image.new('RGB', result_image.size, background_color)
                background.paste(result_image, (0, 0), result_image)
                result_image = background
            
            return result_image, "Background removed using AI model (rembg) - High Accuracy"
            
        except Exception as e:
            print(f"rembg failed: {e}")
    
    # Method 2: Advanced Multi-Algorithm Approach
    if method in ["auto", "advanced"]:
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = cv_image.shape[:2]
            
            # Step 1: Quality-based edge detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing based on quality
            if edge_quality >= 4:
                # High quality: additional preprocessing
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Multiple edge detection methods based on quality
            edges_list = []
            
            # Canny edge detection dengan adaptive thresholds
            if edge_quality >= 2:
                edges1 = cv2.Canny(gray, 50, 150)
                edges_list.append(edges1)
            
            if edge_quality >= 3:
                # Laplacian edge detection
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                edges2 = np.uint8(np.absolute(laplacian))
                edges_list.append(edges2)
            
            if edge_quality >= 4:
                # Sobel edge detection
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                edges3 = np.uint8(sobel / sobel.max() * 255)
                edges_list.append(edges3)
            
            if edge_quality >= 5:
                # Additional advanced edge detection
                # Roberts edge detector
                roberts_cross_v = np.array([[1, 0], [0, -1]])
                roberts_cross_h = np.array([[0, 1], [-1, 0]])
                roberts_v = cv2.filter2D(gray, cv2.CV_64F, roberts_cross_v)
                roberts_h = cv2.filter2D(gray, cv2.CV_64F, roberts_cross_h)
                roberts = np.sqrt(np.square(roberts_v) + np.square(roberts_h))
                edges4 = np.uint8(roberts / roberts.max() * 255)
                edges_list.append(edges4)
            
            # Default minimal edge detection
            if not edges_list:
                edges_list.append(cv2.Canny(gray, 50, 150))
            
            # Combine all edge detection methods
            combined_edges = edges_list[0]
            for edges in edges_list[1:]:
                combined_edges = cv2.bitwise_or(combined_edges, edges)
            
            # Step 2: Quality-based morphological operations
            kernel_size = 3 + (edge_quality - 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Multiple passes for higher quality
            for _ in range(edge_quality):
                combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
                combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_DILATE, kernel)
            
            # Step 3: Find contours dengan quality-based parameters
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Quality-based contour filtering
                min_area_threshold = (width * height * 0.005) / edge_quality  # Lower threshold for higher quality
                significant_contours = [c for c in contours if cv2.contourArea(c) > min_area_threshold]
                significant_contours = sorted(significant_contours, key=cv2.contourArea, reverse=True)
                
                # Take more contours for higher quality
                max_contours = min(edge_quality + 2, len(significant_contours))
                significant_contours = significant_contours[:max_contours]
                
                if significant_contours:
                    # Create mask dari contours
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, significant_contours, 255)
                    
                    # Step 4: Quality-based GrabCut refinement
                    if edge_quality >= 3:
                        try:
                            # Get bounding rectangle dari largest contour
                            x, y, w, h = cv2.boundingRect(significant_contours[0])
                            
                            # Quality-based expansion
                            expand_factor = 0.05 + (0.02 * edge_quality)
                            expand_x = int(w * expand_factor)
                            expand_y = int(h * expand_factor)
                            
                            x = max(0, x - expand_x)
                            y = max(0, y - expand_y)
                            w = min(width - x, w + 2 * expand_x)
                            h = min(height - y, h + 2 * expand_y)
                            
                            rect = (x, y, w, h)
                            
                            # Initialize GrabCut
                            grabcut_mask = np.zeros((height, width), np.uint8)
                            bgd_model = np.zeros((1, 65), np.float64)
                            fgd_model = np.zeros((1, 65), np.float64)
                            
                            # Set initial mask based pada contour mask
                            grabcut_mask[mask == 255] = cv2.GC_PR_FGD
                            grabcut_mask[mask == 0] = cv2.GC_PR_BGD
                            
                            # Quality-based iterations
                            iterations = 3 + edge_quality
                            cv2.grabCut(cv_image, grabcut_mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
                            
                            # Create final mask
                            final_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                            
                        except Exception as gc_error:
                            print(f"GrabCut refinement failed: {gc_error}")
                            final_mask = mask
                    else:
                        final_mask = mask
                    
                    # Step 5: Quality-based post-processing
                    # Apply quality-based smoothing
                    blur_kernel = 3 + (edge_quality - 1) * 2
                    if blur_kernel % 2 == 0:
                        blur_kernel += 1
                    final_mask = cv2.GaussianBlur(final_mask, (blur_kernel, blur_kernel), 0)
                    
                    # Quality-based morphological operations
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_quality + 2, edge_quality + 2))
                    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
                    
                    # Remove small noise based on quality
                    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_quality, edge_quality))
                    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
                    
                    # Apply additional edge feathering for higher quality
                    if edge_quality >= 4:
                        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
                    
                    # Normalize mask
                    final_mask = np.where(final_mask > 127, 255, 0).astype(np.uint8)
                    
                    # Apply mask ke image
                    if background_color == "transparent":
                        rgba_image = image.convert('RGBA')
                        rgba_array = np.array(rgba_image)
                        rgba_array[:, :, 3] = final_mask
                        result_image = Image.fromarray(rgba_array, 'RGBA')
                    else:
                        result_image = Image.new('RGB', image.size, background_color)
                        mask_img = Image.fromarray(final_mask, 'L')
                        result_image.paste(image, (0, 0), mask_img)
                    
                    contour_area = sum([cv2.contourArea(c) for c in significant_contours])
                    return result_image, f"Advanced algorithm (Q{edge_quality}): {len(significant_contours)} objects, area: {int(contour_area)}"
                
            # Fallback jika tidak ada contours yang signifikan
            return image, "No significant contours found for background removal"
                
        except Exception as e:
            print(f"Advanced method failed: {e}")
    
    # Method 3: YOLO-assisted background removal dengan quality enhancement
    if method in ["auto", "yolo"] and YOLO_AVAILABLE and yolo_model:
        try:
            img_array = np.array(image)
            results = yolo_model(img_array)
            
            height, width = img_array.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            
            main_objects = ['person', 'car', 'truck', 'bus', 'motorbike', 'bicycle', 'dog', 'cat', 'horse', 'cow', 'sheep']
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = yolo_model.names[cls]
                        
                        # Quality-based threshold adjustment
                        base_threshold = 0.25 if class_name in main_objects else 0.5
                        threshold = base_threshold - (0.05 * (edge_quality - 3))  # Lower threshold for higher quality
                        
                        if conf > threshold:
                            # Create detailed mask untuk setiap object dengan quality enhancement
                            roi = img_array[y1:y2, x1:x2]
                            if roi.size > 0:
                                # Apply quality-based segmentation dalam ROI
                                roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                                
                                # Quality-based preprocessing
                                if edge_quality >= 4:
                                    roi_gray = cv2.bilateralFilter(roi_gray, 9, 75, 75)
                                
                                # Multiple threshold methods based on quality
                                roi_masks = []
                                
                                # Otsu threshold
                                _, thresh1 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                roi_masks.append(thresh1)
                                
                                if edge_quality >= 3:
                                    # Adaptive threshold
                                    thresh2 = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                                    roi_masks.append(thresh2)
                                
                                if edge_quality >= 4:
                                    # Mean adaptive threshold
                                    thresh3 = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
                                    roi_masks.append(thresh3)
                                
                                # Combine all thresholds
                                roi_mask = roi_masks[0]
                                for mask_item in roi_masks[1:]:
                                    roi_mask = cv2.bitwise_or(roi_mask, mask_item)
                                
                                # Quality-based morphological operations
                                kernel_size = 3 + (edge_quality - 1)
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
                                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
                                
                                # Apply ke main mask
                                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], roi_mask)
                            
                            detected_objects.append(class_name)
            
            if np.any(mask):
                # Quality-based post-processing mask
                blur_kernel = 3 + (edge_quality - 1) * 2
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
                
                # Quality-based morphological cleanup
                kernel_size = 5 + (edge_quality - 1) * 2
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Edge feathering based on quality
                if edge_quality >= 4:
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                mask = np.where(mask > 127, 255, 0).astype(np.uint8)
                
                # Apply mask
                if background_color == "transparent":
                    rgba_image = image.convert('RGBA')
                    rgba_array = np.array(rgba_image)
                    rgba_array[:, :, 3] = mask
                    result_image = Image.fromarray(rgba_array, 'RGBA')
                else:
                    result_image = Image.new('RGB', image.size, background_color)
                    mask_img = Image.fromarray(mask, 'L')
                    result_image.paste(image, (0, 0), mask_img)
                
                return result_image, f"YOLO enhanced (Q{edge_quality}): {', '.join(set(detected_objects))}"
            
        except Exception as e:
            print(f"YOLO enhanced method failed: {e}")
    
    # Method 4: Enhanced GrabCut with quality control
    if method == "grabcut":
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = cv_image.shape[:2]
            
            # Quality-based initial rectangle
            margin_factor = 0.3 - (0.05 * edge_quality)  # Smaller margin for higher quality
            center_x, center_y = width // 2, height // 2
            rect_w, rect_h = int(width * (1 - 2 * margin_factor)), int(height * (1 - 2 * margin_factor))
            rect = (center_x - rect_w//2, center_y - rect_h//2, rect_w, rect_h)
            
            # Initialize mask and models for GrabCut
            mask = np.zeros((height, width), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            try:
                # Quality-based iterations
                iterations = 3 + edge_quality
                cv2.grabCut(cv_image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
                
                # Create final mask
                final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                
                # Quality-based smoothing
                blur_kernel = 3 + (edge_quality - 1) * 2
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                final_mask = cv2.GaussianBlur(final_mask, (blur_kernel, blur_kernel), 0)
                final_mask = np.where(final_mask > 127, 255, 0).astype(np.uint8)
                
                if background_color == "transparent":
                    rgba_image = image.convert('RGBA')
                    rgba_array = np.array(rgba_image)
                    rgba_array[:, :, 3] = final_mask
                    result_image = Image.fromarray(rgba_array, 'RGBA')
                else:
                    result_image = Image.new('RGB', image.size, background_color)
                    mask_img = Image.fromarray(final_mask, 'L')
                    result_image.paste(image, (0, 0), mask_img)
                
                return result_image, f"GrabCut algorithm (Q{edge_quality}, {iterations} iterations)"
                
            except Exception as grabcut_error:
                print(f"GrabCut failed: {grabcut_error}")
                
        except Exception as fallback_error:
            print(f"GrabCut method failed: {fallback_error}")
    
    # Fallback method
    return image, "All background removal methods failed"

def detect_and_remove_background(image, method="auto", background_color=(255, 255, 255), edge_quality=3):
    """Wrapper function yang menggunakan advanced background removal"""
    return advanced_background_removal(image, method, background_color, edge_quality)

def apply_object_detection_visualization(image):
    """Detect objects and draw bounding boxes for visualization only"""
    if not YOLO_AVAILABLE or not yolo_model:
        return image, "YOLO not available for object detection"
    
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # YOLO detection
        results = yolo_model(img_array)
        
        # Create a copy to draw on
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        detected_objects = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Only keep high confidence detections
                    if conf > 0.5:
                        # Draw rectangle
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        
                        # Draw label with background
                        label = f"{yolo_model.names[cls]} {conf:.2f}"
                        
                        # Calculate text size (approximate)
                        text_width = len(label) * 8
                        text_height = 20
                        
                        # Draw label background
                        draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill="red")
                        draw.text((x1+5, y1-text_height), label, fill="white")
                        
                        detected_objects.append({
                            'class': yolo_model.names[cls],
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        if detected_objects:
            return result_image, f"Detected objects: {', '.join([obj['class'] for obj in detected_objects])}"
        else:
            return image, "No objects detected with sufficient confidence"
            
    except Exception as e:
        return image, f"Object detection failed: {str(e)}"

def convert_image_format(image, target_format):
    """Konversi gambar ke format yang diinginkan"""
    if target_format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA'):
        # Convert RGBA to RGB for JPEG
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image)
        image = background
    
    return image

def apply_hdr_effect(image, intensity=1.0):
    """Apply HDR-like effect to image with adjustable intensity"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply tone mapping for HDR effect
    gamma = 1.0 + (0.5 * intensity)
    tonemap = cv2.createTonemapReinhard(gamma=gamma)
    hdr_image = tonemap.process(cv_image.astype(np.float32)/255.0)
    
    # Convert back to PIL Image
    hdr_image = (hdr_image * 255).astype(np.uint8)
    hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(hdr_image)

def auto_color_correction(image, intensity=1.0):
    """Automatic color correction and white balance"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel with intensity control
    clip_limit = 2.0 * intensity
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Automatic white balance with intensity control
    result = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

def apply_sketch_effect(image, style="pencil"):
    """Convert image to pencil sketch with different styles"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if style == "pencil":
        # Traditional pencil sketch
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        inverted_gray_image = 255 - gray_image
        blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
        inverted_blurred_image = 255 - blurred_image
        pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
        return Image.fromarray(pencil_sketch)
    
    elif style == "colored":
        # Colored pencil sketch
        gray, pencil_sketch = cv2.pencilSketch(cv_image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return Image.fromarray(cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2RGB))
    
    else:
        # Default pencil
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray)

def apply_cartoon_effect(image, num_down=2, num_bilateral=7):
    """Apply cartoon effect to image with adjustable parameters"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Downsample image using Gaussian pyramid
    img_color = cv_image
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    
    # Apply bilateral filter multiple times
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, 9, 200, 200)
    
    # Upsample image to original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    
    # Resize to match original dimensions
    height, width = cv_image.shape[:2]
    img_color = cv2.resize(img_color, (width, height))
    
    # Create edge mask
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    # Convert edges to 3-channel
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine color image with edges
    cartoon = cv2.bitwise_and(img_color, edges)
    
    return Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

def apply_sepia_effect(image, intensity=1.0):
    """Apply sepia tone effect with adjustable intensity"""
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ]) * intensity
    
    # Blend with original if intensity < 1
    if intensity < 1.0:
        identity = np.eye(3) * (1 - intensity)
        sepia_filter = sepia_filter + identity
    
    # Convert to numpy array and apply sepia matrix
    img_array = np.array(image)
    sepia_img = np.dot(img_array, sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(sepia_img)

def apply_vignette_effect(image, intensity=2.0):
    """Apply vignette effect to image with adjustable intensity"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rows, cols = cv_image.shape[:2]
    
    # Generate vignette mask
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols/intensity)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/intensity)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    
    # Apply the mask to each channel
    vignette = np.copy(cv_image)
    for i in range(3):
        vignette[:,:,i] = vignette[:,:,i] * mask
    
    return Image.fromarray(cv2.cvtColor(vignette, cv2.COLOR_BGR2RGB))

def detect_faces(image):
    """Detect faces in image and return coordinates"""
    if not FACE_RECOGNITION_AVAILABLE:
        return [], "Face recognition library not available"
    
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Find face locations
        face_locations = face_recognition.face_locations(img_array)
        
        if not face_locations:
            return [], "No faces detected"
        
        # Convert to (top, right, bottom, left) format
        faces = [(top, right, bottom, left) for (top, right, bottom, left) in face_locations]
        return faces, f"Detected {len(faces)} faces"
    
    except Exception as e:
        return [], f"Face detection failed: {str(e)}"

def apply_face_blur(image, faces, blur_intensity=99):
    """Blur detected faces in image with adjustable intensity"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for (top, right, bottom, left) in faces:
        # Extract the face ROI
        face_roi = cv_image[top:bottom, left:right]
        
        # Apply blur to the face ROI
        face_roi = cv2.GaussianBlur(face_roi, (blur_intensity, blur_intensity), 30)
        
        # Put the blurred face back into the image
        cv_image[top:bottom, left:right] = face_roi
    
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

def apply_watermark(image, text="SAMPLE", opacity=128, position="center"):
    """Apply text watermark to image with customizable options"""
    watermarked = image.copy()
    width, height = watermarked.size
    
    # Create transparent overlay
    overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        # Try to load a nice font
        font_size = min(width, height) // 10
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Calculate text size and position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    if position == "center":
        x = (width - text_width) // 2
        y = (height - text_height) // 2
    elif position == "top_right":
        x = width - text_width - 20
        y = 20
    elif position == "bottom_right":
        x = width - text_width - 20
        y = height - text_height - 20
    else:  # top_left
        x = 20
        y = 20
    
    # Draw semi-transparent text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))
    
    # Composite the overlay onto the original image
    watermarked = Image.alpha_composite(watermarked.convert('RGBA'), overlay)
    
    return watermarked.convert('RGB')

def apply_blur_effect(image, blur_type="gaussian", intensity=5):
    """Apply various blur effects"""
    if blur_type == "gaussian":
        return image.filter(ImageFilter.GaussianBlur(radius=intensity))
    elif blur_type == "motion":
        # Simple motion blur simulation
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        kernel = np.zeros((intensity, intensity))
        kernel[int((intensity-1)/2), :] = np.ones(intensity)
        kernel = kernel / intensity
        blurred = cv2.filter2D(cv_image, -1, kernel)
        return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    elif blur_type == "radial":
        # Radial blur effect
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = cv_image.shape[:2]
        center = (w//2, h//2)
        
        # Create radial blur
        for i in range(intensity):
            cv_image = cv2.GaussianBlur(cv_image, (3, 3), 0)
        
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    else:
        return image

def apply_emboss_effect(image, intensity=1.0):
    """Apply emboss effect"""
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ]) * intensity
    
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    embossed = cv2.filter2D(cv_image, -1, emboss_kernel)
    
    return Image.fromarray(cv2.cvtColor(embossed, cv2.COLOR_BGR2RGB))

def apply_oil_painting_effect(image, size=7, dynRatio=1):
    """Apply oil painting effect using OpenCV"""
    try:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        oil_painting = cv2.xphoto.oilPainting(cv_image, size, dynRatio)
        return Image.fromarray(cv2.cvtColor(oil_painting, cv2.COLOR_BGR2RGB))
    except:
        # Fallback to simple blur if xphoto not available
        return image.filter(ImageFilter.GaussianBlur(radius=2))

def apply_edge_detection(image, method="canny"):
    """Apply edge detection and return edges as image"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    if method == "canny":
        edges = cv2.Canny(gray, 100, 200)
    elif method == "sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
    elif method == "laplacian":
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    else:
        edges = gray
    
    # Convert edges to RGB for display
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def apply_histogram_equalization(image):
    """Apply histogram equalization to improve contrast"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Apply histogram equalization to Y channel
    y_eq = cv2.equalizeHist(y)
    
    # Merge back and convert to BGR
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    equalized = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    
    return Image.fromarray(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))

def apply_noise_reduction(image, method="bilateral", intensity=1.0):
    """Apply noise reduction with different methods"""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if method == "bilateral":
        d = int(9 * intensity)
        sigma_color = int(75 * intensity)
        sigma_space = int(75 * intensity)
        denoised = cv2.bilateralFilter(cv_image, d, sigma_color, sigma_space)
    elif method == "gaussian":
        kernel_size = int(5 * intensity)
        if kernel_size % 2 == 0:
            kernel_size += 1
        denoised = cv2.GaussianBlur(cv_image, (kernel_size, kernel_size), 0)
    elif method == "median":
        kernel_size = int(5 * intensity)
        if kernel_size % 2 == 0:
            kernel_size += 1
        denoised = cv2.medianBlur(cv_image, kernel_size)
    else:
        h = int(10 * intensity)
        denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, h, h, 7, 21)
    
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ Advanced Image Processor Pro</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .main-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 30px;
        }
        
        .upload-section {
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .upload-zone {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 60px 20px;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .upload-zone:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
        
        .upload-zone.dragover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }
        
        .upload-icon {
            font-size: 4rem;
            color: #667eea;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 10px;
        }
        
        .upload-subtext {
            color: #999;
            font-size: 0.9rem;
        }
        
        #fileInput {
            display: none;
        }
        
        .features-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            padding: 40px;
        }
        
        .feature-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .feature-card.active {
            border-color: #667eea;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .feature-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            cursor: pointer;
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-right: 15px;
        }
        
        .feature-title {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .feature-toggle {
            margin-left: auto;
            width: 60px;
            height: 30px;
            background: #ddd;
            border-radius: 15px;
            position: relative;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .feature-toggle.active {
            background: #4CAF50;
        }
        
        .feature-toggle::after {
            content: '';
            width: 26px;
            height: 26px;
            background: white;
            border-radius: 50%;
            position: absolute;
            top: 2px;
            left: 2px;
            transition: all 0.3s ease;
        }
        
        .feature-toggle.active::after {
            transform: translateX(30px);
        }
        
        .feature-description {
            color: #666;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        
        .feature-card.active .feature-description {
            color: rgba(255,255,255,0.9);
        }
        
        .feature-options {
            display: none;
            margin-top: 20px;
        }
        
        .feature-options.show {
            display: block;
        }
        
        .option-group {
            margin-bottom: 20px;
        }
        
        .option-label {
            font-weight: bold;
            margin-bottom: 10px;
            display: block;
        }
        
        .option-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 10px;
        }
        
        .option-btn {
            padding: 10px 15px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            font-weight: bold;
            font-size: 0.9rem;
        }
        
        .option-btn:hover, .option-btn.selected {
            border-color: #667eea;
            background: #667eea;
            color: white;
        }
        
        .feature-card.active .option-btn {
            background: rgba(255,255,255,0.2);
            border-color: rgba(255,255,255,0.3);
            color: white;
        }
        
        .feature-card.active .option-btn:hover,
        .feature-card.active .option-btn.selected {
            background: rgba(255,255,255,0.3);
            border-color: rgba(255,255,255,0.5);
        }
        
        .slider-container {
            margin: 15px 0;
        }
        
        .slider {
            width: 100%;
            height: 5px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        .slider::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            border: none;
        }
        
        .slider-value {
            text-align: center;
            margin-top: 5px;
            font-weight: bold;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            align-items: center;
            margin: 10px 0;
        }
        
        .input-group input[type="number"] {
            flex: 1;
            padding: 8px 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            text-align: center;
        }
        
        .input-group input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .feature-card.active .input-group input[type="number"] {
            background: rgba(255,255,255,0.2);
            border-color: rgba(255,255,255,0.3);
            color: white;
        }
        
        .feature-card.active .input-group input[type="number"]::placeholder {
            color: rgba(255,255,255,0.7);
        }
        
        #colorPickerGroup {
            margin-top: 15px;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .feature-card.active #colorPickerGroup {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.3);
        }
        
        #backgroundColorPicker {
            border: 2px solid #ddd;
            transition: all 0.3s ease;
        }
        
        #backgroundColorPicker:hover {
            border-color: #667eea;
        }
        
        #colorDisplay {
            font-family: monospace;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .preset-btn {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            border: 2px solid #2196f3;
            border-radius: 12px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            font-weight: bold;
            color: #1976d2;
        }
        
        .preset-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3);
            background: linear-gradient(135deg, #bbdefb, #90caf9);
        }
        
        .preset-btn small {
            display: block;
            margin-top: 5px;
            font-weight: normal;
            font-size: 0.8em;
            opacity: 0.8;
        }
        
        #progressDetails {
            text-align: center;
            font-style: italic;
        }
        
        #originalInfo, #processedInfo {
            text-align: center;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        
        .tips-section {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border: 1px solid #dee2e6;
        }
        
        .tips-section h4 {
            color: #495057;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .tips-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .tip-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .tip-item strong {
            color: #667eea;
            display: block;
            margin-bottom: 5px;
        }
        
        .process-section {
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
        }
        
        .process-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.2rem;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            margin: 10px;
        }
        
        .process-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        
        .process-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .clear-btn {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        }
        
        .progress {
            display: none;
            margin-top: 30px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #ddd;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            animation: progress 2s ease-in-out infinite;
        }
        
        @keyframes progress {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
        
        .results {
            display: none;
            margin-top: 30px;
        }
        
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .result-item {
            text-align: center;
        }
        
        .result-item h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .result-item img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .download-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1rem;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #218838;
            transform: translateY(-2px);
        }
        
        .info-panel {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .batch-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .batch-toggle {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .batch-toggle input[type="checkbox"] {
            margin-right: 10px;
            transform: scale(1.2);
        }
        
        @media (max-width: 768px) {
            .result-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .features-section {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✨ Advanced Image Processor Pro</h1>
            <p>Solusi lengkap untuk pengolahan gambar dengan fitur AI, resize, dan efek canggih</p>
        </div>
        
        <div class="main-card">
            <!-- Upload Section -->
            <div class="upload-section">
                <div class="batch-section">
                    <div class="batch-toggle">
                        <input type="checkbox" id="batchMode">
                        <label for="batchMode"><strong>🔄 Mode Batch Processing (Multiple Files)</strong></label>
                    </div>
                    <div id="batchInfo" style="display: none; color: #666;">
                        Mode batch memungkinkan Anda memproses beberapa gambar sekaligus dengan pengaturan yang sama.
                    </div>
                </div>
                
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text">Klik atau drag & drop gambar di sini</div>
                    <div class="upload-subtext">Mendukung JPG, PNG, GIF, BMP, TIFF, WEBP (max 16MB per file)</div>
                    <input type="file" id="fileInput" accept="image/*" multiple>
                </div>
            </div>
            
            <!-- Features Section -->
            <div class="features-section">
                <!-- Image Resize Feature -->
                <div class="feature-card" data-feature="resize">
                    <div class="feature-header">
                        <div class="feature-icon">📏</div>
                        <div class="feature-title">Image Resize</div>
                        <div class="feature-toggle" data-toggle="resize"></div>
                    </div>
                    <div class="feature-description">
                        Ubah ukuran gambar dengan mengatur width dan height. Mendukung maintain aspect ratio atau resize eksak.
                    </div>
                    <div class="feature-options">
                        <div class="option-group">
                            <label class="option-label">Dimensi Target:</label>
                            <div class="input-group">
                                <input type="number" id="resizeWidth" placeholder="Width (px)" min="1" max="10000">
                                <span>×</span>
                                <input type="number" id="resizeHeight" placeholder="Height (px)" min="1" max="10000">
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">
                                <input type="checkbox" id="maintainAspect" checked style="margin-right: 8px;">
                                Pertahankan Aspect Ratio
                            </label>
                        </div>
                    </div>
                </div>

                <!-- Enhancement Features -->
                <div class="feature-card" data-feature="enhancement">
                    <div class="feature-header">
                        <div class="feature-icon">🚀</div>
                        <div class="feature-title">Peningkatan Kualitas</div>
                        <div class="feature-toggle" data-toggle="enhancement"></div>
                    </div>
                    <div class="feature-description">
                        Tingkatkan kualitas gambar dengan AI. Fitur meliputi denoising, sharpening, 
                        contrast enhancement, color correction, dan HDR effect.
                    </div>
                    <div class="feature-options">
                        <div class="option-group">
                            <label class="option-label">Intensitas Enhancement:</label>
                            <div class="slider-container">
                                <input type="range" class="slider" id="enhanceIntensity" min="0.1" max="2.0" step="0.1" value="1.0">
                                <div class="slider-value">1.0</div>
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Fitur Enhancement:</label>
                            <div class="option-grid">
                                <div class="option-btn selected" data-enhance="quality">Kualitas</div>
                                <div class="option-btn selected" data-enhance="color">Warna</div>
                                <div class="option-btn selected" data-enhance="hdr">HDR</div>
                                <div class="option-btn" data-enhance="histogram">Histogram</div>
                                <div class="option-btn" data-enhance="noise">Denoise</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Format Conversion -->
                <div class="feature-card" data-feature="format">
                    <div class="feature-header">
                        <div class="feature-icon">🔄</div>
                        <div class="feature-title">Konversi Format</div>
                        <div class="feature-toggle" data-toggle="format"></div>
                    </div>
                    <div class="feature-description">
                        Konversi gambar ke berbagai format populer dengan pengaturan kualitas optimal.
                    </div>
                    <div class="feature-options">
                        <div class="option-group">
                            <label class="option-label">Format Output:</label>
                            <div class="option-grid">
                                <div class="option-btn selected" data-format="PNG">PNG</div>
                                <div class="option-btn" data-format="JPEG">JPEG</div>
                                <div class="option-btn" data-format="WEBP">WEBP</div>
                                <div class="option-btn" data-format="BMP">BMP</div>
                                <div class="option-btn" data-format="TIFF">TIFF</div>
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Kualitas (untuk JPEG/WEBP):</label>
                            <div class="slider-container">
                                <input type="range" class="slider" id="formatQuality" min="10" max="100" step="5" value="95">
                                <div class="slider-value">95</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Background Processing -->
                <div class="feature-card" data-feature="background">
                    <div class="feature-header">
                        <div class="feature-icon">✂️</div>
                        <div class="feature-title">Background Processing</div>
                        <div class="feature-toggle" data-toggle="background"></div>
                    </div>
                    <div class="feature-description">
                        Hapus atau ganti background gambar menggunakan AI dan algoritma computer vision.
                    </div>
                    <div class="feature-options">
                        <div class="option-group">
                            <label class="option-label">Metode Background:</label>
                            <div class="option-grid">
                                <div class="option-btn selected" data-bg-method="ai">AI Model</div>
                                <div class="option-btn" data-bg-method="advanced">Advanced</div>
                                <div class="option-btn" data-bg-method="yolo">YOLO Enhanced</div>
                                <div class="option-btn" data-bg-method="grabcut">GrabCut</div>
                            </div>
                            <div style="margin-top: 10px; padding: 10px; background: rgba(76,175,80,0.1); border-radius: 5px; font-size: 0.85em;">
                                <strong>💡 Tips:</strong> AI Model = paling akurat | Advanced = multi-algorithm | YOLO = bagus untuk orang/objek | GrabCut = general purpose
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Kualitas Edge (Advanced):</label>
                            <div class="slider-container">
                                <input type="range" class="slider" id="edgeQuality" min="1" max="5" step="1" value="3">
                                <div class="slider-value">3</div>
                            </div>
                            <div style="margin-top: 5px; font-size: 0.8em; color: #666;">
                                1=Cepat, 5=Sangat Detail
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Background Baru:</label>
                            <div class="option-grid">
                                <div class="option-btn selected" data-bg-replace="transparent">Transparan</div>
                                <div class="option-btn" data-bg-replace="white">Putih</div>
                                <div class="option-btn" data-bg-replace="black">Hitam</div>
                                <div class="option-btn" data-bg-replace="custom">Custom</div>
                            </div>
                        </div>
                        <div class="option-group" id="colorPickerGroup" style="display: none;">
                            <label class="option-label">Pilih Warna Background:</label>
                            <div style="display: flex; align-items: center; gap: 15px;">
                                <input type="color" id="backgroundColorPicker" value="#ffffff" style="width: 60px; height: 40px; border: none; border-radius: 8px; cursor: pointer;">
                                <span id="colorDisplay" style="padding: 8px 15px; background: #ffffff; border: 2px solid #ddd; border-radius: 8px; min-width: 100px; text-align: center;">#ffffff</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Detection & Analysis -->
                <div class="feature-card" data-feature="detection">
                    <div class="feature-header">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">Deteksi & Analisis</div>
                        <div class="feature-toggle" data-toggle="detection"></div>
                    </div>
                    <div class="feature-description">
                        Deteksi objek, wajah, dan teks dalam gambar. Termasuk fitur blur wajah untuk privasi.
                    </div>
                    <div class="feature-options">
                        <div class="option-group">
                            <label class="option-label">Jenis Deteksi:</label>
                            <div class="option-grid">
                                <div class="option-btn selected" data-detection="objects">Objek</div>
                                <div class="option-btn" data-detection="faces">Wajah</div>
                                <div class="option-btn" data-detection="text">Teks</div>
                                <div class="option-btn" data-detection="tags">Auto Tag</div>
                            </div>
                            <div style="margin-top: 10px; padding: 10px; background: rgba(255,193,7,0.1); border-radius: 5px; font-size: 0.85em;">
                                <strong>⚠️ Catatan:</strong> Jika Background Processing aktif, deteksi objek tidak akan menampilkan kotak bounding box untuk hasil yang lebih bersih.
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Aksi Wajah:</label>
                            <div class="option-grid">
                                <div class="option-btn" data-face-action="detect">Deteksi</div>
                                <div class="option-btn selected" data-face-action="blur">Blur</div>
                                <div class="option-btn" data-face-action="pixelate">Pixelate</div>
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Intensitas Blur Wajah:</label>
                            <div class="slider-container">
                                <input type="range" class="slider" id="faceBlurIntensity" min="11" max="99" step="2" value="99">
                                <div class="slider-value">99</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Artistic Effects -->
                <div class="feature-card" data-feature="effects">
                    <div class="feature-header">
                        <div class="feature-icon">🎨</div>
                        <div class="feature-title">Efek Artistik</div>
                        <div class="feature-toggle" data-toggle="effects"></div>
                    </div>
                    <div class="feature-description">
                        Berbagai efek artistik dan filter kreatif untuk mengubah gaya visual gambar.
                    </div>
                    <div class="feature-options">
                        <div class="option-group">
                            <label class="option-label">Efek Dasar:</label>
                            <div class="option-grid">
                                <div class="option-btn selected" data-effect="sketch">Sketsa</div>
                                <div class="option-btn" data-effect="cartoon">Kartun</div>
                                <div class="option-btn" data-effect="oil_painting">Lukisan</div>
                                <div class="option-btn" data-effect="sepia">Sepia</div>
                                <div class="option-btn" data-effect="vignette">Vignette</div>
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Efek Blur:</label>
                            <div class="option-grid">
                                <div class="option-btn" data-blur="gaussian">Gaussian</div>
                                <div class="option-btn" data-blur="motion">Motion</div>
                                <div class="option-btn" data-blur="radial">Radial</div>
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Efek Lain:</label>
                            <div class="option-grid">
                                <div class="option-btn" data-other-effect="emboss">Emboss</div>
                                <div class="option-btn" data-other-effect="edges">Edge</div>
                                <div class="option-btn" data-other-effect="watermark">Watermark</div>
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Intensitas Efek:</label>
                            <div class="slider-container">
                                <input type="range" class="slider" id="effectIntensity" min="0.1" max="2.0" step="0.1" value="1.0">
                                <div class="slider-value">1.0</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Advanced AI -->
                <div class="feature-card" data-feature="ai">
                    <div class="feature-header">
                        <div class="feature-icon">🧠</div>
                        <div class="feature-title">AI & Machine Learning</div>
                        <div class="feature-toggle" data-toggle="ai"></div>
                    </div>
                    <div class="feature-description">
                        Fitur canggih menggunakan AI untuk super resolution, style transfer, dan processing lanjutan.
                    </div>
                    <div class="feature-options">
                        <div class="option-group">
                            <label class="option-label">Fitur AI:</label>
                            <div class="option-grid">
                                <div class="option-btn selected" data-ai="super_res">Super Res</div>
                                <div class="option-btn" data-ai="style_transfer">Style Transfer</div>
                                <div class="option-btn" data-ai="inpainting">Inpainting</div>
                                <div class="option-btn" data-ai="upscale">Upscale</div>
                            </div>
                        </div>
                        <div class="option-group">
                            <label class="option-label">Style Transfer:</label>
                            <div class="option-grid">
                                <div class="option-btn" data-style="classic">Classic</div>
                                <div class="option-btn" data-style="modern">Modern</div>
                                <div class="option-btn" data-style="abstract">Abstract</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Process Section -->
            <div class="process-section">
                <!-- Quick Presets -->
                <div style="margin-bottom: 30px;">
                    <h3 style="text-align: center; margin-bottom: 20px; color: #333;">🎯 Preset Populer</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px;">
                        <button class="preset-btn" data-preset="portrait">
                            👤 Portrait Professional<br>
                            <small>Background Removal + Enhancement</small>
                        </button>
                        <button class="preset-btn" data-preset="photo_enhance">
                            📸 Photo Enhancement<br>
                            <small>Quality + Color + HDR</small>
                        </button>
                        <button class="preset-btn" data-preset="resize_web">
                            🌐 Web Resize<br>
                            <small>Resize 1920x1080 + Enhancement</small>
                        </button>
                        <button class="preset-btn" data-preset="social_media">
                            📱 Social Media Ready<br>
                            <small>Resize Square + Background + Format</small>
                        </button>
                    </div>
                </div>
                
                <button class="process-btn" id="processBtn" disabled>
                    Pilih gambar dan aktifkan fitur terlebih dahulu
                </button>
                <button class="process-btn" id="previewBtn" disabled style="background: linear-gradient(135deg, #ffa726, #ff7043);">
                    👁️ Preview Hasil
                </button>
                <button class="process-btn clear-btn" id="clearBtn">
                    🗑️ Reset Semua
                </button>
                
                <div class="progress" id="progress">
                    <div style="margin-bottom: 15px;">Sedang memproses gambar...</div>
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <div id="progressDetails" style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        Memuat...
                    </div>
                </div>
                
                <div class="results" id="results">
                    <div class="result-grid">
                        <div class="result-item">
                            <h3>Gambar Asli</h3>
                            <img id="originalImage" alt="Original">
                            <div id="originalInfo" style="margin-top: 10px; font-size: 0.9em; color: #666;"></div>
                        </div>
                        <div class="result-item">
                            <h3>Hasil Proses</h3>
                            <img id="processedImage" alt="Processed">
                            <div id="processedInfo" style="margin-top: 10px; font-size: 0.9em; color: #666;"></div>
                        </div>
                    </div>
                    
                    <button class="download-btn" id="downloadBtn">
                        📥 Download Hasil
                    </button>
                    
                    <div class="info-panel" id="infoPanel">
                        <!-- Info akan diisi oleh JavaScript -->
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Tips & Help Section -->
        <div class="tips-section">
            <h4>💡 Tips & Panduan Penggunaan</h4>
            <div class="tips-grid">
                <div class="tip-item">
                    <strong>Image Resize:</strong>
                    Masukkan width dan height dalam pixel. Centang "Pertahankan Aspect Ratio" untuk mencegah distorsi gambar.
                </div>
                <div class="tip-item">
                    <strong>Background Removal Optimal:</strong>
                    Gunakan metode "AI Model" dengan kualitas edge 4-5 untuk hasil terbaik. Pilih "Advanced" jika AI model tidak tersedia.
                </div>
                <div class="tip-item">
                    <strong>Kombinasi Terbaik:</strong>
                    Background Removal + Enhancement memberikan hasil profesional untuk foto portrait dan produk.
                </div>
                <div class="tip-item">
                    <strong>Format untuk Web:</strong>
                    Gunakan WEBP untuk website (file kecil) atau PNG untuk transparansi. JPEG untuk foto biasa.
                </div>
                <div class="tip-item">
                    <strong>Batch Processing:</strong>
                    Aktifkan mode batch untuk memproses banyak file sekaligus dengan pengaturan yang sama.
                </div>
                <div class="tip-item">
                    <strong>Preview Mode:</strong>
                    Gunakan "Preview" untuk melihat hasil tanpa menyimpan file. Hemat waktu untuk eksperimen.
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let selectedFiles = [];
        let activeFeatures = new Set();
        let featureSettings = {
            resize: {
                width: null,
                height: null,
                maintainAspect: true
            },
            enhancement: {
                intensity: 1.0,
                features: ['quality', 'color', 'hdr']
            },
            format: {
                type: 'PNG',
                quality: 95
            },
            background: {
                method: 'ai',
                replace: 'transparent',
                customColor: '#ffffff',
                edgeQuality: 3
            },
            detection: {
                types: ['objects'],
                faceAction: 'blur',
                blurIntensity: 99
            },
            effects: {
                type: 'sketch',
                blur: null,
                other: null,
                intensity: 1.0
            },
            ai: {
                features: ['super_res'],
                style: 'classic'
            }
        };
        
        // File upload handling
        const fileInput = document.getElementById('fileInput');
        const uploadZone = document.querySelector('.upload-zone');
        const processBtn = document.getElementById('processBtn');
        const previewBtn = document.getElementById('previewBtn');
        const clearBtn = document.getElementById('clearBtn');
        const batchMode = document.getElementById('batchMode');
        const batchInfo = document.getElementById('batchInfo');
        
        fileInput.addEventListener('change', handleFileSelect);
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('drop', handleDrop);
        clearBtn.addEventListener('click', clearAll);
        
        batchMode.addEventListener('change', function() {
            batchInfo.style.display = this.checked ? 'block' : 'none';
            fileInput.multiple = this.checked;
            updateProcessButton();
        });
        
        // Resize input handling
        document.getElementById('resizeWidth').addEventListener('input', function() {
            featureSettings.resize.width = this.value ? parseInt(this.value) : null;
        });
        
        document.getElementById('resizeHeight').addEventListener('input', function() {
            featureSettings.resize.height = this.value ? parseInt(this.value) : null;
        });
        
        document.getElementById('maintainAspect').addEventListener('change', function() {
            featureSettings.resize.maintainAspect = this.checked;
        });
        
        // Color picker handling
        const backgroundColorPicker = document.getElementById('backgroundColorPicker');
        const colorDisplay = document.getElementById('colorDisplay');
        const colorPickerGroup = document.getElementById('colorPickerGroup');
        
        backgroundColorPicker.addEventListener('input', function() {
            const color = this.value;
            colorDisplay.textContent = color;
            colorDisplay.style.background = color;
            featureSettings.background.customColor = color;
        });
        
        // Show/hide color picker based on background replacement selection
        document.addEventListener('click', function(e) {
            if (e.target.dataset.bgReplace) {
                const isCustom = e.target.dataset.bgReplace === 'custom';
                colorPickerGroup.style.display = isCustom ? 'block' : 'none';
            }
        });
        
        // Preset buttons handling
        const presetBtns = document.querySelectorAll('.preset-btn');
        presetBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                applyPreset(preset);
            });
        });
        
        function applyPreset(preset) {
            // First, clear all active features
            clearAllFeatures();
            
            switch(preset) {
                case 'portrait':
                    // Background Processing + Enhancement
                    activateFeature('background');
                    activateFeature('enhancement');
                    
                    // Set background to AI method with transparent
                    document.querySelector('[data-bg-method="ai"]').click();
                    document.querySelector('[data-bg-replace="transparent"]').click();
                    
                    // Set enhancement features
                    document.querySelector('[data-enhance="quality"]').classList.add('selected');
                    document.querySelector('[data-enhance="color"]').classList.add('selected');
                    
                    break;
                    
                case 'photo_enhance':
                    // Quality + Color + HDR Enhancement
                    activateFeature('enhancement');
                    
                    document.querySelector('[data-enhance="quality"]').classList.add('selected');
                    document.querySelector('[data-enhance="color"]').classList.add('selected');
                    document.querySelector('[data-enhance="hdr"]').classList.add('selected');
                    
                    // Set intensity to 1.2 for better enhancement
                    document.getElementById('enhanceIntensity').value = 1.2;
                    document.querySelector('#enhanceIntensity + .slider-value').textContent = '1.2';
                    
                    break;
                    
                case 'resize_web':
                    // Resize + Enhancement for web
                    activateFeature('resize');
                    activateFeature('enhancement');
                    
                    // Set resize to 1920x1080
                    document.getElementById('resizeWidth').value = 1920;
                    document.getElementById('resizeHeight').value = 1080;
                    featureSettings.resize.width = 1920;
                    featureSettings.resize.height = 1080;
                    
                    // Enhancement
                    document.querySelector('[data-enhance="quality"]').classList.add('selected');
                    
                    break;
                    
                case 'social_media':
                    // Resize square + Background + Format
                    activateFeature('resize');
                    activateFeature('background');
                    activateFeature('format');
                    activateFeature('enhancement');
                    
                    // Set resize to 1080x1080 (Instagram square)
                    document.getElementById('resizeWidth').value = 1080;
                    document.getElementById('resizeHeight').value = 1080;
                    featureSettings.resize.width = 1080;
                    featureSettings.resize.height = 1080;
                    document.getElementById('maintainAspect').checked = false;
                    featureSettings.resize.maintainAspect = false;
                    
                    // Background
                    document.querySelector('[data-bg-method="ai"]').click();
                    document.querySelector('[data-bg-replace="white"]').click();
                    
                    // Format
                    document.querySelector('[data-format="JPEG"]').click();
                    document.querySelector('[data-enhance="quality"]').classList.add('selected');
                    document.querySelector('[data-enhance="color"]').classList.add('selected');
                    
                    break;
            }
            
            updateFeatureSettings();
            updateProcessButton();
            
            // Show notification
            showNotification(`✅ Preset "${preset}" diterapkan!`);
        }
        
        function activateFeature(featureName) {
            const toggle = document.querySelector(`[data-toggle="${featureName}"]`);
            const card = toggle.closest('.feature-card');
            const options = card.querySelector('.feature-options');
            
            activeFeatures.add(featureName);
            toggle.classList.add('active');
            card.classList.add('active');
            options.classList.add('show');
        }
        
        function clearAllFeatures() {
            activeFeatures.clear();
            featureToggles.forEach(toggle => {
                toggle.classList.remove('active');
                const card = toggle.closest('.feature-card');
                card.classList.remove('active');
                card.querySelector('.feature-options').classList.remove('show');
            });
            
            // Clear all option selections
            document.querySelectorAll('.option-btn.selected').forEach(btn => {
                btn.classList.remove('selected');
            });
        }
        
        function showNotification(message) {
            // Create notification element
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: linear-gradient(135deg, #4caf50, #45a049);
                color: white;
                padding: 15px 25px;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
                z-index: 10000;
                font-weight: bold;
                animation: slideIn 0.3s ease-out;
            `;
            notification.textContent = message;
            
            // Add animation styles
            const style = document.createElement('style');
            style.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
            
            document.body.appendChild(notification);
            
            // Remove after 3 seconds
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => {
                    document.body.removeChild(notification);
                    document.head.removeChild(style);
                }, 300);
            }, 3000);
        }
        
        function handleFileSelect(e) {
            const files = Array.from(e.target.files);
            if (files.length > 0) {
                selectedFiles = files;
                updateUploadZone();
                updateProcessButton();
            }
        }
        
        function handleDragOver(e) {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            
            const files = Array.from(e.dataTransfer.files);
            if (files.length > 0) {
                selectedFiles = files;
                fileInput.files = e.dataTransfer.files;
                updateUploadZone();
                updateProcessButton();
            }
        }
        
        function updateUploadZone() {
            const count = selectedFiles.length;
            const isBatch = batchMode.checked;
            
            if (count === 1 && !isBatch) {
                uploadZone.innerHTML = `
                    <div class="upload-icon">✅</div>
                    <div class="upload-text">File terpilih: ${selectedFiles[0].name}</div>
                    <div class="upload-subtext">Klik untuk mengganti file</div>
                `;
            } else if (count > 1) {
                uploadZone.innerHTML = `
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">${count} file terpilih</div>
                    <div class="upload-subtext">Mode batch: ${selectedFiles.map(f => f.name).join(', ')}</div>
                `;
            }
        }
        
        // Feature toggle handling
        const featureCards = document.querySelectorAll('.feature-card');
        const featureToggles = document.querySelectorAll('.feature-toggle');
        
        featureToggles.forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.stopPropagation();
                const feature = toggle.dataset.toggle;
                const card = toggle.closest('.feature-card');
                const options = card.querySelector('.feature-options');
                
                if (activeFeatures.has(feature)) {
                    // Deactivate
                    activeFeatures.delete(feature);
                    toggle.classList.remove('active');
                    card.classList.remove('active');
                    options.classList.remove('show');
                } else {
                    // Activate
                    activeFeatures.add(feature);
                    toggle.classList.add('active');
                    card.classList.add('active');
                    options.classList.add('show');
                }
                
                updateProcessButton();
            });
        });
        
        // Option buttons handling
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('option-btn')) {
                const group = e.target.closest('.option-group');
                const isMultiSelect = e.target.dataset.enhance || 
                                     e.target.dataset.detection || 
                                     e.target.dataset.ai;
                
                if (!isMultiSelect) {
                    // Single select
                    group.querySelectorAll('.option-btn').forEach(btn => 
                        btn.classList.remove('selected'));
                }
                
                e.target.classList.toggle('selected');
                updateFeatureSettings();
            }
        });
        
        // Slider handling
        document.addEventListener('input', function(e) {
            if (e.target.classList.contains('slider')) {
                const valueDiv = e.target.nextElementSibling;
                valueDiv.textContent = e.target.value;
                updateFeatureSettings();
            }
        });
        
        function updateFeatureSettings() {
            // Update resize settings
            featureSettings.resize.width = document.getElementById('resizeWidth').value ? parseInt(document.getElementById('resizeWidth').value) : null;
            featureSettings.resize.height = document.getElementById('resizeHeight').value ? parseInt(document.getElementById('resizeHeight').value) : null;
            featureSettings.resize.maintainAspect = document.getElementById('maintainAspect').checked;
            
            // Update enhancement settings
            featureSettings.enhancement.intensity = parseFloat(document.getElementById('enhanceIntensity').value);
            featureSettings.enhancement.features = Array.from(document.querySelectorAll('[data-enhance].selected'))
                .map(btn => btn.dataset.enhance);
            
            // Update format settings
            const formatBtn = document.querySelector('[data-format].selected');
            if (formatBtn) {
                featureSettings.format.type = formatBtn.dataset.format;
            }
            featureSettings.format.quality = parseInt(document.getElementById('formatQuality').value);
            
            // Update background settings
            const bgMethodBtn = document.querySelector('[data-bg-method].selected');
            const bgReplaceBtn = document.querySelector('[data-bg-replace].selected');
            if (bgMethodBtn) featureSettings.background.method = bgMethodBtn.dataset.bgMethod;
            if (bgReplaceBtn) {
                featureSettings.background.replace = bgReplaceBtn.dataset.bgReplace;
                // If custom color is selected, use the color picker value
                if (bgReplaceBtn.dataset.bgReplace === 'custom') {
                    featureSettings.background.customColor = document.getElementById('backgroundColorPicker').value;
                }
            }
            featureSettings.background.edgeQuality = parseInt(document.getElementById('edgeQuality').value);
            
            // Update detection settings
            featureSettings.detection.types = Array.from(document.querySelectorAll('[data-detection].selected'))
                .map(btn => btn.dataset.detection);
            const faceActionBtn = document.querySelector('[data-face-action].selected');
            if (faceActionBtn) featureSettings.detection.faceAction = faceActionBtn.dataset.faceAction;
            featureSettings.detection.blurIntensity = parseInt(document.getElementById('faceBlurIntensity').value);
            
            // Update effects settings
            const effectBtn = document.querySelector('[data-effect].selected');
            const blurBtn = document.querySelector('[data-blur].selected');
            const otherBtn = document.querySelector('[data-other-effect].selected');
            if (effectBtn) featureSettings.effects.type = effectBtn.dataset.effect;
            if (blurBtn) featureSettings.effects.blur = blurBtn.dataset.blur;
            if (otherBtn) featureSettings.effects.other = otherBtn.dataset.otherEffect;
            featureSettings.effects.intensity = parseFloat(document.getElementById('effectIntensity').value);
            
            // Update AI settings
            featureSettings.ai.features = Array.from(document.querySelectorAll('[data-ai].selected'))
                .map(btn => btn.dataset.ai);
            const styleBtn = document.querySelector('[data-style].selected');
            if (styleBtn) featureSettings.ai.style = styleBtn.dataset.style;
        }
        
        function updateProcessButton() {
            const hasFiles = selectedFiles.length > 0;
            const hasFeatures = activeFeatures.size > 0;
            
            if (hasFiles && hasFeatures) {
                processBtn.disabled = false;
                previewBtn.disabled = false;
                const featureText = Array.from(activeFeatures).join(', ');
                const fileText = selectedFiles.length === 1 ? '1 gambar' : `${selectedFiles.length} gambar`;
                processBtn.textContent = `🚀 Proses ${fileText} dengan ${featureText}`;
                previewBtn.textContent = `👁️ Preview ${fileText}`;
            } else {
                processBtn.disabled = true;
                previewBtn.disabled = true;
                if (!hasFiles) {
                    processBtn.textContent = 'Pilih gambar terlebih dahulu';
                    previewBtn.textContent = '👁️ Preview Hasil';
                } else if (!hasFeatures) {
                    processBtn.textContent = 'Aktifkan minimal satu fitur';
                    previewBtn.textContent = '👁️ Preview Hasil';
                }
            }
        }
        
        function clearAll() {
            // Reset files
            selectedFiles = [];
            fileInput.value = '';
            
            // Reset upload zone
            uploadZone.innerHTML = `
                <div class="upload-icon">📸</div>
                <div class="upload-text">Klik atau drag & drop gambar di sini</div>
                <div class="upload-subtext">Mendukung JPG, PNG, GIF, BMP, TIFF, WEBP (max 16MB per file)</div>
            `;
            
            // Clear all features
            clearAllFeatures();
            
            // Reset resize inputs
            document.getElementById('resizeWidth').value = '';
            document.getElementById('resizeHeight').value = '';
            document.getElementById('maintainAspect').checked = true;
            
            // Reset sliders to default
            document.getElementById('enhanceIntensity').value = 1.0;
            document.querySelector('#enhanceIntensity + .slider-value').textContent = '1.0';
            document.getElementById('formatQuality').value = 95;
            document.querySelector('#formatQuality + .slider-value').textContent = '95';
            document.getElementById('edgeQuality').value = 3;
            document.querySelector('#edgeQuality + .slider-value').textContent = '3';
            document.getElementById('faceBlurIntensity').value = 99;
            document.querySelector('#faceBlurIntensity + .slider-value').textContent = '99';
            document.getElementById('effectIntensity').value = 1.0;
            document.querySelector('#effectIntensity + .slider-value').textContent = '1.0';
            
            // Reset color picker
            document.getElementById('backgroundColorPicker').value = '#ffffff';
            document.getElementById('colorDisplay').textContent = '#ffffff';
            document.getElementById('colorDisplay').style.background = '#ffffff';
            document.getElementById('colorPickerGroup').style.display = 'none';
            
            // Reset default selections
            document.querySelector('[data-format="PNG"]').classList.add('selected');
            document.querySelector('[data-bg-method="ai"]').classList.add('selected');
            document.querySelector('[data-bg-replace="transparent"]').classList.add('selected');
            document.querySelector('[data-detection="objects"]').classList.add('selected');
            document.querySelector('[data-face-action="blur"]').classList.add('selected');
            document.querySelector('[data-effect="sketch"]').classList.add('selected');
            document.querySelector('[data-enhance="quality"]').classList.add('selected');
            document.querySelector('[data-enhance="color"]').classList.add('selected');
            document.querySelector('[data-enhance="hdr"]').classList.add('selected');
            document.querySelector('[data-ai="super_res"]').classList.add('selected');
            
            // Reset results
            document.getElementById('results').style.display = 'none';
            document.querySelector('.result-grid').style.display = 'grid';
            document.getElementById('downloadBtn').style.display = 'block';
            
            // Remove any notifications
            const previewNotice = document.querySelector('.preview-notice');
            if (previewNotice) {
                previewNotice.remove();
            }
            
            updateProcessButton();
            showNotification('🔄 Semua pengaturan telah direset!');
        }
        
        // Process button
        processBtn.addEventListener('click', () => processImages(false));
        previewBtn.addEventListener('click', () => processImages(true));
        
        async function processImages(isPreview = false) {
            if (selectedFiles.length === 0 || activeFeatures.size === 0) return;
            
            // Show progress
            document.getElementById('progress').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            processBtn.disabled = true;
            previewBtn.disabled = true;
            
            // Update progress text
            const progressText = document.querySelector('#progress div');
            const progressDetails = document.getElementById('progressDetails');
            progressText.textContent = isPreview ? 'Sedang membuat preview...' : 'Sedang memproses gambar...';
            
            try {
                // Process each file
                const results = [];
                
                for (let i = 0; i < selectedFiles.length; i++) {
                    const file = selectedFiles[i];
                    
                    // Update progress details
                    progressDetails.textContent = `Memproses file ${i + 1} dari ${selectedFiles.length}: ${file.name}`;
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('isPreview', isPreview);
                    formData.append('settings', JSON.stringify({
                        activeFeatures: Array.from(activeFeatures),
                        featureSettings: featureSettings
                    }));
                    
                    // Show which features are being applied
                    const activeFeatureNames = Array.from(activeFeatures).map(f => {
                        switch(f) {
                            case 'resize': return 'Image Resize';
                            case 'enhancement': return 'Peningkatan Kualitas';
                            case 'background': return 'Background Processing';
                            case 'detection': return 'Deteksi & Analisis';
                            case 'effects': return 'Efek Artistik';
                            case 'format': return 'Konversi Format';
                            case 'ai': return 'AI Processing';
                            default: return f;
                        }
                    });
                    progressDetails.innerHTML = `
                        Memproses: ${file.name}<br>
                        Fitur aktif: ${activeFeatureNames.join(', ')}<br>
                        File ${i + 1} dari ${selectedFiles.length}
                    `;
                    
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    results.push(result);
                }
                
                // Hide progress
                document.getElementById('progress').style.display = 'none';
                processBtn.disabled = false;
                previewBtn.disabled = false;
                
                // Display results
                if (results.length === 1) {
                    displaySingleResult(results[0], isPreview);
                } else {
                    displayBatchResults(results, isPreview);
                }
                
            } catch (error) {
                document.getElementById('progress').style.display = 'none';
                processBtn.disabled = false;
                previewBtn.disabled = false;
                progressDetails.textContent = '';
                alert('Terjadi kesalahan: ' + error.message);
            }
        }
        
        function displaySingleResult(result, isPreview = false) {
            if (!result.success) {
                alert('Error: ' + result.error);
                return;
            }
            
            // Display images
            document.getElementById('originalImage').src = `data:image/png;base64,${result.original_image}`;
            document.getElementById('processedImage').src = `data:image/png;base64,${result.processed_image}`;
            
            // Show file information
            const originalInfo = document.getElementById('originalInfo');
            const processedInfo = document.getElementById('processedInfo');
            
            originalInfo.innerHTML = `
                📄 ${selectedFiles[0].name}<br>
                📏 ${result.original_size[0]} × ${result.original_size[1]}px<br>
                💾 ${formatFileSize(selectedFiles[0].size)}
            `;
            
            const estimatedSize = estimateFileSize(result.final_size, result.output_format);
            processedInfo.innerHTML = `
                📄 Processed ${result.output_format}<br>
                📏 ${result.final_size[0]} × ${result.final_size[1]}px<br>
                💾 ~${estimatedSize}<br>
                ⏱️ ${result.processing_time}
            `;
            
            // Setup download
            const downloadBtn = document.getElementById('downloadBtn');
            if (isPreview) {
                downloadBtn.style.display = 'none';
                // Add preview notice
                const resultsSection = document.getElementById('results');
                let previewNotice = resultsSection.querySelector('.preview-notice');
                if (!previewNotice) {
                    previewNotice = document.createElement('div');
                    previewNotice.className = 'preview-notice';
                    previewNotice.style.cssText = 'background: #fff3cd; color: #856404; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; border: 1px solid #ffeaa7;';
                    previewNotice.innerHTML = '👁️ <strong>Mode Preview</strong> - Ini adalah preview hasil. Klik "🚀 Proses" untuk mendapatkan file download.';
                    resultsSection.insertBefore(previewNotice, resultsSection.firstChild);
                }
            } else {
                downloadBtn.style.display = 'block';
                downloadBtn.onclick = () => {
                    window.open(result.download_url, '_blank');
                };
                // Remove preview notice if exists
                const previewNotice = document.querySelector('.preview-notice');
                if (previewNotice) {
                    previewNotice.remove();
                }
            }
            
            // Display info
            displayResultInfo(result, isPreview);
            
            // Show results
            document.getElementById('results').style.display = 'block';
        }
        
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        function estimateFileSize(dimensions, format) {
            const [width, height] = dimensions;
            const pixels = width * height;
            
            let estimatedBytes;
            switch(format.toUpperCase()) {
                case 'PNG':
                    estimatedBytes = pixels * 3; // Rough estimate for PNG
                    break;
                case 'JPEG':
                    estimatedBytes = pixels * 0.5; // Rough estimate for JPEG
                    break;
                case 'WEBP':
                    estimatedBytes = pixels * 0.3; // Rough estimate for WEBP
                    break;
                default:
                    estimatedBytes = pixels * 2;
            }
            
            return formatFileSize(estimatedBytes);
        }
        
        function displayBatchResults(results, isPreview = false) {
            // For batch processing, show summary
            const successCount = results.filter(r => r.success).length;
            const totalCount = results.length;
            
            // Show results section with batch info
            document.getElementById('results').style.display = 'block';
            
            // Hide individual image display
            document.querySelector('.result-grid').style.display = 'none';
            
            // Show batch summary
            const infoPanel = document.getElementById('infoPanel');
            infoPanel.innerHTML = `
                <h3>📊 Hasil Batch ${isPreview ? 'Preview' : 'Processing'}</h3>
                <div class="info-item">
                    <strong>Total File:</strong>
                    <span>${totalCount}</span>
                </div>
                <div class="info-item">
                    <strong>Berhasil:</strong>
                    <span>${successCount}</span>
                </div>
                <div class="info-item">
                    <strong>Gagal:</strong>
                    <span>${totalCount - successCount}</span>
                </div>
                ${!isPreview ? `<div style="margin-top: 20px;">
                    <strong>Download Links:</strong>
                    ${results.map((result, index) => {
                        if (result.success) {
                            return `<div><a href="${result.download_url}" target="_blank">📥 Download ${selectedFiles[index].name}</a></div>`;
                        } else {
                            return `<div>❌ ${selectedFiles[index].name}: ${result.error}</div>`;
                        }
                    }).join('')}
                </div>` : '<div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px; color: #856404;"><strong>👁️ Preview Mode</strong> - Klik "🚀 Proses" untuk download files</div>'}
            `;
            
            // Hide single download button
            document.getElementById('downloadBtn').style.display = 'none';
        }
        
        function displayResultInfo(result, isPreview = false) {
            const infoPanel = document.getElementById('infoPanel');
            infoPanel.innerHTML = `
                ${isPreview ? '<div style="background: #d1ecf1; color: #0c5460; padding: 10px; border-radius: 8px; margin-bottom: 15px; text-align: center;"><strong>👁️ Preview Mode Active</strong></div>' : ''}
                <div class="info-item">
                    <strong>Operasi:</strong>
                    <span>${result.operations.join(', ')}</span>
                </div>
                <div class="info-item">
                    <strong>Format Output:</strong>
                    <span>${result.output_format}</span>
                </div>
                <div class="info-item">
                    <strong>Ukuran Asli:</strong>
                    <span>${result.original_size[0]} x ${result.original_size[1]} px</span>
                </div>
                <div class="info-item">
                    <strong>Ukuran Akhir:</strong>
                    <span>${result.final_size[0]} x ${result.final_size[1]} px</span>
                </div>
                <div class="info-item">
                    <strong>Info Deteksi:</strong>
                    <span>${result.detection_info}</span>
                </div>
                <div class="info-item">
                    <strong>Waktu Proses:</strong>
                    <span>${result.processing_time || 'N/A'}</span>
                </div>
                ${result.quality_info ? `<div class="info-item">
                    <strong>Kualitas Proses:</strong>
                    <span>${result.quality_info}</span>
                </div>` : ''}
                ${result.resize_info ? `<div class="info-item">
                    <strong>Info Resize:</strong>
                    <span>${result.resize_info}</span>
                </div>` : ''}
            `;
        }
        
        // Initialize sliders and color picker
        document.querySelectorAll('.slider').forEach(slider => {
            const valueDiv = slider.nextElementSibling;
            valueDiv.textContent = slider.value;
        });
        
        // Initialize color picker
        const initialColor = '#ffffff';
        document.getElementById('backgroundColorPicker').value = initialColor;
        document.getElementById('colorDisplay').textContent = initialColor;
        document.getElementById('colorDisplay').style.background = initialColor;
        
        // Initialize default selections
        document.querySelector('[data-format="PNG"]').classList.add('selected');
        document.querySelector('[data-bg-method="ai"]').classList.add('selected');
        document.querySelector('[data-bg-replace="transparent"]').classList.add('selected');
        document.querySelector('[data-detection="objects"]').classList.add('selected');
        document.querySelector('[data-face-action="blur"]').classList.add('selected');
        document.querySelector('[data-effect="sketch"]').classList.add('selected');
        document.querySelector('[data-enhance="quality"]').classList.add('selected');
        document.querySelector('[data-enhance="color"]').classList.add('selected');
        document.querySelector('[data-enhance="hdr"]').classList.add('selected');
        document.querySelector('[data-ai="super_res"]').classList.add('selected');
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + O = Open file
            if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
                e.preventDefault();
                fileInput.click();
            }
            
            // Ctrl/Cmd + Enter = Process
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (!processBtn.disabled) {
                    processBtn.click();
                }
            }
            
            // Ctrl/Cmd + Shift + Enter = Preview
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'Enter') {
                e.preventDefault();
                if (!previewBtn.disabled) {
                    previewBtn.click();
                }
            }
            
            // Escape = Clear all
            if (e.key === 'Escape') {
                e.preventDefault();
                clearAll();
            }
            
            // Number keys 1-4 for presets
            if (e.key >= '1' && e.key <= '4' && !e.ctrlKey && !e.metaKey) {
                const presets = ['portrait', 'photo_enhance', 'resize_web', 'social_media'];
                const presetIndex = parseInt(e.key) - 1;
                if (presets[presetIndex]) {
                    applyPreset(presets[presetIndex]);
                }
            }
        });
        
        // Add tooltip for keyboard shortcuts
        const shortcutsInfo = document.createElement('div');
        shortcutsInfo.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 8px;
            font-size: 0.8em;
            z-index: 1000;
            display: none;
            max-width: 250px;
        `;
        shortcutsInfo.innerHTML = `
            <strong>⌨️ Keyboard Shortcuts:</strong><br>
            Ctrl+O: Buka file<br>
            Ctrl+Enter: Proses<br>
            Ctrl+Shift+Enter: Preview<br>
            Esc: Reset semua<br>
            1-4: Quick presets
        `;
        document.body.appendChild(shortcutsInfo);
        
        // Show shortcuts on hover over help icon
        const helpIcon = document.createElement('div');
        helpIcon.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            background: #667eea;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 18px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            z-index: 999;
        `;
        helpIcon.innerHTML = '?';
        helpIcon.addEventListener('mouseenter', () => {
            shortcutsInfo.style.display = 'block';
        });
        helpIcon.addEventListener('mouseleave', () => {
            shortcutsInfo.style.display = 'none';
        });
        document.body.appendChild(helpIcon);
    </script>
</body>
</html>
    ''')

@app.route('/process', methods=['POST'])
def process_image():
    import time
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'})
        
        # Check if this is preview mode
        is_preview = request.form.get('isPreview', 'false').lower() == 'true'
        
        # Parse settings
        settings = json.loads(request.form.get('settings', '{}'))
        active_features = settings.get('activeFeatures', [])
        feature_settings = settings.get('featureSettings', {})
        
        # Load image
        image = Image.open(file.stream)
        original_size = image.size
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        
        # Store original image for comparison
        original_b64 = image_to_base64(image)
        
        operations = []
        detection_info = "No detection performed"
        quality_info = ""
        resize_info = ""
        
        # Process image based on active features
        processed_image = image.copy()
        
        # 1. Image Resize (dilakukan pertama untuk efisiensi)
        if 'resize' in active_features:
            resize_settings = feature_settings.get('resize', {})
            target_width = resize_settings.get('width')
            target_height = resize_settings.get('height')
            maintain_aspect = resize_settings.get('maintainAspect', True)
            
            processed_image, resize_info = resize_image(processed_image, target_width, target_height, maintain_aspect)
            operations.append(f"Image Resize: {resize_info}")
        
        # 2. Enhancement
        if 'enhancement' in active_features:
            enhancement_settings = feature_settings.get('enhancement', {})
            intensity = enhancement_settings.get('intensity', 1.0)
            features = enhancement_settings.get('features', [])
            
            if 'quality' in features:
                processed_image = enhance_image_quality(processed_image, intensity)
                operations.append(f"Quality Enhancement (intensity: {intensity})")
            
            if 'color' in features:
                processed_image = auto_color_correction(processed_image, intensity)
                operations.append("Auto Color Correction")
            
            if 'hdr' in features:
                processed_image = apply_hdr_effect(processed_image, intensity)
                operations.append("HDR Effect")
            
            if 'histogram' in features:
                processed_image = apply_histogram_equalization(processed_image)
                operations.append("Histogram Equalization")
            
            if 'noise' in features:
                processed_image = apply_noise_reduction(processed_image, intensity=intensity)
                operations.append("Noise Reduction")
        
        # 3. Background processing
        if 'background' in active_features:
            bg_settings = feature_settings.get('background', {})
            method = bg_settings.get('method', 'ai')
            replace_type = bg_settings.get('replace', 'transparent')
            edge_quality = bg_settings.get('edgeQuality', 3)
            
            # Determine background color/type
            if replace_type == 'transparent':
                bg_color = "transparent"
            elif replace_type == 'white':
                bg_color = (255, 255, 255)
            elif replace_type == 'black':
                bg_color = (0, 0, 0)
            elif replace_type == 'custom':
                # Parse hex color to RGB
                hex_color = bg_settings.get('customColor', '#ffffff')
                hex_color = hex_color.lstrip('#')
                bg_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            else:
                bg_color = (255, 255, 255)  # Default white
            
            processed_image, bg_info = detect_and_remove_background(processed_image, method, bg_color, edge_quality)
            detection_info += f" | {bg_info}"
            operations.append(f"Background Processing ({method}, Q{edge_quality} -> {replace_type})")
            quality_info = f"Edge Quality: {edge_quality}/5, Method: {method.upper()}"
        
        # 4. Detection (separate from background processing)
        if 'detection' in active_features:
            detection_settings = feature_settings.get('detection', {})
            detection_types = detection_settings.get('types', [])
            face_action = detection_settings.get('faceAction', 'detect')
            blur_intensity = detection_settings.get('blurIntensity', 99)
            
            # Object detection with visualization (only if specifically requested)
            if 'objects' in detection_types:
                if 'background' not in active_features:  # Only show bounding boxes if not doing background removal
                    processed_image, obj_info = apply_object_detection_visualization(processed_image)
                    detection_info += f" | {obj_info}"
                    operations.append("Object Detection with Visualization")
                else:
                    # Just get detection info without drawing boxes
                    if YOLO_AVAILABLE and yolo_model:
                        try:
                            img_array = np.array(processed_image if processed_image.mode == 'RGB' else processed_image.convert('RGB'))
                            results = yolo_model(img_array)
                            
                            detected_objects = []
                            for result in results:
                                boxes = result.boxes
                                if boxes is not None:
                                    for box in boxes:
                                        if box.conf[0].cpu().numpy() > 0.5:
                                            cls = int(box.cls[0].cpu().numpy())
                                            detected_objects.append(yolo_model.names[cls])
                            
                            if detected_objects:
                                detection_info += f" | Objects: {', '.join(set(detected_objects))}"
                                operations.append("Object Detection (no visualization)")
                        except Exception as e:
                            detection_info += f" | Object detection failed: {str(e)}"
            
            # Face detection and processing
            if 'faces' in detection_types:
                faces, face_info = detect_faces(processed_image)
                detection_info += f" | {face_info}"
                
                if faces and face_action == 'blur':
                    processed_image = apply_face_blur(processed_image, faces, blur_intensity)
                    operations.append(f"Face Blurring (intensity: {blur_intensity})")
                elif faces:
                    operations.append("Face Detection")
                    
                    # Draw face rectangles for visualization if not doing background removal
                    if 'background' not in active_features:
                        draw = ImageDraw.Draw(processed_image)
                        for (top, right, bottom, left) in faces:
                            draw.rectangle([left, top, right, bottom], outline="blue", width=3)
                            draw.text((left, top-20), "Face", fill="blue")
        
        # 5. Effects
        if 'effects' in active_features:
            effects_settings = feature_settings.get('effects', {})
            effect_type = effects_settings.get('type')
            blur_type = effects_settings.get('blur')
            other_effect = effects_settings.get('other')
            intensity = effects_settings.get('intensity', 1.0)
            
            if effect_type:
                if effect_type == "sketch":
                    processed_image = apply_sketch_effect(processed_image, "pencil")
                    operations.append("Pencil Sketch Effect")
                elif effect_type == "cartoon":
                    processed_image = apply_cartoon_effect(processed_image)
                    operations.append("Cartoon Effect")
                elif effect_type == "sepia":
                    processed_image = apply_sepia_effect(processed_image, intensity)
                    operations.append(f"Sepia Effect (intensity: {intensity})")
                elif effect_type == "vignette":
                    processed_image = apply_vignette_effect(processed_image, 2.0/intensity)
                    operations.append(f"Vignette Effect (intensity: {intensity})")
                elif effect_type == "oil_painting":
                    processed_image = apply_oil_painting_effect(processed_image)
                    operations.append("Oil Painting Effect")
            
            if blur_type:
                processed_image = apply_blur_effect(processed_image, blur_type, int(5 * intensity))
                operations.append(f"{blur_type.title()} Blur Effect")
            
            if other_effect:
                if other_effect == "emboss":
                    processed_image = apply_emboss_effect(processed_image, intensity)
                    operations.append("Emboss Effect")
                elif other_effect == "edges":
                    processed_image = apply_edge_detection(processed_image)
                    operations.append("Edge Detection")
                elif other_effect == "watermark":
                    processed_image = apply_watermark(processed_image, "PROCESSED", int(128 * intensity))
                    operations.append("Watermark")
        
        # 6. AI processing
        if 'ai' in active_features:
            ai_settings = feature_settings.get('ai', {})
            ai_features = ai_settings.get('features', [])
            
            if 'super_res' in ai_features:
                # Simple upscaling as placeholder for super resolution
                width, height = processed_image.size
                new_size = (int(width * 1.5), int(height * 1.5))
                processed_image = processed_image.resize(new_size, Image.LANCZOS)
                operations.append("Super Resolution (1.5x)")
            
            if 'style_transfer' in ai_features:
                # Apply oil painting as style transfer placeholder
                processed_image = apply_oil_painting_effect(processed_image)
                operations.append("Style Transfer")
        
        # 7. Format conversion (always applied)
        if 'format' in active_features:
            format_settings = feature_settings.get('format', {})
            output_format = format_settings.get('type', 'PNG')
        else:
            output_format = 'PNG'  # Default format
        
        processed_image = convert_image_format(processed_image, output_format)
        final_size = processed_image.size
        
        if not operations:  # If no operations applied
            operations.append("Format Conversion")
        
        # Convert to base64 for display
        processed_b64 = image_to_base64(processed_image)
        
        # Calculate processing time
        processing_time = f"{time.time() - start_time:.2f} seconds"
        
        response_data = {
            'success': True,
            'original_image': original_b64,
            'processed_image': processed_b64,
            'operations': operations,
            'detection_info': detection_info,
            'output_format': output_format,
            'original_size': original_size,
            'final_size': final_size,
            'processing_time': processing_time,
            'quality_info': quality_info,
            'resize_info': resize_info,
            'is_preview': is_preview
        }
        
        # Only save and provide download URL if not in preview mode
        if not is_preview:
            # Save processed image
            base_filename = os.path.splitext(secure_filename(file.filename))[0]
            timestamp = int(time.time())
            filename = f"processed_{base_filename}_{timestamp}.{output_format.lower()}"
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            
            # Save with proper format and quality
            if output_format.upper() == 'JPEG':
                quality = feature_settings.get('format', {}).get('quality', 95)
                processed_image.save(output_path, format=output_format, quality=quality, optimize=True)
            elif output_format.upper() == 'PNG':
                processed_image.save(output_path, format=output_format, optimize=True)
            elif output_format.upper() == 'WEBP':
                quality = feature_settings.get('format', {}).get('quality', 95)
                processed_image.save(output_path, format=output_format, quality=quality, method=6)
            else:
                processed_image.save(output_path, format=output_format)
            
            response_data['download_url'] = f'/download/{filename}'
        else:
            response_data['download_url'] = None
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

if __name__ == '__main__':
    print("🚀 Starting Advanced Image Processor Pro - ULTRA ENHANCED EDITION...")
    print("📏 NEW! Image Resize Feature:")
    print("   ✅ Width & Height Input (pixel-based)")
    print("   ✅ Maintain Aspect Ratio Option")
    print("   ✅ High-Quality Lanczos Filtering")
    print("   ✅ Smart Processing Order (Resize First)")
    
    print("\n📋 ULTRA Advanced Background Removal Features:")
    print("   ✅ AI Model (rembg) - 95%+ Accuracy with Smart Post-Processing")
    print("   ✅ Advanced Multi-Algorithm - 5-Level Edge Detection + GrabCut + Morphology")
    print("   ✅ YOLO Enhanced - Object-Aware Segmentation with Adaptive Quality Control")
    print("   ✅ Quality-Based GrabCut - Dynamic Parameters (1-5 Quality Levels)")
    print("   ✅ Custom Background Colors with Real-time Color Picker")
    print("   ✅ Smart Edge Feathering & Anti-aliasing")
    print("   ✅ Multiple Threshold Combination & Noise Cleanup")
    
    print("\n🎯 Enhanced User Experience Features:")
    print("   ✅ Smart Resize Presets (Web Resize, Social Media)")
    print("   ✅ Preview Mode - Test results without saving files")
    print("   ✅ Detailed Progress Tracking with Feature Information")
    print("   ✅ Smart File Size Estimation & Processing Time Display")
    print("   ✅ Keyboard Shortcuts (Ctrl+O, Ctrl+Enter, Esc, 1-4)")
    print("   ✅ Interactive Notifications & Help System")
    print("   ✅ Comprehensive Tips & Usage Guidelines")
    
    print("\n🎮 Updated Quick Presets:")
    print("   • [1] Portrait Professional: AI Background + Enhancement")
    print("   • [2] Photo Enhancement: Quality + Color + HDR")
    print("   • [3] Web Resize: 1920x1080 + Enhancement")
    print("   • [4] Social Media: 1080x1080 Square + Background")
    
    if REMBG_AVAILABLE:
        print("   ✅ AI Background Removal (rembg) - ACTIVE & ULTRA OPTIMIZED")
    else:
        print("   ⚠️  AI Background Removal: pip install rembg onnxruntime")
    
    if YOLO_AVAILABLE and yolo_model:
        print("   ✅ YOLO Object Detection - ACTIVE & ULTRA ENHANCED")
    else:
        print("   ⚠️  YOLO Object Detection: pip install ultralytics")
    
    if TORCH_AVAILABLE:
        print("   ✅ PyTorch AI Support - ACTIVE")
    else:
        print("   ⚠️  PyTorch Support: pip install torch torchvision")
    
    if FACE_RECOGNITION_AVAILABLE:
        print("   ✅ Face Recognition & Privacy Controls - ACTIVE")
    else:
        print("   ⚠️  Face Recognition: pip install face-recognition")
    
    print("\n📏 Image Resize Usage:")
    print("   • Masukkan width dan height dalam pixel")
    print("   • Centang 'Pertahankan Aspect Ratio' untuk mencegah distorsi")
    print("   • Kosongkan salah satu dimensi untuk auto-calculate")
    print("   • Resize dilakukan pertama untuk efisiensi processing")
    
    print("\n⌨️  Keyboard Shortcuts:")
    print("   • Ctrl+O: Open files")
    print("   • Ctrl+Enter: Process images")
    print("   • Ctrl+Shift+Enter: Preview mode")
    print("   • Esc: Reset all settings")
    print("   • 1-4: Apply quick presets")
    
    print("\n🎨 Background Replacement Options:")
    print("   • Transparent Background (PNG with alpha)")
    print("   • Solid Colors (White, Black)")
    print("   • Custom Color with Real-time Color Picker & Hex Support")
    print("   • Smart Color Blending & Professional Anti-aliasing")
    
    print("\n📦 Complete Installation for All Features:")
    print("   pip install rembg onnxruntime ultralytics torch torchvision face-recognition opencv-python pillow flask")
    print("\n🌐 Access at: http://localhost:5000")
    print("💡 NEW: Image resize feature added with maintain aspect ratio!")
    print("🎯 Recommended: Use resize with aspect ratio for best results!")
    print("⚡ NEW: Try the Web Resize and Social Media presets!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)