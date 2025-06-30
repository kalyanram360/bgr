import streamlit as st
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from rembg import remove, new_session
import cv2

st.set_page_config(layout="wide", page_title="🖼️ Professional Background Remover")
st.title("🖼️ Professional Background Removal - HD Quality") 

# Available models in rembg
MODELS = {
    "u2net": "U²-Net (Best for general use)",
    "u2net_human_seg": "U²-Net Human (Best for people)",
    "u2netp": "U²-Net+ (Lightweight & fast)",
    "birefnet-general": "BiRefNet General (High accuracy)",
    "birefnet-portrait": "BiRefNet Portrait (Best for portraits)",
    "isnet-general-use": "IS-Net (Good balance)",
    "sam": "SAM (Segment Anything Model)"
}

@st.cache_resource
def load_session(model_name):
    """Load rembg session with specified model"""
    try:
        return new_session(model_name)
    except Exception as e:
        st.warning(f"Could not load {model_name}, falling back to u2net")
        return new_session('u2net')

def enhance_image_quality(image: Image.Image, enhance_factor: float = 1.2) -> Image.Image:
    """Enhance image quality before processing"""
    if enhance_factor != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(enhance_factor)
    return image

def smooth_edges(image: Image.Image, blur_radius: int = 1) -> Image.Image:
    """Smooth edges of the alpha channel for better quality"""
    if image.mode != 'RGBA':
        return image
    
    # Extract alpha channel
    alpha = image.split()[-1]
    
    # Apply slight blur to alpha channel for smoother edges
    if blur_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Reconstruct image with smoothed alpha
    rgb = image.convert('RGB')
    result = Image.new('RGBA', image.size)
    result.paste(rgb)
    result.putalpha(alpha)
    
    return result

def remove_background_advanced(image: Image.Image, model_session, post_process: bool = True) -> Image.Image:
    """Remove background using rembg with advanced post-processing"""
    
    # Convert image to bytes for rembg
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Remove background
    result_bytes = remove(img_byte_arr, session=model_session)
    result_image = Image.open(BytesIO(result_bytes))
    
    # Post-processing for better quality
    if post_process:
        result_image = smooth_edges(result_image, blur_radius=0)
    
    return result_image

def create_background_variants(image: Image.Image):
    """Create different background variants"""
    variants = {}
    
    # Transparent
    variants['transparent'] = image
    
    # Common background colors
    colors = {
        'white': '#FFFFFF',
        'black': '#000000',
        'blue': '#0066CC',
        'green': '#00AA00',
        'red': '#CC0000'
    }
    
    for name, color in colors.items():
        bg = Image.new('RGB', image.size, color)
        if image.mode == 'RGBA':
            bg.paste(image, mask=image.split()[-1])
        else:
            bg.paste(image)
        variants[name] = bg
    
    return variants

# Streamlit UI
st.markdown("### Professional Background Removal with Multiple AI Models")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Model selection
selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    list(MODELS.keys()),
    format_func=lambda x: MODELS[x],
    index=0,
    help="Different models work better for different types of images"
)

# Quality settings
enhance_quality = st.sidebar.checkbox("Enhance input quality", value=True)
post_process = st.sidebar.checkbox("Advanced post-processing", value=True)
edge_smoothing = st.sidebar.slider("Edge smoothing", 0, 3, 1, help="Higher values = smoother edges")

# Load model session
with st.spinner(f"Loading {MODELS[selected_model]}..."):
    session = load_session(selected_model)

st.sidebar.success(f"✅ {MODELS[selected_model]} loaded")

# File upload
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'],
    help="Supports all common image formats. Higher resolution = better results!"
)

if uploaded_file is not None:
    # Load original image
    original_image = Image.open(uploaded_file)
    
    # Display original
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(original_image, use_column_width=True)
        st.info(f"Size: {original_image.size[0]} × {original_image.size[1]} pixels")
    
    # Processing button
    if st.button("🚀 Remove Background", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Enhance input if selected
            status_text.text("Step 1/4: Preparing image...")
            progress_bar.progress(25)
            
            processed_image = original_image.copy()
            if enhance_quality:
                processed_image = enhance_image_quality(processed_image)
            
            # Step 2: Remove background
            status_text.text("Step 2/4: Removing background with AI...")
            progress_bar.progress(50)
            
            result_image = remove_background_advanced(
                processed_image, 
                session, 
                post_process=post_process
            )
            
            # Step 3: Apply edge smoothing
            status_text.text("Step 3/4: Smoothing edges...")
            progress_bar.progress(75)
            
            if edge_smoothing > 0:
                result_image = smooth_edges(result_image, blur_radius=edge_smoothing)
            
            # Step 4: Generate variants
            status_text.text("Step 4/4: Generating download options...")
            progress_bar.progress(100)
            
            # Display result
            with col2:
                st.subheader("✨ Result")
                st.image(result_image, use_column_width=True)
                st.success(f"Background removed! Size: {result_image.size[0]} × {result_image.size[1]} pixels")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Download section
            st.markdown("---")
            st.subheader("📥 Download Options")
            
            # Create variants
            variants = create_background_variants(result_image)
            
            # Download buttons in columns
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                # Transparent PNG
                buf_transparent = BytesIO()
                result_image.save(buf_transparent, format="PNG", optimize=True)
                st.download_button(
                    "🔍 Transparent PNG",
                    buf_transparent.getvalue(),
                    file_name=f"transparent_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col2:
                # White background
                buf_white = BytesIO()
                variants['white'].save(buf_white, format="PNG", quality=95)
                st.download_button(
                    "⚪ White Background",
                    buf_white.getvalue(),
                    file_name=f"white_bg_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with dl_col3:
                # Black background
                buf_black = BytesIO()
                variants['black'].save(buf_black, format="PNG", quality=95)
                st.download_button(
                    "⚫ Black Background",
                    buf_black.getvalue(),
                    file_name=f"black_bg_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Custom background section
            st.markdown("---")
            st.subheader("🎨 Custom Background")
            
            custom_col1, custom_col2 = st.columns([1, 2])
            
            with custom_col1:
                # Color picker
                custom_color = st.color_picker("Choose background color", "#FFFFFF")
                
                # Background type selection
                bg_type = st.radio(
                    "Background type",
                    ["Solid Color", "Gradient"],
                    horizontal=True
                )
                
                if bg_type == "Gradient":
                    gradient_color2 = st.color_picker("Second color", "#000000")
                    gradient_direction = st.selectbox(
                        "Gradient direction",
                        ["Vertical", "Horizontal", "Diagonal"]
                    )
                
                apply_custom = st.button("Apply Custom Background", use_container_width=True)
            
            if apply_custom:
                with custom_col2:
                    if bg_type == "Solid Color":
                        # Solid color background
                        custom_bg = Image.new('RGB', result_image.size, custom_color)
                        if result_image.mode == 'RGBA':
                            custom_bg.paste(result_image, mask=result_image.split()[-1])
                        else:
                            custom_bg.paste(result_image)
                    else:
                        # Gradient background
                        width, height = result_image.size
                        gradient = Image.new('RGB', (width, height))
                        
                        # Create gradient array
                        if gradient_direction == "Vertical":
                            for y in range(height):
                                ratio = y / height
                                r = int((1-ratio) * int(custom_color[1:3], 16) + ratio * int(gradient_color2[1:3], 16))
                                g = int((1-ratio) * int(custom_color[3:5], 16) + ratio * int(gradient_color2[3:5], 16))
                                b = int((1-ratio) * int(custom_color[5:7], 16) + ratio * int(gradient_color2[5:7], 16))
                                for x in range(width):
                                    gradient.putpixel((x, y), (r, g, b))
                        elif gradient_direction == "Horizontal":
                            for x in range(width):
                                ratio = x / width
                                r = int((1-ratio) * int(custom_color[1:3], 16) + ratio * int(gradient_color2[1:3], 16))
                                g = int((1-ratio) * int(custom_color[3:5], 16) + ratio * int(gradient_color2[3:5], 16))
                                b = int((1-ratio) * int(custom_color[5:7], 16) + ratio * int(gradient_color2[5:7], 16))
                                for y in range(height):
                                    gradient.putpixel((x, y), (r, g, b))
                        else:  # Diagonal
                            for x in range(width):
                                for y in range(height):
                                    ratio = (x + y) / (width + height)
                                    r = int((1-ratio) * int(custom_color[1:3], 16) + ratio * int(gradient_color2[1:3], 16))
                                    g = int((1-ratio) * int(custom_color[3:5], 16) + ratio * int(gradient_color2[3:5], 16))
                                    b = int((1-ratio) * int(custom_color[5:7], 16) + ratio * int(gradient_color2[5:7], 16))
                                    gradient.putpixel((x, y), (r, g, b))
                        
                        custom_bg = gradient
                        if result_image.mode == 'RGBA':
                            custom_bg.paste(result_image, mask=result_image.split()[-1])
                        else:
                            custom_bg.paste(result_image)
                    
                    st.image(custom_bg, use_column_width=True)
                    
                    # Download custom background
                    buf_custom = BytesIO()
                    custom_bg.save(buf_custom, format="PNG", quality=95)
                    st.download_button(
                        "📥 Download Custom Background",
                        buf_custom.getvalue(),
                        file_name=f"custom_bg_{uploaded_file.name.split('.')[0]}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("💡 Try a different model or check if your image is corrupted.")

else:
    st.info("👆 Upload an image to get started!")
    
    # Feature showcase
    st.markdown("---")
    st.markdown("### ✨ Features")
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown("""
        **🤖 Multiple AI Models**
        - U²-Net for general use
        - Specialized human segmentation
        - BiRefNet for high accuracy
        - SAM (Segment Anything)
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **🎯 Perfect Quality**
        - Maintains original resolution
        - Advanced edge smoothing
        - Quality enhancement options
        - Professional results
        """)
    
    with feature_cols[2]:
        st.markdown("""
        **🎨 Flexible Outputs**
        - Transparent PNG
        - Custom solid colors
        - Beautiful gradients
        - Multiple format support
        """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - **Use high-resolution images** for better detail preservation
    - **Choose the right model**: Portrait models for people, general models for objects
    - **Good lighting** in original photo helps AI detection
    - **Clear subject boundaries** work best
    - **Try different models** if first result isn't perfect
    """)

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using rembg, OpenCV, and Streamlit** | Professional AI-powered background removal")