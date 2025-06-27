# ✨ Advanced Image Processor Pro

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Stars](https://img.shields.io/github/stars/reyhannazera16/Image-Prosesing?style=social)

**Solusi lengkap untuk pengolahan gambar dengan AI, background removal, resize, dan efek artistik canggih**

[Demo](#demo) • [Installation](#installation) • [Features](#features) • [Documentation](#documentation) • [Support](#support)

</div>

---

## 🚀 **Overview**

Advanced Image Processor Pro adalah aplikasi web berbasis Flask yang menyediakan tools pengolahan gambar tingkat professional dengan interface yang user-friendly. Dibangun dengan teknologi AI dan computer vision terdepan untuk memberikan hasil berkualitas tinggi.

### 🎯 **Key Highlights**
- 🤖 **AI-Powered Background Removal** dengan akurasi 95%+
- 📏 **Smart Image Resize** dengan maintain aspect ratio
- 🎨 **20+ Artistic Effects** dan filters professional
- 🔍 **Object & Face Detection** dengan YOLO dan face-recognition
- ⚡ **Batch Processing** untuk multiple files
- 👁️ **Preview Mode** untuk testing tanpa save
- 🎮 **Quick Presets** untuk workflow cepat

---

## 📸 **Demo**

<div align="center">

| Original | Background Removed | Enhanced |
|----------|-------------------|----------|
| ![Original](demo/original.jpg) | ![Processed](demo/background-removed.jpg) | ![Enhanced](demo/enhanced.jpg) |

*Screenshot aplikasi akan segera ditambahkan*

</div>

---

## ✨ **Features**

### 🎨 **Core Features**

#### 📏 **Image Resize**
- ✅ Width & Height input dengan pixel precision
- ✅ Maintain aspect ratio option
- ✅ High-quality Lanczos filtering
- ✅ Smart auto-calculation untuk dimensi kosong

#### ✂️ **Background Processing**
- 🤖 **AI Model (rembg)** - 95%+ accuracy dengan smart post-processing
- 🔧 **Advanced Multi-Algorithm** - 5-level edge detection + GrabCut + Morphology
- 🎯 **YOLO Enhanced** - Object-aware segmentation dengan adaptive quality
- 🎚️ **Quality Control** - 1-5 quality levels untuk hasil optimal
- 🎨 **Custom Background Colors** dengan real-time color picker

#### 🚀 **Image Enhancement**
- 🔧 **AI-powered quality improvement** dengan denoising dan sharpening
- 🌈 **Auto color correction** dan white balance
- ✨ **HDR effects** untuk dynamic range lebih baik
- 📊 **Histogram equalization** untuk contrast optimal
- 🔇 **Advanced noise reduction** dengan multiple methods

#### 🔍 **Detection & Analysis**
- 🎯 **Object Detection** dengan YOLO v8 (80+ classes)
- 👤 **Face Detection** dengan face-recognition library
- 🔒 **Privacy Controls** - face blurring dengan adjustable intensity
- 📝 **Auto tagging** dan content analysis

#### 🎨 **Artistic Effects**
- ✏️ **Sketch Effects** - Pencil, charcoal, colored pencil
- 🎪 **Cartoon Effect** dengan edge detection
- 🖼️ **Oil Painting** simulation
- 📸 **Vintage Effects** - Sepia, vignette
- 💫 **Blur Effects** - Gaussian, motion, radial
- ⚡ **Advanced Filters** - Emboss, edge detection, watermark

#### 🤖 **AI & Machine Learning**
- 🔍 **Super Resolution** untuk upscaling berkualitas
- 🎨 **Style Transfer** dengan neural networks
- 🖌️ **AI Inpainting** untuk object removal
- 📈 **Smart Upscaling** dengan multiple algorithms

#### 🔄 **Format Conversion**
- 📁 **Multiple Formats** - PNG, JPEG, WEBP, BMP, TIFF
- ⚙️ **Quality Control** untuk lossy formats
- 🗜️ **Optimization** untuk file size terkecil
- 🖼️ **Transparency Support** untuk PNG dan WEBP

### 🎮 **User Experience**

#### ⚡ **Quick Presets**
- 👤 **Portrait Professional** - Background removal + enhancement
- 📸 **Photo Enhancement** - Quality + color + HDR
- 🌐 **Web Resize** - 1920x1080 + optimization
- 📱 **Social Media** - 1080x1080 square + background

#### 🖥️ **Interface Features**
- 📱 **Responsive Design** untuk semua device
- 🎭 **Drag & Drop** file upload
- 📊 **Real-time Progress** tracking
- 👁️ **Preview Mode** tanpa save file
- ⌨️ **Keyboard Shortcuts** untuk workflow cepat
- 🔄 **Batch Processing** untuk multiple files

---

## 🛠️ **Installation**

### **Prerequisites**
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### **Quick Start**

```bash
# Clone repository
git clone https://github.com/reyhannazera16/Image-Prosesing.git
cd Image-Prosesing

# Create virtual environment
python -m venv image_processor
source image_processor/bin/activate  # Linux/Mac
# atau
image_processor\Scripts\activate     # Windows

# Install core dependencies
pip install flask opencv-python pillow numpy werkzeug

# Install optional AI features (recommended)
pip install rembg onnxruntime ultralytics torch torchvision face-recognition

# Run application
python app.py
```

### **Docker Installation**

```dockerfile
# Dockerfile akan segera ditambahkan
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

### **Dependencies**

#### **Core Requirements**
```
flask>=2.0.0
opencv-python>=4.5.0
pillow>=8.0.0
numpy>=1.20.0
werkzeug>=2.0.0
```

#### **AI Features (Optional)**
```
rembg>=2.0.0              # AI background removal
onnxruntime>=1.10.0       # ONNX runtime for AI models
ultralytics>=8.0.0        # YOLO object detection
torch>=1.10.0             # PyTorch for deep learning
torchvision>=0.11.0       # Computer vision utilities
face-recognition>=1.3.0   # Face detection and recognition
```

---

## 📖 **Usage**

### **Web Interface**

1. **Start the application**
```bash
python app.py
```

2. **Open browser** dan akses `http://localhost:5000`

3. **Upload gambar** dengan drag & drop atau click upload

4. **Pilih fitur** yang diinginkan dengan toggle switches

5. **Adjust settings** sesuai kebutuhan

6. **Preview atau Process** gambar

7. **Download hasil** yang sudah diproses

### **API Usage**

```python
import requests

# Upload and process image
files = {'file': open('image.jpg', 'rb')}
data = {
    'settings': json.dumps({
        'activeFeatures': ['resize', 'enhancement'],
        'featureSettings': {
            'resize': {'width': 1920, 'height': 1080, 'maintainAspect': True},
            'enhancement': {'intensity': 1.2, 'features': ['quality', 'color']}
        }
    })
}

response = requests.post('http://localhost:5000/process', files=files, data=data)
result = response.json()

if result['success']:
    download_url = result['download_url']
    print(f"Download: {download_url}")
```

### **Keyboard Shortcuts**

| Shortcut | Action |
|----------|--------|
| `Ctrl + O` | Open file upload |
| `Ctrl + Enter` | Process images |
| `Ctrl + Shift + Enter` | Preview mode |
| `Esc` | Reset all settings |
| `1-4` | Apply quick presets |

---

## 🎯 **Use Cases**

### **Professional Photography**
- 📸 **Portrait Processing** - Background removal + enhancement
- 🎨 **Creative Effects** - Artistic filters dan stylization
- 📐 **Batch Resizing** - Multiple formats untuk client
- 🔍 **Quality Control** - Noise reduction dan sharpening

### **E-commerce**
- 🛍️ **Product Photography** - Clean backgrounds
- 📱 **Multi-platform Sizing** - Different dimensions
- ⚡ **Batch Processing** - Hundreds of products
- 🎨 **Consistent Styling** - Brand guidelines

### **Social Media**
- 📱 **Platform Optimization** - Instagram, Facebook, Twitter
- 🎭 **Creative Filters** - Engagement-boosting effects
- 📐 **Perfect Sizing** - Square, landscape, portrait
- ⚡ **Quick Workflows** - One-click presets

### **Web Development**
- 🌐 **Web Optimization** - Perfect sizing dan compression
- 🖼️ **Hero Images** - High-quality backgrounds
- 📱 **Responsive Images** - Multiple breakpoints
- ⚡ **Performance** - Optimized file sizes

---

## 📊 **Performance**

### **Processing Speed**
- 📏 **Resize**: ~0.1-0.5 seconds
- 🚀 **Enhancement**: ~1-3 seconds
- ✂️ **Background Removal**: ~2-8 seconds (depending on method)
- 🎨 **Effects**: ~0.5-2 seconds

### **Accuracy**
- 🤖 **AI Background Removal**: 95%+ accuracy
- 🎯 **Object Detection**: 90%+ accuracy (YOLO v8)
- 👤 **Face Detection**: 98%+ accuracy

### **Supported Formats**
- **Input**: JPG, PNG, GIF, BMP, TIFF, WEBP
- **Output**: PNG, JPEG, WEBP, BMP, TIFF
- **Max File Size**: 16MB per file

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Optional configuration
export MAX_CONTENT_LENGTH=16777216  # 16MB
export UPLOAD_FOLDER=uploads
export OUTPUT_FOLDER=outputs
export DEBUG=True
```

### **Advanced Settings**
```python
# app.py configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
```

---

## 🤝 **Contributing**

Kami sangat welcome untuk contributions! Berikut cara contribute:

### **Getting Started**
1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### **Development Setup**
```bash
# Clone your fork
git clone https://github.com/reyhannazera16/Image-Prosesing.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 .
black .
```

### **Areas for Contribution**
- 🐛 Bug fixes dan improvements
- ✨ New features dan effects
- 📚 Documentation improvements
- 🌐 Internationalization
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations

---

## 📋 **Roadmap**

### **Version 2.0** (Coming Soon)
- [ ] 🗃️ **Database Support** untuk user accounts
- [ ] ☁️ **Cloud Storage** integration (AWS S3, Google Cloud)
- [ ] 🔗 **API Authentication** dengan JWT tokens
- [ ] 📊 **Analytics Dashboard** untuk usage tracking
- [ ] 🌐 **Multi-language Support** (English, Indonesia, etc.)

### **Version 2.1**
- [ ] 🤖 **Advanced AI Models** - GPT-4 Vision, DALL-E integration
- [ ] 🎬 **Video Processing** support
- [ ] 📱 **Mobile App** (React Native)
- [ ] 🔌 **Plugin System** untuk custom effects

### **Version 3.0**
- [ ] 🌐 **Microservices Architecture**
- [ ] 🚀 **Kubernetes Deployment**
- [ ] 💼 **Enterprise Features**
- [ ] 🔄 **Workflow Automation**

---

## 💖 **Support & Donation**

Jika project ini berguna untuk Anda, pertimbangkan untuk mendukung pengembangan selanjutnya:

<div align="center">

### 🎉 **Dukung Project Ini**

[![Saweria](https://img.shields.io/badge/☕_Saweria-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://saweria.co/reyhannazera16)

**Atau scan QR Code:**

![Saweria QR](https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://saweria.co/reyhannazera16)

</div>

### **Cara Lain Mendukung:**
- ⭐ **Star** repository ini
- 🐛 **Report bugs** atau **suggest features**
- 📢 **Share** ke teman-teman developer
- 📝 **Write tutorials** atau blog posts
- 💼 **Hire us** untuk custom development

### **Sponsor Benefits:**
- 🏷️ **Credit** di README dan aplikasi
- 🎯 **Priority Support** untuk issues
- 💡 **Feature Requests** prioritas
- 📞 **Direct Contact** untuk consultation

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Reyhan Nazera - Advanced Image Processor Pro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 **Acknowledgments**

Special thanks to:

- 🤖 **[rembg](https://github.com/danielgatis/rembg)** - AI background removal
- 🎯 **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - Object detection
- 👤 **[face-recognition](https://github.com/ageitgey/face_recognition)** - Face detection
- 🖼️ **[OpenCV](https://opencv.org/)** - Computer vision library
- 🌐 **[Flask](https://flask.palletsprojects.com/)** - Web framework
- 🎨 **[Pillow](https://pillow.readthedocs.io/)** - Image processing

### **Contributors**
- 👨‍💻 **[Reyhan Nazera](https://github.com/reyhannazera16)** - Creator & Lead Developer

---

## 📞 **Contact**

<div align="center">

**Questions? Suggestions? Collaboration?**

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:reyhannazera16@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/reyhannazera16)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/reyhannazera16)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/reyhannazera16)

</div>

---

<div align="center">

**Made with ❤️ by [Reyhan Nazera](https://github.com/reyhannazera16)**

⭐ **Don't forget to star this repository if you found it helpful!** ⭐

</div>

---
