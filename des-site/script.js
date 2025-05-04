console.log('Hello, World!');

import ImageProcessor from './processor.js';


document.addEventListener('DOMContentLoaded', () => {

    // Processor Elements


    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const previewContainer = document.getElementById('preview-container');
    const clearImageBtn = document.getElementById('clearImage');
    const complexitySlider = document.getElementById('complexity');
    const complexityValue = document.getElementById('complexityValue');
    const generateBtn = document.getElementById('generateBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultCanvas = document.getElementById('resultCanvas');
    const lineEquationsDiv = document.getElementById('lineEquations');
    const parametricEquationsDiv = document.getElementById('parametricEquations');
    const circleEquationsDiv = document.getElementById('circleEquations');
    const desmosLinkInput = document.getElementById('desmosLink');
    const copyLinkBtn = document.getElementById('copyLink');
    const openDesmosBtn = document.getElementById('openDesmos');
    const loadingDiv = document.getElementById('loading');
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    // Initialize the image processor
    const imageProcessor = new ImageProcessor();

    // Variables
    let imageData = null;
    let imageElement = new Image();
    let desmos = {
        lines: [],
        parametricCurves: [],
        circles: [],
        url: ''
    };

    // Event Listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect);
    clearImageBtn.addEventListener('click', clearImage);
    complexitySlider.addEventListener('input', updateComplexityValue);
    generateBtn.addEventListener('click', generateDesmosArt);
    copyLinkBtn.addEventListener('click', copyDesmosLink);
    openDesmosBtn.addEventListener('click', openDesmosCalculator);

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            switchTab(tabId);
        });
    });

    // Functions
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length) {
            handleFiles(files);
        }
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length) {
            handleFiles(files);
        }
    }

    function handleFiles(files) {
        const file = files[0];
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            generateBtn.disabled = false;
            
            // Create image element for processing
            imageElement.onload = () => {
                imageData = {
                    width: imageElement.width,
                    height: imageElement.height,
                    src: e.target.result
                };
            };
            imageElement.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        preview.src = '';
        previewContainer.classList.add('hidden');
        generateBtn.disabled = true;
        imageData = null;
        resultsContainer.classList.add('hidden');
    }

    function updateComplexityValue() {
        complexityValue.textContent = complexitySlider.value;
    }

    function generateDesmosArt() {
        if (!imageData) {
            alert('Please select an image first');
            return;
        }

        loadingDiv.classList.remove('hidden');
        
        // Use the ImageProcessor to process the image
        setTimeout(() => {
            imageProcessor.processImage(imageData, parseInt(complexitySlider.value))
                .then(result => {
                    // Update the UI with the results
                    updateResults(result);
                    
                    // Hide loading screen
                    loadingDiv.classList.add('hidden');
                })
                .catch(error => {
                    console.error('Error processing image:', error);
                    alert('An error occurred while processing the image.');
                    loadingDiv.classList.add('hidden');
                });
        }, 100);
    }

    function updateResults(result) {
        // Store results
        desmos = result;
        
        // Update results UI
        resultsContainer.classList.remove('hidden');
        
        // Draw on canvas
        const ctx = resultCanvas.getContext('2d');
        resultCanvas.width = imageData.width;
        resultCanvas.height = imageData.height;
        
        // Draw original image faded
        ctx.globalAlpha = 0.3;
        ctx.drawImage(imageElement, 0, 0);
        ctx.globalAlpha = 1.0;
        
        // Draw lines
        ctx.strokeStyle = '#1a73e8';
        ctx.lineWidth = 2;
        for (const line of result.lines) {
            const [[x1, y1], [x2, y2]] = line.points;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        
        // Draw parametric curves
        ctx.strokeStyle = '#9c27b0';
        ctx.lineWidth = 2;
        for (const curve of result.parametricCurves) {
            ctx.beginPath();
            for (let i = 0; i < curve.points.length; i++) {
                const [x, y] = curve.points[i];
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
        
        // Draw circles
        ctx.strokeStyle = '#4caf50';
        ctx.lineWidth = 2;
        for (const circle of result.circles) {
            const [x, y] = circle.center;
            const r = circle.radius;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.stroke();
        }
        
        // Update equation lists
        updateEquationLists(result);
        
        // Update Desmos link
        desmosLinkInput.value = result.url;
    }

    function updateEquationLists(result) {
        // Clear previous equations
        lineEquationsDiv.innerHTML = '';
        parametricEquationsDiv.innerHTML = '';
        circleEquationsDiv.innerHTML = '';
        
        // Add line equations
        for (let i = 0; i < result.lines.length; i++) {
            const div = document.createElement('div');
            div.className = 'equation-item';
            div.textContent = result.lines[i].equation;
            div.addEventListener('click', () => copyToClipboard(result.lines[i].equation, div));
            lineEquationsDiv.appendChild(div);
        }
        
        // Add parametric equations
        for (let i = 0; i < result.parametricCurves.length; i++) {
            const curve = result.parametricCurves[i];
            const div = document.createElement('div');
            div.className = 'equation-item';
            div.innerHTML = `${curve.x_equation}<br>${curve.y_equation}`;
            div.addEventListener('click', () => {
                const text = `${curve.x_equation}\n${curve.y_equation}`;
                copyToClipboard(text, div);
            });
            parametricEquationsDiv.appendChild(div);
        }
        
        // Add circle equations
        for (let i = 0; i < result.circles.length; i++) {
            const circle = result.circles[i];
            const div = document.createElement('div');
            div.className = 'equation-item';
            div.textContent = circle.equation;
            div.addEventListener('click', () => copyToClipboard(circle.equation, div));
            circleEquationsDiv.appendChild(div);
        }
    }

    function copyToClipboard(text, element) {
        navigator.clipboard.writeText(text).then(() => {
            // Visual feedback
            const originalBackground = element.style.backgroundColor;
            element.style.backgroundColor = '#c8e6c9';
            setTimeout(() => {
                element.style.backgroundColor = originalBackground;
            }, 500);
        }).catch(err => {
            console.error('Could not copy text: ', err);
        });
    }

    function copyDesmosLink() {
        if (desmosLinkInput.value) {
            navigator.clipboard.writeText(desmosLinkInput.value).then(() => {
                // Flash button for feedback
                copyLinkBtn.classList.add('copied');
                setTimeout(() => {
                    copyLinkBtn.classList.remove('copied');
                }, 1000);
            }).catch(err => {
                console.error('Could not copy link: ', err);
            });
        }
    }

    function openDesmosCalculator() {
        if (desmosLinkInput.value) {
            window.open(desmosLinkInput.value, '_blank');
        }
    }

    function switchTab(tabId) {
        // Hide all tab contents
        tabContents.forEach(content => {
            content.classList.add('hidden');
        });
        
        // Deactivate all tab buttons
        tabButtons.forEach(button => {
            button.classList.remove('active');
        });
        
        // Show selected tab content
        document.getElementById(tabId).classList.remove('hidden');
        
        // Activate the selected tab button
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
    }

    // Initialize
    updateComplexityValue();
    switchTab('lineEquations'); 
});