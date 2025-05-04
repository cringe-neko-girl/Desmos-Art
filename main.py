
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy import optimize
import base64
import io
import re
import random
import math
from skimage import feature, morphology, filters
import json
import base64

class DesmosArtGenerator:
    def __init__(self):
        self.lines = []
        self.equations = []
        self.curves = []
        self.parametric_curves = []
        self.desmos_equations = []
        self.original_image = None
        self.edge_image = None
        self.contours = []
        self.polynomial_equations = []
        self.fourier_series = []
        
    def download_image(self, image_url):
        """Download an image from a URL"""
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.original_image = image
                return image
            else:
                print(f"Failed to retrieve image. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
            
    def load_image(self, file_path):
        """Load an image from a local file"""
        try:
            image = cv2.imread(file_path)
            if image is not None:
                self.original_image = image
                return image
            else:
                print(f"Failed to load image from {file_path}")
                return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
            
    def enhance_image(self, image):
        """Enhance image contrast and details"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen the image
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
            
    def preprocess_image(self, image, resize_width=None, high_quality=True):
        """Preprocess image for better edge detection"""
        # Make a copy of the image
        img = image.copy()
        
        # Resize if specified
        if resize_width is not None:
            height, width = img.shape[:2]
            ratio = resize_width / width
            new_height = int(height * ratio)
            img = cv2.resize(img, (resize_width, new_height))
            
        # Enhance the image if high quality is requested
        if high_quality:
            img = self.enhance_image(img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction while preserving edges
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Create multiple edge images with different techniques and combine them
        
        # 1. Canny edge detection with multiple thresholds
        edges1 = cv2.Canny(denoised, 30, 150)
        edges2 = cv2.Canny(denoised, 50, 200)
        
        # 2. Sobel edge detection
        sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, sobel_edges = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
        
        # 3. Laplacian edge detection
        laplacian = cv2.Laplacian(denoised, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, laplacian_edges = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
        
        # 4. Use scikit-image Canny implementation for better results
        try:
            ski_edges = feature.canny(denoised, sigma=1.2, low_threshold=0.09, high_threshold=0.18)
            ski_edges = ski_edges.astype(np.uint8) * 255
        except:
            ski_edges = np.zeros_like(edges1)  # Fallback if skimage is not available
        
        # Combine all edge images
        combined_edges = cv2.bitwise_or(edges1, edges2)
        combined_edges = cv2.bitwise_or(combined_edges, sobel_edges)
        combined_edges = cv2.bitwise_or(combined_edges, laplacian_edges)
        combined_edges = cv2.bitwise_or(combined_edges, ski_edges)
        
        # Thinning and cleaning of the edges
        # Apply morphological operations to clean up the edges
        kernel = np.ones((2, 2), np.uint8)
        cleaned_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Thinning - using skeletonization if available, otherwise erosion
        try:
            thinned_edges = morphology.skeletonize(cleaned_edges > 0)
            thinned_edges = thinned_edges.astype(np.uint8) * 255
        except:
            thinned_edges = cv2.erode(cleaned_edges, kernel, iterations=1)
        
        # Store the edge image
        self.edge_image = thinned_edges
        
        return thinned_edges
        
    def find_contours(self, edges, min_contour_length=10, filter_size=True):
        """Find contours in the edge image with improved filtering"""
        # Find contours in the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # Initial filtering by size
        if filter_size:
            filtered_contours = [cnt for cnt in contours if len(cnt) >= min_contour_length]
        else:
            filtered_contours = contours
        
        # Analyze contour properties for better filtering
        quality_contours = []
        for cnt in filtered_contours:
            # Calculate contour area and perimeter
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Skip tiny contours or those with very small area
            if area < 10:
                continue
                
            # Skip contours that are too circular (might be noise)
            circularity = 0
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.9 and area < 100:  # Small, very circular contours are often noise
                continue
                
            quality_contours.append(cnt)
        
        print(f"Found {len(quality_contours)} quality contours")
        self.contours = quality_contours
        return quality_contours
        
    def convert_frame_to_vectors(self, frame, min_line_length=30, max_line_gap=5, high_quality=True):
        """Convert frame to vector lines using advanced techniques"""
        # Create grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply pre-processing for better line detection
        if high_quality:
            # Use bilateral filter to preserve edges while removing noise
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive thresholding
            thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Combine thresholded and edges
            combined = cv2.bitwise_or(thresholded, edges)
            
            # Clean up with morphological operations
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        else:
            # Simple processing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            processed = edges

        # Use probabilistic Hough transform with optimized parameters
        lines_p = cv2.HoughLinesP(processed, rho=1, theta=np.pi/180, 
                                 threshold=max(30, min_line_length//2),  # Dynamic threshold
                                 minLineLength=min_line_length, 
                                 maxLineGap=max_line_gap)

        self.lines = []
        self.equations = []
        if lines_p is not None:
            print(f"Found {len(lines_p)} line segments")
            
            # Process and filter lines
            for line in lines_p:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Skip very short lines
                if length < min_line_length:
                    continue
                    
                self.lines.append(((x1, y1), (x2, y2)))
                
                # Generate equation for the line
                if x2 != x1:  # Not a vertical line
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    equation = f"y = {m:.6f}x + {b:.6f}"
                    self.equations.append(equation)
                    # Format for Desmos
                    self.desmos_equations.append(f"{m}x+{b}")
                else:  # Vertical line
                    self.equations.append(f"x = {x1}")
                    self.desmos_equations.append(f"x={x1}")
        else:
            print("No lines detected. Try adjusting parameters.")
            
        return self.lines

    def fit_polynomial_to_points(self, points, degree=3):
        """Fit a polynomial curve to a set of points with improved accuracy"""
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # Check if we have enough points for the requested degree
        if len(points) <= degree:
            degree = len(points) - 1
            if degree < 1:
                degree = 1
                
        # Fit the polynomial
        try:
            # Try weighted fit for better results
            weights = np.ones(len(points))
            # Give more weight to endpoints
            if len(points) > 5:
                weights[0] *= 2
                weights[-1] *= 2
                
            coeffs = np.polyfit(x_vals, y_vals, degree, w=weights)
        except:
            # Fallback to standard fit
            coeffs = np.polyfit(x_vals, y_vals, degree)
        
        # Create polynomial function
        p = np.poly1d(coeffs)
        
        # Generate equation string for Desmos
        equation = "y = "
        for i, coeff in enumerate(coeffs):
            if abs(coeff) < 1e-10:  # Skip very small coefficients
                continue
                
            power = degree - i
            if power > 0:
                if power == 1:
                    equation += f"{coeff:.6f}x + "
                else:
                    equation += f"{coeff:.6f}x^{power} + "
            else:
                equation += f"{coeff:.6f}"
        
        # Clean up equation for Desmos
        equation = equation.replace("+ -", "- ")
        if equation.endswith("+ "):
            equation = equation[:-2]
        desmos_eq = equation.replace("y = ", "").replace(" ", "")
        
        return {
            "equation": equation,
            "desmos": desmos_eq,
            "function": p,
            "domain": [min(x_vals), max(x_vals)]
        }

    def fit_parametric_curve(self, points, degree=3):
        """Fit a parametric curve to a set of points with improved accuracy"""
        # Create parameter t (0 to 1) based on cumulative distance
        total_dist = 0
        dists = [0]
        
        for i in range(1, len(points)):
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_dist += dist
            dists.append(total_dist)
        
        # Normalize t to [0, 1]
        if total_dist > 0:
            t_vals = [d/total_dist for d in dists]
        else:
            # Fallback if total_dist is zero (all points are the same)
            t_vals = np.linspace(0, 1, len(points))
        
        # Extract x and y coordinates
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # Check if we have enough points for the requested degree
        if len(points) <= degree:
            degree = len(points) - 1
            if degree < 1:
                degree = 1
        
        # Fit polynomials for x(t) and y(t)
        try:
            # Weighted fit for better results
            weights = np.ones(len(points))
            # Give more weight to endpoints and critical points
            if len(points) > 5:
                weights[0] *= 3  # Start point
                weights[-1] *= 3  # End point
                
                # Find critical points (corners) using angle changes
                for i in range(1, len(points)-1):
                    prev_vec = (points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
                    next_vec = (points[i+1][0] - points[i][0], points[i+1][1] - points[i][1])
                    
                    # Calculate normalized dot product to find angle
                    prev_len = np.sqrt(prev_vec[0]**2 + prev_vec[1]**2)
                    next_len = np.sqrt(next_vec[0]**2 + next_vec[1]**2)
                    
                    if prev_len > 0 and next_len > 0:
                        norm_dot = (prev_vec[0]*next_vec[0] + prev_vec[1]*next_vec[1]) / (prev_len * next_len)
                        # If angle is sharp, increase weight
                        if norm_dot < 0.7:  # Approximately 45 degrees or more
                            weights[i] *= 2.5
                            
            x_coeffs = np.polyfit(t_vals, x_vals, degree, w=weights)
            y_coeffs = np.polyfit(t_vals, y_vals, degree, w=weights)
        except:
            # Fallback to standard fit
            x_coeffs = np.polyfit(t_vals, x_vals, degree)
            y_coeffs = np.polyfit(t_vals, y_vals, degree)
        
        # Create polynomial functions
        x_poly = np.poly1d(x_coeffs)
        y_poly = np.poly1d(y_coeffs)
        
        # Generate equation strings
        x_eq = "x(t) = "
        for i, coeff in enumerate(x_coeffs):
            if abs(coeff) < 1e-10:  # Skip very small coefficients
                continue
                
            power = degree - i
            if power > 0:
                if power == 1:
                    x_eq += f"{coeff:.4f}t + "
                else:
                    x_eq += f"{coeff:.4f}t^{power} + "
            else:
                x_eq += f"{coeff:.4f}"
        
        y_eq = "y(t) = "
        for i, coeff in enumerate(y_coeffs):
            if abs(coeff) < 1e-10:  # Skip very small coefficients
                continue
                
            power = degree - i
            if power > 0:
                if power == 1:
                    y_eq += f"{coeff:.4f}t + "
                else:
                    y_eq += f"{coeff:.4f}t^{power} + "
            else:
                y_eq += f"{coeff:.4f}"
        
        # Clean up equations
        x_eq = x_eq.replace("+ -", "- ")
        y_eq = y_eq.replace("+ -", "- ")
        if x_eq.endswith("+ "):
            x_eq = x_eq[:-2]
        if y_eq.endswith("+ "):
            y_eq = y_eq[:-2]
        
        # Format for Desmos
        x_desmos = x_eq.replace("x(t) = ", "").replace(" ", "")
        y_desmos = y_eq.replace("y(t) = ", "").replace(" ", "")
        
        return {
            "x_equation": x_eq,
            "y_equation": y_eq,
            "x_desmos": x_desmos,
            "y_desmos": y_desmos,
            "x_function": x_poly,
            "y_function": y_poly
        }

    def convert_contours_to_equations(self, min_points=10, max_points=200, degree=3):
        """Convert contours to polynomial or parametric equations with improved accuracy"""
        self.polynomial_equations = []
        self.parametric_curves = []
        
        if not self.contours:
            print("No contours available. Run find_contours first.")
            return
        
        for i, contour in enumerate(self.contours):
            # Skip if contour is too small
            if len(contour) < min_points:
                continue
                
            # Simplify contour using Douglas-Peucker algorithm
            # Use a dynamic epsilon based on contour length/complexity
            contour_length = cv2.arcLength(contour, True)
            epsilon = 0.001 * contour_length
            approx = cv2.approxPolyDP(contour, epsilon, False)
            
            # Extract points from contour
            points = [(p[0][0], p[0][1]) for p in approx]
            
            # Limit points for efficiency, but ensure we keep critical points
            if len(points) > max_points:
                # Instead of linear sampling, try to preserve the shape
                # Find critical points (high curvature)
                critical_indices = []
                
                # Add start and end points
                critical_indices.append(0)
                critical_indices.append(len(points) - 1)
                
                # Add points with high curvature
                for j in range(1, len(points) - 1):
                    # Calculate angle between adjacent segments
                    prev_pt = points[j-1]
                    curr_pt = points[j]
                    next_pt = points[j+1]
                    
                    vec1 = np.array([curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]])
                    vec2 = np.array([next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]])
                    
                    # Calculate magnitudes
                    mag1 = np.linalg.norm(vec1)
                    mag2 = np.linalg.norm(vec2)
                    
                    if mag1 > 0 and mag2 > 0:
                        # Calculate dot product and angle
                        cos_angle = np.dot(vec1, vec2) / (mag1 * mag2)
                        # Clip to handle numerical instability
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        
                        # If angle is significant, mark as critical
                        if angle > 0.3:  # About 17 degrees
                            critical_indices.append(j)
                
                # Ensure we include some regular samples if critical points are too few
                if len(critical_indices) < max_points // 2:
                    # Add regular samples
                    remaining_points = max_points - len(critical_indices)
                    step = len(points) / (remaining_points + 1)
                    
                    for j in range(remaining_points):
                        idx = int((j + 1) * step)
                        if idx not in critical_indices and 0 < idx < len(points) - 1:
                            critical_indices.append(idx)
                            
                # Sort indices and extract points
                critical_indices = sorted(critical_indices)
                points = [points[idx] for idx in critical_indices]
                
                # If still too many points, use linear sampling as fallback
                if len(points) > max_points:
                    indices = np.linspace(0, len(points)-1, max_points, dtype=int)
                    points = [points[idx] for idx in indices]
            
            # Adaptively choose degree based on number of points and complexity
            adaptive_degree = min(degree, max(2, len(points) // 10))
            
            # If contour has enough points, fit a parametric curve
            if len(points) >= adaptive_degree + 1:
                try:
                    curve = self.fit_parametric_curve(points, adaptive_degree)
                    curve["contour_index"] = i
                    self.parametric_curves.append(curve)
                    
                    # Try to fit a polynomial as well for simple contours
                    try:
                        # Check if this contour is mostly horizontal/vertical (good for polynomial)
                        x_vals = [p[0] for p in points]
                        y_vals = [p[1] for p in points]
                        x_range = max(x_vals) - min(x_vals)
                        y_range = max(y_vals) - min(y_vals)
                        
                        if x_range > 0 and y_range / x_range < 3:  # Not too steep
                            poly = self.fit_polynomial_to_points(points, min(3, adaptive_degree))
                            poly["contour_index"] = i
                            self.polynomial_equations.append(poly)
                    except:
                        pass
                except Exception as e:
                    print(f"Failed to fit curve to contour {i}: {e}")
                    
        print(f"Created {len(self.parametric_curves)} parametric curves")
        print(f"Created {len(self.polynomial_equations)} polynomial equations")
        
        return self.parametric_curves
        
    def generate_circle_equation(self, x, y, r):
        """Generate equation for a circle in Desmos format"""
        return f"(x-{x})^2+(y-{y})^2={r*r}"
        
    def detect_and_draw_curves(self, frame):
        """Detect circles and ellipses with enhanced precision"""
        # Create grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect circles using Hough Circle Transform with optimized parameters
        circles = cv2.HoughCircles(
            filtered, 
            cv2.HOUGH_GRADIENT, 
            dp=1.5,               # Higher dp for better precision
            minDist=30,           # Minimum distance between circles
            param1=70,            # Upper threshold for edge detection
            param2=30,            # Threshold for center detection
            minRadius=10, 
            maxRadius=150
        )

        self.curves = []
        output = frame.copy()
        
        if circles is not None:
            # Convert to integer coordinates
            circles = np.round(circles[0, :]).astype("int")
            
            # Filter circles based on quality metrics
            quality_circles = []
            for (x, y, r) in circles:
                # Skip circles outside image boundaries
                if x - r < 0 or y - r < 0 or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
                    continue
                    
                # Create a mask for the circular region
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Check how well the circle matches edges
                edges = cv2.Canny(filtered, 50, 150)
                matches = cv2.bitwise_and(edges, mask)
                edge_pixels = np.sum(matches == 255)
                
                # Calculate perimeter of the circle
                perimeter = 2 * np.pi * r
                
                # Calculate match quality
                if perimeter > 0:
                    match_quality = edge_pixels / perimeter
                    
                    # Keep only good matches
                    if match_quality > 0.3:
                        quality_circles.append((x, y, r))
                        # Draw on output image
                        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            
            self.curves = quality_circles
            print(f"Found {len(quality_circles)} quality circles")
                
        return output, self.curves
        
    def approximation_with_fourier(self, contour, n_harmonics=10):
        """Approximate a contour with Fourier series for smooth curves"""
        # Flatten contour points
        points = contour.reshape(-1, 2)
        
        # Convert to complex numbers for easier computation
        complex_points = points[:, 0] + 1j * points[:, 1]
        
        # Number of points
        N = len(complex_points)
        
        # Compute Fourier coefficients
        coeffs = np.zeros(2*n_harmonics+1, dtype=complex)
        
        for k in range(-n_harmonics, n_harmonics+1):
            for n in range(N):
                coeffs[k+n_harmonics] += complex_points[n] * np.exp(-2j * np.pi * k * n / N)
            coeffs[k+n_harmonics] /= N
            
        # Create parametric equations for Desmos
        x_eq = ""
        y_eq = ""
        
        for k in range(-n_harmonics, n_harmonics+1):
            if abs(coeffs[k+n_harmonics]) < 1e-4:  # Skip negligible coefficients
                continue
                
            if k == 0:
                x_eq += f"{coeffs[k+n_harmonics].real:.4f}"
                y_eq += f"{coeffs[k+n_harmonics].imag:.4f}"
            else:
                amp = abs(coeffs[k+n_harmonics])
                phase = np.angle(coeffs[k+n_harmonics])
                x_eq += f" + {amp:.4f}\\cos({k}t + {phase:.4f})"
                y_eq += f" + {amp:.4f}\\sin({k}t + {phase:.4f})"
                
        return {
            "x_equation": f"x(t) = {x_eq}",
            "y_equation": f"y(t) = {y_eq}",
            "x_desmos": x_eq.replace(" ", ""),
            "y_desmos": y_eq.replace(" ", ""),
            "coeffs": coeffs,
            "n_harmonics": n_harmonics
        }
        
    def apply_fourier_to_complex_contours(self, min_points=30):
        """Apply Fourier series to complex contours for better representations"""
        if not self.contours:
            print("No contours available. Run find_contours first.")
            return []
            
        self.fourier_series = []
        
        for i, contour in enumerate(self.contours):
            # Only apply Fourier to sufficiently complex contours
            if len(contour) >= min_points:
                try:
                    # Determine number of harmonics based on contour complexity
                    n_harmonics = min(20, max(5, len(contour) // 10))
                    
                    fourier = self.approximation_with_fourier(contour, n_harmonics)
                    fourier["contour_index"] = i
                    self.fourier_series.append(fourier)
                except Exception as e:
                    print(f"Failed to apply Fourier to contour {i}: {e}")
                    
        print(f"Created {len(self.fourier_series)} Fourier series representations")
        return self.fourier_series
    
    def generate_desmos_url(self):
     """Generate a sharable Desmos URL with all equations"""
     

     base_url = "https://www.desmos.com/calculator?state="

     expressions = []

     # Add line equations
     for eq in self.desmos_equations[:20]:  # Limit to 20
        expressions.append({"type": "expression", "latex": f"y={eq}"})

     # Add parametric curves
     for curve in self.parametric_curves[:15]:  # Limit to 15
        param_eq = f"\\left(x,y\\right)=\\left({curve['x_desmos']},{curve['y_desmos']}\\right)\\left\\{{0\\le t\\le1\\right\\}}"
        expressions.append({"type": "expression", "latex": param_eq})

     # Add circles
     for x, y, r in self.curves[:10]:  # Limit to 10
        circle_eq = f"(x-{x})^2+(y-{y})^2={r*r}"
        expressions.append({"type": "expression", "latex": circle_eq})

     # Build calculator state
     state = {
        "version": 7,
        "expressions": {"list": expressions}
     }

     # Encode state as base64 URL-safe string
     json_state = json.dumps(state, separators=(',', ':'))
     encoded_state = base64.urlsafe_b64encode(json_state.encode()).decode()

     return base_url + encoded_state  
 
    def generate_desmos_code(self):
        """Generate Desmos calculator code"""
        expressions = []
        
        # Add line equations
        for i, eq in enumerate(self.desmos_equations[:20]):  # Limit to 20 equations
            expressions.append({"type": "expression", "latex": f"y={eq}"})
            
        # Add parametric curves
        for i, curve in enumerate(self.parametric_curves[:15]):  # Limit to 15 curves
            param_eq = f"\\left(x,y\\right)=\\left({curve['x_desmos']},{curve['y_desmos']}\\right)\\left\\{{0\\le t\\le1\\right\\}}"
            expressions.append({"type": "expression", "latex": param_eq})
            
        # Add circles
        for x, y, r in self.curves[:10]:  # Limit to 10 circles
            circle_eq = f"(x-{x})^2+(y-{y})^2={r*r}"
            expressions.append({"type": "expression", "latex": circle_eq})
            
        # Build Desmos calculator state
        calculator_state = {
            "version": 7,
            "expressions": {"list": expressions}
        }
        
        return calculator_state
        
    def plot_custom_graph(self, show_original=False, alpha=0.5, show_contours=True):
        """Plot detected lines and curves over the original image if available"""
        plt.figure(figsize=(12, 10))
        show_original=False
        
        if show_original and self.original_image is not None:
            img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb, alpha=alpha)
        elif self.edge_image is not None:
            plt.imshow(self.edge_image, cmap='gray', alpha=alpha)
            
        # Plot lines - REMOVED NUMBERS/LABELS
        for ((x1, y1), (x2, y2)) in self.lines:
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
            # Removed the label/number text

        # Plot circles - REMOVED NUMBERS/LABELS
        #for (x, y, r) in self.curves:
            #plt.gca().add_artist(plt.Circle((x, y), r, color='green', fill=False, linewidth=2))
            # Removed the label/number text
            
        # Plot contours
        if show_contours and self.contours:
            # Choose a set of distinct colors for contours
            colors = plt.cm.viridis(np.linspace(0, 1, min(50, len(self.contours))))
            
            for i, contour in enumerate(self.contours):
                if i < 50:  # Limit number of displayed contours
                    # Extract x and y coordinates
                    x = contour[:, 0, 0]
                    y = contour[:, 0, 1]
                    
                    # Plot contour
                    plt.plot(x, y, '-', color=colors[i], linewidth=1, alpha=0.7)
            
        # Plot parametric curves
        if self.parametric_curves:
            t_vals = np.linspace(0, 1, 100)
            for i, curve in enumerate(self.parametric_curves):
                if i < 20:  # Limit number of displayed curves
                    x_vals = [curve["x_function"](t) for t in t_vals]
                    y_vals = [curve["y_function"](t) for t in t_vals]
                    plt.plot(x_vals, y_vals, '--', color='purple', linewidth=2, alpha=0.8)
        
        # Set plot dimensions based on original image
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]
            plt.xlim(0, width)
            plt.ylim(height, 0)  # Invert y-axis to match image coordinates
        else:
            plt.gca().invert_yaxis()

        plt.title("Vectorized Representation with Mathematical Equations")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def display_line_equations(self):
        """Display all detected line equations"""
        if not self.equations:
            print("No line equations available. Run conversion first.")
            return
            
        print("\n--- Line Equations ---")
        for i, eq in enumerate(self.equations):
            print(f"Line {i+1}: {eq}")
            
    def display_parametric_equations(self):
        """Display all parametric curve equations"""
        if not self.parametric_curves:
            print("No parametric curves available. Run convert_contours_to_equations first.")
            return
            
        print("\n--- Parametric Equations ---")
        for i, curve in enumerate(self.parametric_curves[:10]):  # Limit to first 10 curves
            print(f"Curve {i+1}:")
            print(f"  {curve['x_equation']}")
            print(f"  {curve['y_equation']}")
            print()
            
    def export_desmos_instructions(self):
        """Export instructions for pasting equations into Desmos"""
        if not self.desmos_equations and not self.parametric_curves:
            print("No equations available to export.")
            return
            
        print("\n--- Desmos Instructions ---")
        print("1. Go to https://www.desmos.com/calculator")
        print("2. Copy and paste the following equations one by one:")
        
        # Export line equations
        if self.desmos_equations:
            print("\nLine Equations:")
            for i, eq in enumerate(self.desmos_equations[:20]):  # Limit to 20 equations
                print(f"  y = {eq}")
                
        # Export parametric equations
        if self.parametric_curves:
            print("\nParametric Equations:")
            for i, curve in enumerate(self.parametric_curves[:15]):  # Limit to 15 curves
                print(f"  Curve {i+1}:")
                print(f"  x(t) = {curve['x_desmos']}")
                print(f"  y(t) = {curve['y_desmos']}")
                print(f"  Domain: 0 ≤ t ≤ 1")
                
        # Export circle equations
        if self.curves:
            print("\nCircle Equations:")
            for i, (x, y, r) in enumerate(self.curves[:10]):  # Limit to 10 circles
                print(f"  (x-{x})^2+(y-{y})^2={r*r}")
                
    def create_desmos_art(self, image, complexity=3):
        """Create Desmos art from an image with one function call"""
        if image is None:
            print("No image provided.")
            return
            
        # Preprocess image
        print("Preprocessing image...")
        edges = self.preprocess_image(image, resize_width=400)
        
        # Find contours
        print("Finding contours...")
        contours = self.find_contours(edges)
        
        # Convert contours to equations
        print("Converting contours to mathematical equations...")
        self.convert_contours_to_equations(degree=complexity)
        
        # Detect circles
        print("Detecting circles...")
        self.detect_and_draw_curves(image)
        
        # Convert frame to vectors (lines)
        print("Detecting lines...")
        self.convert_frame_to_vectors(image, min_line_length=20, max_line_gap=10)
        
        # Plot result
        self.plot_custom_graph(show_original=True, alpha=0.5, show_contours=False)
        
        # Display equations
        self.display_parametric_equations()
        self.display_line_equations()
        
        # Export Desmos instructions
        self.export_desmos_instructions()
        
        print("\nDesmos art generation complete!")
        print(f"Created {len(self.parametric_curves)} parametric curves, {len(self.lines)} lines, and {len(self.curves)} circles.")
        
        print(f"Genrated Desmos Url:", self.generate_desmos_url())
        
        return {
            "parametric_curves": self.parametric_curves,
            "lines": self.lines,
            "circles": self.curves
        }

def main():
    print("=== Desmos Art Generator ===")
    
    # Get image input
    input_type = input("Enter '1' for URL or '2' for local file: ").strip()
    
    generator = DesmosArtGenerator()
    image = None
    
    if input_type == "1":
        image_url = input("Enter an image URL: ").strip()
        image = generator.download_image(image_url)
    else:
        file_path = input("Enter file path: ").strip()
        image = generator.load_image(file_path)
        
    if image is None:
        print("Failed to load image. Exiting.")
        return
        
    # Get complexity
    complexity = int(input("Enter complexity level (1-5, higher is more complex): ").strip() or "3")
    complexity = max(1, min(5, complexity))
    
    # Create Desmos art
    generator.create_desmos_art(image, complexity)

if __name__ == "__main__":
    main()