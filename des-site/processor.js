/**
 * Image Processor Module for Mathematical Art Generation
 *
 * This module processes images to detect edges and generate mathematical
 * equations that represent the image content.
 */
class ImageProcessor {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
    }

    /**
     * Process an image and generate mathematical representations
     * @param {Object} imageData - Object containing image information (e.g., { src: 'path/to/image.jpg' })
     * @param {number} complexity - Level of detail (1-10)
     * @returns {Promise<Object>} A promise that resolves with the generated equations and Desmos data
     */
    processImage(imageData, complexity) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => {
                // Setup canvas with image dimensions
                this.canvas.width = image.width;
                this.canvas.height = image.height;

                // Draw the image on the canvas
                this.ctx.drawImage(image, 0, 0);

                try {
                    // Process the image
                    const edges = this.detectEdges();
                    const contours = this.findContours(edges, complexity);

                    // Generate mathematical equations
                    const result = this.generateEquations(contours, complexity, image.width, image.height);

                    resolve(result);
                } catch (error) {
                    reject(error); // Reject the promise if processing fails
                }
            };
            image.onerror = (error) => {
                reject(error); // Reject the promise if image loading fails
            };
            image.src = imageData.src;
        });
    }

    /**
     * Detect edges in the image
     * @returns {Uint8ClampedArray} Binary edge detection result (0 or 255)
     */
    detectEdges() {
        // Create a grayscale version
        const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const grayscale = this.convertToGrayscale(imgData);

        // Apply Sobel operator for edge detection
        const edges = this.applySobelOperator(grayscale);

        // Apply threshold to create binary edge image
        const threshold = this.calculateThreshold(edges);
        const binaryEdges = this.applyThreshold(edges, threshold);

        return binaryEdges;
    }

    /**
     * Convert image data to grayscale
     * @param {ImageData} imgData - Original image data
     * @returns {Uint8ClampedArray} Grayscale image data
     */
    convertToGrayscale(imgData) {
        const data = imgData.data;
        const grayscale = new Uint8ClampedArray(this.canvas.width * this.canvas.height);

        for (let i = 0; i < data.length; i += 4) {
            // Standard grayscale conversion
            const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            grayscale[i / 4] = gray;
        }

        return grayscale;
    }

    /**
     * Apply Sobel operator for edge detection
     * @param {Uint8ClampedArray} grayscale - Grayscale image data
     * @returns {Uint8ClampedArray} Edge strength at each pixel
     */
    applySobelOperator(grayscale) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const edges = new Uint8ClampedArray(width * height);

        // Sobel kernels
        const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let gx = 0;
                let gy = 0;

                // Apply kernels
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = (y + ky) * width + (x + kx);
                        const kernelIdx = (ky + 1) * 3 + (kx + 1);

                        gx += grayscale[idx] * sobelX[kernelIdx];
                        gy += grayscale[idx] * sobelY[kernelIdx];
                    }
                }

                // Calculate gradient magnitude
                const magnitude = Math.sqrt(gx * gx + gy * gy);
                edges[y * width + x] = Math.min(255, magnitude);
            }
        }

        return edges;
    }

    /**
     * Calculate appropriate threshold value using Otsu's method
     * @param {Uint8ClampedArray} edges - Edge strength data
     * @returns {number} Threshold value
     */
    calculateThreshold(edges) {
        const histogram = new Array(256).fill(0);

        // Build histogram
        for (let i = 0; i < edges.length; i++) {
            histogram[edges[i]]++;
        }

        const total = edges.length;
        let sum = 0;
        for (let i = 0; i < 256; i++) {
            sum += i * histogram[i];
        }

        let sumB = 0;
        let wB = 0;
        let wF = 0;
        let maxVariance = 0;
        let threshold = 0;

        for (let t = 0; t < 256; t++) {
            wB += histogram[t];                // Weight background
            if (wB === 0) continue;

            wF = total - wB;                   // Weight foreground
            if (wF === 0) break;

            sumB += t * histogram[t];
            const mB = sumB / wB;              // Mean background
            const mF = (sum - sumB) / wF;      // Mean foreground

            // Calculate between-class variance
            const variance = wB * wF * (mB - mF) * (mB - mF);

            if (variance > maxVariance) {
                maxVariance = variance;
                threshold = t;
            }
        }

        return threshold;
    }

    /**
     * Apply threshold to edge detection result
     * @param {Uint8ClampedArray} edges - Edge strength data
     * @param {number} threshold - Threshold value
     * @returns {Uint8ClampedArray} Binary edge image (0 or 255)
     */
    applyThreshold(edges, threshold) {
        const binary = new Uint8ClampedArray(edges.length);

        for (let i = 0; i < edges.length; i++) {
            binary[i] = edges[i] > threshold ? 255 : 0;
        }

        return binary;
    }

    /**
     * Find contours in the edge-detected image
     * This implementation uses k-means clustering on edge points as a simplified contour finding.
     * More robust contour finding algorithms like Marching Squares or Suzuki-Abe are more complex.
     * @param {Uint8ClampedArray} edges - Binary edge image
     * @param {number} complexity - Level of detail
     * @returns {Array<Array<Array<number>>>} Array of contours, where each contour is an array of [x, y] points
     */
    findContours(edges, complexity) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const contours = [];

        // Number of contours to find based on complexity
        const targetContourCount = 5 + complexity * 3;

        // Find significant edge points
        const significantPoints = [];
        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                const idx = y * width + x;
                if (edges[idx] > 0) {
                    significantPoints.push([x, y]);
                }
            }
        }

        // If there are too many edge points, sample them
        const maxPointsForClustering = 2000; // Limit the number of points for k-means
        let pointsToCluster = significantPoints;
        if (significantPoints.length > maxPointsForClustering) {
            const samplingRate = Math.ceil(significantPoints.length / maxPointsForClustering);
            pointsToCluster = [];
            for (let i = 0; i < significantPoints.length; i += samplingRate) {
                pointsToCluster.push(significantPoints[i]);
            }
        }

        // Perform k-means clustering on sampled edge points
        const k = Math.min(targetContourCount, Math.floor(pointsToCluster.length / 10));
        if (k <= 0) {
            return []; // No points or k is too small
        }

        const clusters = this.kMeansClustering(pointsToCluster, k);

        // For each cluster, attempt to form a contour
        for (let i = 0; i < clusters.length; i++) {
            const cluster = clusters[i];
            if (cluster.length < 3) continue;

            // Order points to form a contour (simplified approach)
            const contour = this.orderPointsIntoContour(cluster);

            // If contour has enough points, add it
            if (contour.length >= 3) {
                contours.push(contour);
            }
        }

        return contours;
    }

    /**
     * Perform k-means clustering on points
     * @param {Array<Array<number>>} points - Array of [x, y] coordinates
     * @param {number} k - Number of clusters
     * @returns {Array<Array<Array<number>>>} Array of clusters, where each cluster is an array of [x, y] points
     */
    kMeansClustering(points, k) {
        if (points.length <= k) {
            // If we have fewer points than clusters, return each point as its own cluster
            return points.map(p => [p]);
        }

        // Initialize centroids randomly from the existing points
        const centroids = [];
        const usedIndices = new Set();

        for (let i = 0; i < k; i++) {
            let randomIndex;
            do {
                randomIndex = Math.floor(Math.random() * points.length);
            } while (usedIndices.has(randomIndex));

            usedIndices.add(randomIndex);
            centroids.push([...points[randomIndex]]);
        }

        // Perform k-means iterations
        let clusters = new Array(k).fill(null).map(() => []);
        const maxIterations = 10; // Limit iterations to prevent infinite loops

        for (let iter = 0; iter < maxIterations; iter++) {
            // Clear previous clusters
            clusters.forEach(cluster => cluster.length = 0);

            // Assign points to nearest centroid
            for (const point of points) {
                let minDist = Infinity;
                let closestClusterIndex = 0;

                for (let i = 0; i < k; i++) {
                    const dist = this.distance(point, centroids[i]);
                    if (dist < minDist) {
                        minDist = dist;
                        closestClusterIndex = i;
                    }
                }
                clusters[closestClusterIndex].push([...point]);
            }

            // Recalculate centroids and check for changes
            let changed = false;
            for (let i = 0; i < k; i++) {
                if (clusters[i].length === 0) {
                    // Handle empty clusters: re-initialize with a random point
                    if (points.length > 0) {
                         // Find a point not assigned to any cluster yet
                         const unassignedPoint = points.find(p => {
                            for(let j=0; j<k; j++) {
                                if (j !== i && clusters[j].some(cp => cp[0] === p[0] && cp[1] === p[1])) {
                                    return false; // Point is in another cluster
                                }
                            }
                            return true; // Point is not in any other cluster
                         });
                         if(unassignedPoint) {
                            centroids[i] = [...unassignedPoint];
                            changed = true;
                         } else {
                            // If all points are assigned, randomly pick one
                             const randomPointIndex = Math.floor(Math.random() * points.length);
                             centroids[i] = [...points[randomPointIndex]];
                             changed = true;
                         }
                    }
                    continue; // Skip centroid calculation for empty cluster for this iteration
                }

                const newCentroid = [0, 0];
                for (const point of clusters[i]) {
                    newCentroid[0] += point[0];
                    newCentroid[1] += point[1];
                }

                newCentroid[0] /= clusters[i].length;
                newCentroid[1] /= clusters[i].length;

                // Check if centroid has moved significantly
                if (this.distance(newCentroid, centroids[i]) > 0.5) {
                    changed = true;
                }

                centroids[i] = newCentroid;
            }

            // If centroids didn't change much, break
            if (!changed) break;
        }

        // Filter out empty clusters
        return clusters.filter(cluster => cluster.length > 0);
    }


    /**
     * Calculate Euclidean distance between two points
     * @param {Array<number>} p1 - First point [x, y]
     * @param {Array<number>} p2 - Second point [x, y]
     * @returns {number} Distance
     */
    distance(p1, p2) {
        const dx = p1[0] - p2[0];
        const dy = p1[1] - p2[1];
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Order points to form a contour. This is a simplified approach based on angle from centroid.
     * More sophisticated methods like traversing the edge pixels are needed for true contour ordering.
     * @param {Array<Array<number>>} points - Array of [x, y] coordinates belonging to a potential contour
     * @returns {Array<Array<number>>} Ordered points forming a contour (approximate)
     */
    orderPointsIntoContour(points) {
        if (points.length <= 3) return points;

        // Find centroid
        let cx = 0,
            cy = 0;
        for (const point of points) {
            cx += point[0];
            cy += point[1];
        }
        cx /= points.length;
        cy /= points.length;

        // Sort points by angle from centroid
        const sortedPoints = [...points];
        sortedPoints.sort((a, b) => {
            const angleA = Math.atan2(a[1] - cy, a[0] - cx);
            const angleB = Math.atan2(b[1] - cy, b[0] - cx);
            return angleA - angleB;
        });

        return sortedPoints;
    }

    /**
     * Generate mathematical equations from contours
     * @param {Array<Array<Array<number>>>} contours - Array of contours
     * @param {number} complexity - Level of detail
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @returns {Object} Generated equations and Desmos URL
     */
    generateEquations(contours, complexity, width, height) {
        const lines = this.generateLines(contours, complexity, width, height);
        const parametricCurves = this.generateParametricCurves(contours, complexity);
        const circles = this.generateCircles(contours, complexity);

        // Generate Desmos URL
        const url = this.generateDesmosUrl(lines, parametricCurves, circles);

        return {
            lines,
            parametricCurves,
            circles,
            url
        };
    }

    /**
     * Generate line equations from contours. This uses a random sampling approach.
     * A more accurate method would involve fitting lines to segments of contours.
     * @param {Array<Array<Array<number>>>} contours - Array of contours
     * @param {number} complexity - Level of detail
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @returns {Array<Object>} Array of line objects with points, equation, and desmos latex
     */
    generateLines(contours, complexity, width, height) {
        const lines = [];
        const lineCount = 10 + complexity * 2; // Number of lines to generate

        // Flatten all contour points
        let allPoints = contours.flat();

        if (allPoints.length === 0) {
            return []; // No points to generate lines from
        }

        // Sample points based on complexity to avoid too many calculations
        const samplingRate = Math.max(1, Math.floor(allPoints.length / (50 * complexity))); // Adjust sampling
        const sampledPoints = [];
        for (let i = 0; i < allPoints.length; i += samplingRate) {
            sampledPoints.push(allPoints[i]);
        }

        // Generate lines by picking random points
        for (let attempt = 0; attempt < lineCount * 5 && lines.length < lineCount; attempt++) { // Increased attempts
            // Select two random points to form a line
            const idx1 = Math.floor(Math.random() * sampledPoints.length);
            let idx2 = Math.floor(Math.random() * sampledPoints.length);

            // Ensure we get two different points
            if (idx1 === idx2) {
                idx2 = (idx2 + 1) % sampledPoints.length;
            }

            const p1 = sampledPoints[idx1];
            const p2 = sampledPoints[idx2];

            // Ensure minimum distance between points
            const dist = this.distance(p1, p2);
            if (dist < Math.max(width, height) * 0.05) continue; // Minimum distance based on image size

            // Calculate line equation y = mx + b
            const dx = p2[0] - p1[0];
            const dy = p2[1] - p1[1];

            // Avoid near-vertical lines by using x=c form or a large slope
            let m, b, desmosLatex;
            if (Math.abs(dx) < 0.001) { // Vertical line
                // Normalize x-coordinate for Desmos
                const normX = (p1[0] / width) * 20 - 10;
                desmosLatex = `x=${normX.toFixed(4)}`;
                m = Infinity; // Represent vertical line with infinite slope
                b = p1[0]; // Represent x-intercept for vertical line
            } else {
                m = dy / dx;
                b = p1[1] - m * p1[0];

                // Normalize coordinates to range [-10, 10] for Desmos
                const normalizeX = (x) => (x / width) * 20 - 10;
                const normalizeY = (y) => 10 - (y / height) * 20; // Invert Y axis for typical Desmos view

                const normP1 = [normalizeX(p1[0]), normalizeY(p1[1])];
                const normP2 = [normalizeX(p2[0]), normalizeY(p2[1])];
                const normDx = normP2[0] - normP1[0];
                const normDy = normP2[1] - normP1[1];

                if (Math.abs(normDx) < 0.001) { // Normalized vertical line
                    desmosLatex = `x=${normP1[0].toFixed(4)}`;
                } else {
                     const normM = normDy / normDx;
                     const normB = normP1[1] - normM * normP1[0];
                     desmosLatex = `y=${normM.toFixed(4)}x+${normB.toFixed(4)}`;
                }
            }


            lines.push({
                points: [p1, p2], // Original points
                equation: `y = ${m.toFixed(4)}x + ${b.toFixed(4)}`, // Original coordinates equation
                desmos: desmosLatex // Desmos latex
            });
        }

        return lines;
    }

    /**
     * Generate parametric curve equations from contours using Fourier series fitting.
     * @param {Array<Array<Array<number>>>} contours - Array of contours
     * @param {number} complexity - Level of detail
     * @returns {Array<Object>} Array of parametric curve objects
     */
    generateParametricCurves(contours, complexity) {
        const parametricCurves = [];
        const maxCurves = 15; // Limit the number of parametric curves

        // Process each contour up to the limit
        for (let i = 0; i < Math.min(contours.length, maxCurves); i++) {
            const contour = contours[i];
            if (contour.length < 5) continue; // Need enough points to fit a curve

            // Normalize contour points to [-10, 10] range for Desmos
            const normalizedContour = this.normalizeContour(contour, this.canvas.width, this.canvas.height);

            // Number of coefficients (degree) based on complexity
            const degree = Math.min(10, complexity * 2); // Increased max degree

            // Fit Fourier series
            const xCoeffs = this.fitFourierSeries(normalizedContour.map(p => p[0]), degree);
            const yCoeffs = this.fitFourierSeries(normalizedContour.map(p => p[1]), degree);

            // Create equation strings for Desmos
            let xDesmos = "";
            let yDesmos = "";

            // Add constant term (a0)
            xDesmos += xCoeffs.a0.toFixed(4);
            yDesmos += yCoeffs.a0.toFixed(4);

            // Add cosine terms (a_k)
            for (let k = 0; k < degree; k++) {
                if (Math.abs(xCoeffs.a[k]) > 1e-6) { // Add only if coefficient is significant
                    const sign = xCoeffs.a[k] >= 0 ? " + " : " - ";
                    xDesmos += `${sign}${Math.abs(xCoeffs.a[k]).toFixed(4)}\\cos(${k + 1}\\pi t)`;
                }

                if (Math.abs(yCoeffs.a[k]) > 1e-6) {
                    const sign = yCoeffs.a[k] >= 0 ? " + " : " - ";
                    yDesmos += `${sign}${Math.abs(yCoeffs.a[k]).toFixed(4)}\\cos(${k + 1}\\pi t)`;
                }
            }

            // Add sine terms (b_k)
            for (let k = 0; k < degree; k++) {
                if (Math.abs(xCoeffs.b[k]) > 1e-6) {
                    const sign = xCoeffs.b[k] >= 0 ? " + " : " - ";
                    xDesmos += `${sign}${Math.abs(xCoeffs.b[k]).toFixed(4)}\\sin(${k + 1}\\pi t)`;
                }

                if (Math.abs(yCoeffs.b[k]) > 1e-6) {
                    const sign = yCoeffs.b[k] >= 0 ? " + " : " - ";
                    yDesmos += `${sign}${Math.abs(yCoeffs.b[k]).toFixed(4)}\\sin(${k + 1}\\pi t)`;
                }
            }

             // Add an equation property with a more readable format (optional)
             let xEq = `x(t) = ${xDesmos.replace(/\\cos/g, 'cos').replace(/\\sin/g, 'sin').replace(/\\pi/g, 'π')}`;
             let yEq = `y(t) = ${yDesmos.replace(/\\cos/g, 'cos').replace(/\\sin/g, 'sin').replace(/\\pi/g, 'π')}`;


            parametricCurves.push({
                points: contour, // Original points
                x_equation: xEq,
                y_equation: yEq,
                x_desmos: xDesmos,
                y_desmos: yDesmos
            });
        }

        return parametricCurves;
    }

    /**
     * Normalize contour points to a specific range (e.g., [-10, 10] for Desmos)
     * while maintaining aspect ratio relative to the original image.
     * @param {Array<Array<number>>} contour - Array of [x, y] coordinates
     * @param {number} originalWidth - Original image width
     * @param {number} originalHeight - Original image height
     * @returns {Array<Array<number>>} Normalized contour points
     */
    normalizeContour(contour, originalWidth, originalHeight) {
         return contour.map(([x, y]) => [
            (x / originalWidth) * 20 - 10,          // Normalize x from [0, width] to [-10, 10]
            10 - (y / originalHeight) * 20          // Normalize y from [0, height] to [-10, 10] and invert Y
        ]);
    }


    /**
     * Fit Fourier series to data points. Assumes data corresponds to a function of t in [0, 1].
     * @param {Array<number>} data - Array of values (e.g., x-coordinates or y-coordinates)
     * @param {number} degree - Number of coefficients (excluding a0)
     * @returns {Object} Fourier coefficients { a0, a: [a1, a2, ...], b: [b1, b2, ...] }
     */
    fitFourierSeries(data, degree) {
        const n = data.length;
        if (n === 0) return { a0: 0, a: new Array(degree).fill(0), b: new Array(degree).fill(0) };

        const a0 = data.reduce((sum, val) => sum + val, 0) / n;
        const a = new Array(degree).fill(0);
        const b = new Array(degree).fill(0);

        // Calculate coefficients
        for (let k = 0; k < degree; k++) {
            for (let i = 0; i < n; i++) {
                const t = i / (n - 1); // Parameter t ranges from 0 to 1
                a[k] += data[i] * Math.cos((k + 1) * Math.PI * t);
                b[k] += data[i] * Math.sin((k + 1) * Math.PI * t);
            }
            a[k] = (2 / n) * a[k];
            b[k] = (2 / n) * b[k];
        }

        return { a0, a, b };
    }

    /**
     * Generate circle equations from contours. This attempts to fit circles to contours.
     * A simplified approach is also used to generate random circles if needed.
     * @param {Array<Array<Array<number>>>} contours - Array of contours
     * @param {number} complexity - Level of detail
     * @returns {Array<Object>} Array of circle objects with center, radius, and equation
     */
    generateCircles(contours, complexity) {
        const circles = [];
        const targetCircleCount = 3 + complexity; // Number of circles to generate

        // Try to fit circles to some contours
        for (let i = 0; i < Math.min(contours.length, targetCircleCount); i++) {
            const contour = contours[i];
            if (contour.length < 10) continue; // Need enough points to fit a circle reliably

            // Try to fit a circle to the contour points
            const circle = this.fitCircle(contour);
            if (circle) {
                 // Normalize circle parameters for Desmos [-10, 10] range
                const [cx, cy, r] = circle;

                // Create equation string for Desmos
                const equation = `(x-${cx.toFixed(2)})^2+(y-${cy.toFixed(2)})^2=${(r*r).toFixed(2)}`;

                circles.push({
                    center: [cx, cy], // Normalized center
                    radius: r, // Normalized radius
                    equation: equation
                });
            }
        }

        // If we still need more circles, generate random ones based on image dimensions
        if (circles.length < targetCircleCount) {
             // Flatten all contour points to find a general area
             const allPoints = contours.flat();
             if (allPoints.length === 0) {
                 // If no contours, use image center as a base
                 allPoints.push([this.canvas.width / 2, this.canvas.height / 2]);
             }

            for (let i = circles.length; i < targetCircleCount; i++) {
                // Get a random center point from the contours or image
                const centerIdx = Math.floor(Math.random() * allPoints.length);
                const [x, y] = allPoints[centerIdx];

                // Random radius between 2% and 10% of the smaller image dimension
                const minDim = Math.min(this.canvas.width, this.canvas.height);
                const r = minDim * (0.02 + Math.random() * 0.08);

                 // Normalize for Desmos (-10 to 10 range)
                const normX = (x / this.canvas.width) * 20 - 10;
                const normY = 10 - (y / this.canvas.height) * 20;
                const normR = (r / this.canvas.width) * 20; // Scale radius relative to width for consistency

                const equation = `(x-${normX.toFixed(2)})^2+(y-${normY.toFixed(2)})^2=${(normR*normR).toFixed(2)}`;

                circles.push({
                    center: [normX, normY], // Normalized center
                    radius: normR, // Normalized radius
                    equation: equation
                });
            }
        }


        return circles;
    }

    /**
     * Fit a circle to a set of points using the algebraic method (least squares).
     * @param {Array<Array<number>>} points - Array of [x, y] coordinates
     * @returns {Array<number>|null} [cx, cy, r] (normalized for Desmos) or null if no good fit
     */
    fitCircle(points) {
        if (points.length < 3) return null;

        // Normalize points to a smaller scale first to improve numerical stability
        // Use the bounding box of the points for local normalization
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const [x, y] of points) {
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }

        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        const scale = Math.max(rangeX, rangeY);
        const offsetX = minX;
        const offsetY = minY;

        const scaledPoints = points.map(([x, y]) => [(x - offsetX) / scale, (y - offsetY) / scale]);


        let sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0, sumXY = 0, sumX3 = 0, sumY3 = 0, sumXY2 = 0, sumX2Y = 0;
        const n = scaledPoints.length;

        for (const [x, y] of scaledPoints) {
            const x2 = x * x;
            const y2 = y * y;
            sumX += x;
            sumY += y;
            sumX2 += x2;
            sumY2 += y2;
            sumXY += x * y;
            sumX3 += x2 * x;
            sumY3 += y2 * y;
            sumXY2 += x * y2;
            sumX2Y += x2 * y;
        }

        // Solve linear system for A, B, C where equation is x^2 + y^2 + Ax + By + C = 0
        // Center (cx, cy) = (-A/2, -B/2), Radius r = sqrt((A^2 + B^2)/4 - C)

        // Matrix coefficients
        const M = [
            [sumX2, sumXY, sumX],
            [sumXY, sumY2, sumY],
            [sumX, sumY, n]
        ];

        // Right side vector
        const V = [-sumX3 - sumXY2, -sumY3 - sumX2Y, -sumX2 - sumY2];

        // Solve the linear system using Cramer's rule or matrix inversion
        // Using a simple approach assuming the determinant is non-zero
        const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

        if (Math.abs(det) < 1e-9) { // Check for near-singular matrix
             // Try a simpler 3-point circle fit if available
             if (points.length >= 3) {
                 const p1 = points[0], p2 = points[Math.floor(points.length / 2)], p3 = points[points.length - 1];
                 return this.threePointCircle(p1, p2, p3);
             }
             return null; // Cannot fit a circle
        }

        // Calculate A, B, C using Cramer's rule (simplified)
        const detA = V[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                     M[0][1] * (V[1] * M[2][2] - M[1][2] * V[2]) +
                     M[0][2] * (V[1] * M[2][1] - M[1][1] * V[2]);

        const detB = M[0][0] * (V[1] * M[2][2] - M[1][2] * V[2]) -
                     V[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                     M[0][2] * (M[1][0] * V[2] - V[1] * M[2][0]);

        const detC = M[0][0] * (M[1][1] * V[2] - V[1] * M[2][1]) -
                     M[0][1] * (M[1][0] * V[2] - V[1] * M[2][0]) +
                     V[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

        const A = detA / det;
        const B = detB / det;
        const C = detC / det;

        // Calculate center and radius in scaled coordinates
        const scaledCx = -A / 2;
        const scaledCy = -B / 2;
        const scaledR2 = (A * A + B * B) / 4 - C;

        if (scaledR2 < 0) return null; // Not a real circle
        const scaledR = Math.sqrt(scaledR2);

        // Convert back to original coordinates
        const cx = scaledCx * scale + offsetX;
        const cy = scaledCy * scale + offsetY;
        const r = scaledR * scale;

        // Normalize coordinates for Desmos (-10 to 10 range)
        const normX = (cx / this.canvas.width) * 20 - 10;
        const normY = 10 - (cy / this.canvas.height) * 20;
        const normR = (r / this.canvas.width) * 20; // Scale radius relative to width

        // Basic check if the circle is too large or too small
        if (normR > 20 || normR < 0.1) {
            return null; // Reject circles that are too big or too small in Desmos space
        }

         // Optional: Check how well the fitted circle matches the points
         let error = 0;
         for (const [x, y] of points) {
             const distToCenter = Math.sqrt((x - cx)**2 + (y - cy)**2);
             error += Math.abs(distToCenter - r);
         }
         const averageError = error / n;

         // Reject if the average error is too high (e.g., more than 10% of the radius)
         if (r > 0 && averageError / r > 0.2) { // Increased tolerance slightly
              return null;
         }


        return [normX, normY, normR];
    }

    /**
     * Generate Desmos calculator URL with equations
     * @param {Array<Object>} lines - Array of line objects
     * @param {Array<Object>} parametricCurves - Array of parametric curve objects
     * @param {Array<Object>} circles - Array of circle objects
     * @returns {string} Desmos URL
     */
    generateDesmosUrl(lines, parametricCurves, circles) {
        // Create a Desmos calculator state
        const expressions = [];

        // Add line equations (limit to 20)
        for (let i = 0; i < Math.min(lines.length, 30); i++) { // Increased line limit
            expressions.push({
                type: "expression",
                latex: lines[i].desmos
            });
        }

        // Add parametric curves (limit to 15)
        for (let i = 0; i < Math.min(parametricCurves.length, 20); i++) { // Increased parametric limit
            const curve = parametricCurves[i];
            const paramEq = `\\left(${curve.x_desmos},${curve.y_desmos}\\right)\\left\\{0\\le t\\le1\\right\\}`;
            expressions.push({
                type: "expression",
                latex: paramEq
            });
        }

        // Add circles (limit to 10)
        for (let i = 0; i < Math.min(circles.length, 15); i++) { // Increased circle limit
            expressions.push({
                type: "expression",
                latex: circles[i].equation
            });
        }

        // Build calculator state
        const state = {
            version: 7,
            expressions: { list: expressions },
            graph: {
                viewport: { xmin: -10, ymin: -10, xmax: 10, ymax: 10 },
                // Optional: Add settings like axis labels or grid
                // xAxisLabel: 'X',
                // yAxisLabel: 'Y',
                // showGrid: false,
                // showXAxis: false,
                // showYAxis: false
            }
        };

        // Encode state as base64 URL-safe string
        const jsonState = JSON.stringify(state);
        // Use TextEncoder and base64url encoding for better handling of unicode
        const encoder = new TextEncoder();
        const data = encoder.encode(jsonState);
        const base64 = btoa(String.fromCharCode(...new Uint8Array(data))); // Standard base64 first

        // Convert to base64url
        const encodedState = base64
            .replace(/\+/g, '-')
            .replace(/\//g, '_')
            .replace(/=+$/, '');

        return `https://www.desmos.com/calculator?state=${encodedState}`;
     }

     /**
      * Helper function to fit a circle through three points.
      * Used as a fallback for `fitCircle` when the matrix is singular.
      * @param {Array<number>} p1 - First point [x, y]
      * @param {Array<number>} p2 - Second point [x, y]
      * @param {Array<number>} p3 - Third point [x, y]
      * @returns {Array<number>|null} [cx, cy, r] (normalized for Desmos) or null if points are collinear
      */
     threePointCircle(p1, p2, p3) {
         const [x1, y1] = p1;
         const [x2, y2] = p2;
         const [x3, y3] = p3;

         // Check for collinearity
         const collinearityCheck = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1);
         if (Math.abs(collinearityCheck) < 1e-9) {
             return null; // Points are collinear
         }

         const A = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
         const B = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1);
         const C = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2);
         const D = (x1 * x1 + y1 * y1) * (x2 * y3 - x3 * y2) + (x2 * x2 + y2 * y2) * (x3 * y1 - x1 * y3) + (x3 * x3 + y3 * y3) * (x1 * y2 - x2 * y1);

         // Calculate center
         const cx = -B / (2 * A);
         const cy = -C / (2 * A);

         // Calculate radius
         const r = Math.sqrt((cx - x1)**2 + (cy - y1)**2);

         // Normalize coordinates for Desmos (-10 to 10 range)
         const normX = (cx / this.canvas.width) * 20 - 10;
         const normY = 10 - (cy / this.canvas.height) * 20;
         const normR = (r / this.canvas.width) * 20;

         // Basic check if the circle is too large or too small in Desmos space
         if (normR > 20 || normR < 0.1) {
             return null;
         }

         return [normX, normY, normR];
     }
 }

export default ImageProcessor;