# Key Points for the key approaches in computer vision

## Sections

- [Images](#images-and-colour)
  - [Conversion formulae](#colour-conversion-formulae)
  - [Image noise](#image-noise)
- [Histograms](#histograms)
- [Binary](#binary)
- [Region Segmentation and Connectivity](#region-segmentation-and-connectivity)

## Images and Colour

#### Colour Conversion Formulae

- **RGB to Greyscale: `Y = 0.299R + 0.587G + 0.114B`**
- CMY (Cyan Yellow Magenta): `C = 255-R`, `M = 255-G`, `Y = 255-B`
- YUV: `Y = 0.299R + 0.587G + 0.114B`, `U = 0.492*(B-Y)`, `V = 0.877*(R-Y)`
- **HLS - (Hue Luminance Saturation) - Separates colour and luminance - H = 0...360, L = 0...1, S = 0...1**
  - ![RGB TO HLS CONVERSION](../imgs/RGB2HLS.png)

#### Image Noise

- Signal to Noise: `Clean Pixels/Noisy Pixels`
- Types of Noise: 
  - Gaussian - Gaussian distributed, good approximation to real noise
  - Salt and Pepper - Impulse noise, noise is max or min values
- Smoothing: the process of removing/reducing noise - risk of blurring sharp edges, loss of quality?
- Linear Smoothing Transformations
  - Image averaging - average based on n images (assumes: static camera & scene, statistical independence)
  - Local Averaging and Gaussian Smoothing: Averaging filters - local neighbourhood, can use different masks

<img src="../imgs/masks.png" alt="AVERAGING FILTERS/MASKS" height="200"/>

- Non-Linear Smoothing Transformations 
  - Define multiple masks/regions - need to choose size and shape - use the average of most homogeneous mask
  - For each point - calculate dispersion, assign output point average of mask with min dispersion
  - ![NON-LINEAR SMOOTHING FORMULA](../imgs/non-linear-smoothing.png)
  - Iterative application: convergence.
  - Median Filter - use median value, ignore average, noise resistant, computationally expensive, damages lines
  - Bilateral Filter - weight local pixels, distance from centre, difference in colour/instensity, preserves edges, but causes staircase
    - ![Weighting Pixels](../imgs/pixel-weighting.png)
    - ![Pixel Weight](../imgs/pixel-weight.png)
- Image pyramids: process at multiple scales efficiently 
  - Technique: Smooth image (e.g. Gaussian), sub-sample (generally by factor of 2)


## Histograms

- Represent global information about an image - independent of location/orientation - can use for classification
  - Not unique, different images can have v similar histograms
- Local minima/maxima are useful - but histograms are noisy, many local min/max
  - Smoothing - replace each value in histogram with local average e.g. `h[i] = (h[i-1]+h[i]+h[i+1])/3`
    - First and last values? Discard them/Replace with constant/Wraparound?
- Colour histograms - per channel histogram - colour space influences usefulness
- 3D histograms - Channels aren't independent - consider all simultaneously
  - 8 bits per channel is too much - 2 bits per channel - 64 cells

#### Equalisation
- Deals with poor contrast - redistribute greyscale in an image
  - When working with colour images, only change the luminance values

<img src="../imgs/equalisation.png" alt="Equalisation psuedo code" height="200"/>

#### Histogram Comparison
- Use to find similar images - compares colour distributions

<img src="../imgs/histogram-comparison.png" alt="Histogram Comparison Metrics" height="230">

- Can also use Earth Mover's Distance: min cost of turning one distribution into another - easy for 1D, harder for 3D
- `EMD(-1) = 0`<br>`EMD(i) = h1(i) + EMD(i-1) - h2(i)`<br>`Earth Mover's Distance = sum of all EMD`

#### Back Projection
- Colour selection based on samples
  - Obtain a sample set - histogram the samples - normalize the histogram (max 1.0) - back project normalized histogram onto image f(i,j)
  - Result is **a probability image, p(i,j) indicating the similarity between f(i,j) and sample set**
  - Key considerations - size of hist bins (esp. when using limited sample set) & colour space
  
#### K-Means  Clustering
- Method to reduce the variation in 3D colour space
  - Algorithm detects k exemplars to best represent image (pre-determined k) - colours are associated with the nearest exemplar forming clusters
  - Approach:<br>**Get Starting exemplars** (random (leads to non. determinism)? manual selection? even distribution?)<br>**First Pass:** for all pixels, allocate to nearest exemplar - shift exemplar to c.o.g of assigned colours after each assignment<br>**Second Pass:** use final exemplars from first pass & realocate all pixels
  - **How many exemplars to use?** Less may reflect dominant features better (e.g. felt on snooker table), but more may get important detail (e.g. balls on snooker table)
  - **[Davies-Bouldin](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index#Definition) index to find k:** iterates over the clusters generated - calculates the inter-cluster similarity - takes the worst result for each cluster, sums them and averages them - take k to be the number for which this value is lowest.
  - However: doesn't account for cluster size - cannot account for information retrieval

## Binary

#### Thresholding
- Used to convert a greyscale image into a binary image
  - Brief: Using some threshold, T, all values GTE set to 1, all values LT set to 0
- **Threshold Detection**
  - Manual Setting - magic number - what if lighting changes?
  - Automatic Threshold detection - vital - lighting declines consistently over time
  - Notation: `Image: f(i,j) Histogram: h(g) Probability Distribution: p(g) = h(g)/sum(h)`
- **Bi-Modal Histogram Analysis:** assume fg & bg are centered around 2 distinct greyscale values - use antimode between peaks
  - Drawbacks... histograms are noisy - smoothing/variable step would skew the anti-mode
- **Optimal Thresholding:** aims to deal with failings of bi-modal - i.e. handles noise, handles close modes
  - Model the histogram as the sum of two normal distributions
  - <img src="../imgs/optimal-thresholding.png" alt="Optimal Thresholding" width="200" />
  - What the maths is doing:<br>w<sub>b</sub>(T<sup>t</sup>) calculates the weighting of the background for this threshold - literally speaking: the value between 0 and 1 which represents the amount of the image below the threshold<br>w<sub>f</sub>(T<sup>t</sup>) = 1 - w<sub>b</sub>(T<sup>t</sup>)<br>And then we calculate the mean of the fg distribution and bg distribution
  - The threshold is set to the halfway point between the two means. We iterate until the threshold is consistent.
- **Otsu Thresholding:** what if the distribution isn't normal - minimize spead on either side of threshold
  - Consider all thresholds, select the threshold which minimizes the within class variance
  - <img src="../imgs/otsu-thresholding-raw.png" alt="Otsu thresholding raw" width="350"/>
  - Relatively simple formula: calculate w<sub>b</sub>(T), w<sub>f</sub>(T) and the means as before, and then calculate the variance - i.e. instead of the sum of p(g)\*g as before, we use the sum of p(g)\*(g-<mean>)<sup>2</sup>
  - Can simplify this - minimum within class variance = maximum between class variance
  - Take max value for: <img src="../imgs/otsu-thresholding-simplified.png" alt="simplified otsu" width="200"/>

#### Alternative Thresholding
- **Adaptive Thresholding:** divide an image into sub-images, threshold each sub-image, interpolate thresholds for each point using bilinear interpolation
- **Band Thresholding:** uses 2 thresholds - (theoretically) one below and one above object pixels
  - Border detector?
- **Semi Thresholding:** not used for much other than visualizing
  - If pixel GTE T, pixel retains greyscale value, else 0
- **Multi-Level Thresholding:** threshold all colours separately?
  - Can threshold in 3D colour space - define a 3D region in space, accept pixels within this space
  - Produces rough/pixelated binary images - require post processing

#### Mathematical Morphology
- Normal smoothing operations are inappropriate for binary images - we need clear edges
- Methods for removing noise in binary images - treat images as sets
- **Erosion: Removes noise and narrow bridges - use to remove borders**
  - Minkowski set subtraction
- **Dilation: Fill small holes and gulfs - used to add borders**
  - Minkowski set addition
- **Opening: Erosion followed by dilation - removes small objects(white pixels)**
  - Maintains approximate object sizes
- **Closing: Dilation followed by Erosion - removes small holes(black pixels)**
  - Maintains approximate object size, but distorts the shape
- Using mathematical morphology on a greyscale/colour images is also possible
  - Each level is considered to be set - all points GTE a given level undergo the morphology
  - Can be used to determine local maxima/minima

## Region Segmentation and Connectivity
