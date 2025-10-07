# %% [markdown]
# # CSE 252A Computer Vision I, Fall 2025 - Assignment 0

# %% [markdown]
# Instructor: Ben Ochoa
# 
# Assigment due: Wed, Oct 8, 11:59 PM

# %% [markdown]
# **Name:** Evan Cheng
# 
# **PID:** A69042831

# %% [markdown]
# ## Prior knowledge + certification of commencement of academic activity
# 
# In every course at UC San Diego, per the US Department of Education, we are now required to certify whether students have commenced academic activity for a class to be counted towards eligibility for Title IV federal financial aid. This certification must be completed during the first two weeks of instruction.
# 
# For CSE 252A, this requirement will be fulfilled via an ungraded prior knowledge quiz, which will assist the instructional team by providing information about your background coming into the course. Although this will be part of the first assignment, you are welcome to complete the quiz early. In [Canvas](https://canvas.ucsd.edu), go to the CSE 252A course and navigate to Quizzes. Then, click on the "First Day Survey: Prior Knowledge #FinAid"

# %% [markdown]
# ## Instructions
# 
# Please answer the questions below using Python in the attached Jupyter notebook and follow the guidelines below:
# 
# - This assignment must be completed **individually**. For more details, please follow the Academic Integrity Policy and Collaboration Policy on [Canvas](https://canvas.ucsd.edu).
# 
# - All the solutions must be written in this Jupyter notebook.
# 
# - You may use basic algebra packages (e.g. `NumPy`, `SciPy`, etc) but you are not allowed to use the packages that directly solve the problems. Feel free to ask the instructor and the teaching assistants if you are unsure about the packages to use.
# 
# - It is highly recommended that you begin working on this assignment early.
# 
# - You must **submit 3 files: the Notebook, the PDF and the python file** (i.e. the `.ipynb`, the `.pdf` and the `.py` files) on Gradescope. **You must mark each problem on Gradescope in the pdf.**
#     - To convert the notebook to PDF, you can choose one way below:
# 
#         - You may first export the notebook as HTML, and then print the web page as PDF
# 
#             - e.g., in Chrome: File $\rightarrow$ Save and Export Notebook as $\rightarrow$ "HTML"; or in VScode: Open the Command Palette by pressing Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (macOS), search for Jupyter: Export to HTML
#     
#             - Open the saved web page and right click $\rightarrow$ Print... $\rightarrow$ Choose "Destination: Save as PDF" and click "Save")
# 
#         - If you have XeTex installed on your machine, you may directly export the notebook as PDF: e.g., in Chrome, File $\rightarrow$ Save and Export Notebook as $\rightarrow$ "PDF"
# 
#         - You may use [nbconvert](https://nbconvert.readthedocs.io/en/latest/install.html) to convert the ipynb file to pdf using the following command
#         `jupyter nbconvert --allow-chromium-download --to webpdf filename.ipynb`
# 
#     - To convert the notebook to python file, you can choose one way below:
# 
#         - You may directly export the notebook as py: e.g., in Chrome, File $\rightarrow$ Save and Export Notebook as $\rightarrow$ "Executable script"; or in VScode: Open the Command Palette and search for Jupyter: Export to Python Script
# 
#         - You may use [nbconvert](https://nbconvert.readthedocs.io/en/latest/install.html) to convert the ipynb file to python file using the following command
#     `jupyter nbconvert --to script filename.ipynb --output output_filename.py`
# 
# - Please make sure the content in each cell (e.g. code, output images, printed results, etc.) are clearly visible and are not cut-out or partially cropped in your final PDF file.
# 
# - While submitting on gradescope, please make sure to assign the relevant pages in your PDF submission for each problem.
# 
# **Late Policy:** Assignments submitted late will receive a 15% grade reduction for each 12 hours late (i.e., 30% per day). Assignments will not be accepted 72 hours after the due date. If you require an extension (for personal reasons only) to a due date, you must request one as far in advance as possible. Extensions requested close to or after the due date will only be granted for clear emergencies or clearly unforeseeable circumstances.

# %% [markdown]
# ## Introduction
# 
# Welcome to **CSE 252A Computer Vision I**!
# 
# This course provides a comprehensive introduction to computer vision providing broad coverage including low level vision (image formation, photometry, color, image feature detection), inferring 3D properties from images (shape-from-shading, stereo vision, motion interpretation) and object recognition.
# 
# We will use a variety of tools (e.g. some packages and operations) in this class that may require some initial configuration. To ensure smooth progress, we will setup the majority of the tools to be used in this course in this **Assignment 0**. You will also practice some basic image manipulation techniques.

# %% [markdown]
# ## Piazza, Gradescope and Python
# 
# **Piazza**
# 
# All students are automatically added to the class in [Piazza](https://piazza.com/) once enrolled in this class. You can get access to it from [Canvas](https://canvas.ucsd.edu). You'll be able to ask the professor, the TAs and your classmates questions on Piazza. Class announcements will be made using Piazza, so make sure you check your email or Piazza frequently.
# 
# **Gradescope**
# 
# All students are automatically added to the class in [Gradescope](https://www.gradescope.com/) once enrolled in this class. You can also get access to it from [Canvas](https://canvas.ucsd.edu). All the assignments are required to be submitted to Gradescope for grading. Make sure that you mark each page for different problems.  
# 
# **Python**
# 
# We will use the Python programming language for all assignments in this course, with a few popular libraries (`NumPy`, `Matplotlib`). Assignments will be given in the format of web-based Jupyter notebook that you are currently viewing. We expect that many of you have some experience with `Python` and `NumPy`. And if you have previous knowledge in `MATLAB`, check out the [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) page. The section below will serve as a quick introduction to `NumPy` and some other libraries.

# %% [markdown]
# ## Getting Started with NumPy
# 
# `NumPy` is the fundamental package for scientific computing with Python. It provides a powerful N-dimensional array object and functions for working with these arrays. Some basic use of this packages is shown below. This is **NOT** a problem, but you are highly recommended to run the following code with some of the input changed in order to understand the meaning of the operations.

# %% [markdown]
# ### Arrays

# %%
import numpy as np             # Import the NumPy package

v = np.array([1, 2, 3])        # A 1D array
print(v)
print(v.shape)                 # Print the size / shape of v
print("1D array:", v, "Shape:", v.shape)

v = np.array([[1], [2], [3]])  # A 2D array
print("2D array:", v, "Shape:", v.shape) # Print the size of v and check the difference.

# You can also attempt to compute and print the following values and their size.

v = v.T                        # Transpose of a 2D array
m = np.zeros([3, 4])           # A 2x3 array (i.e. matrix) of zeros
v = np.ones([1, 3])            # A 1x3 array (i.e. a row vector) of ones
v = np.ones([3, 1])            # A 3x1 array (i.e. a column vector) of ones
m = np.eye(4)                  # Identity matrix
m = np.random.rand(2, 3)       # A 2x3 random matrix with values in [0, 1] (sampled from uniform distribution)

# %% [markdown]
# ### Array Indexing

# %%
import numpy as np

print("Matrix")
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # Create a 3x3 array.
print(m)

print("\nAccess a single element")
print(m[0, 1])                        # Access an element
m[1, 1] = 100                         # Modify an element
print("\nModify a single element")
print(m)

print("\nAccess a subarray")
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # Create a 3x3 array.
print(m[1, :])                        # Access a row (to 1D array)
print(m[1:2, :])                      # Access a row (to 2D array)
print(m[1:3, :])                      # Access a sub-matrix
print(m[1:, :])                       # Access a sub-matrix

print("\nModify a subarray")
m = np.array([[1,2,3], [4,5,6], [7,8,9]]) # Create a 3x3 array.
v1 = np.array([1,1,1])
m[0] = v1
print(m)
m = np.array([[1,2,3], [4,5,6], [7,8,9]]) # Create a 3x3 array.
v1 = np.array([1,1,1])
m[:,0] = v1
print(m)
m = np.array([[1,2,3], [4,5,6], [7,8,9]]) # Create a 3x3 array.
m1 = np.array([[1,1],[1,1]])
m[:2,:2] = m1
print(m)

print("\nTranspose a subarray")
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # Create a 3x3 array.
print(m[1, :].T)                                # Notice the difference of the dimension of resulting array
print(m[1:2, :].T)
print(m[1:, :].T)
print(np.transpose(m[1:, :], axes=(1,0)))       # np.transpose() can be used to transpose according given axes list.

print("\nReverse the order of a subarray")
print(m[1, ::-1])                               # Access a row with reversed order (to 1D array)

# Boolean array indexing
# Given a array m, create a new array with values equal to m
# if they are greater than 2, and equal to 0 if they less than or equal to 2
m = np.array([[1, 2, 3], [4, 5, 6]])
m[m > 2] = 0
print("\nBoolean array indexing: Modify with a scaler")
print(m)

# Given a array m, create a new array with values equal to those in m
# if they are greater than 0, and equal to those in n if they less than or equal 0
m = np.array([[1, 2, -3], [4, -5, 6]])
n = np.array([[1, 10, 100], [1, 10, 100]])
n[m > 0] = m[m > 0]
print("\nBoolean array indexing: Modify with another array")
print(n)

# %% [markdown]
# ### Array Dimension Operation

# %%
import numpy as np

print("Matrix")
m = np.array([[1, 2], [3, 4]]) # Create a 2x2 array.
print(m, m.shape)

print("\nReshape")
re_m = m.reshape(1,2,2)  # Add one more dimension at first.
print(re_m, re_m.shape)
re_m = m.reshape(2,1,2)  # Add one more dimension in middle.
print(re_m, re_m.shape)
re_m = m.reshape(2,2,1)  # Add one more dimension at last.
print(re_m, re_m.shape)

print("\nStack")
m1 = np.array([[1, 2], [3, 4]]) # Create a 2x2 array.
m2 = np.array([[1, 1], [1, 1]]) # Create a 2x2 array.
print(np.stack((m1,m2)))

print("\nConcatenate")
m1 = np.array([[1, 2], [3, 4]]) # Create a 2x2 array.
m2 = np.array([[1, 1], [1, 1]]) # Create a 2x2 array.
print(np.concatenate((m1,m2)))
print(np.concatenate((m1,m2), axis=0))
print(np.concatenate((m1,m2), axis=1))

# %% [markdown]
# ### Math Operations on Array
# **Element-wise Operations**

# %%
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(a * 3)                                            # Scalar multiplication
print(a / 2)                                            # Scalar division
print(np.round(a / 2))
print(np.power(a, 2))
print(np.log(a))
print(np.exp(a))

b = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
print(a + b)                                            # Elementwise sum
print(a - b)                                            # Elementwise difference
print(a * b)                                            # Elementwise product
print(a / b)                                            # Elementwise division
print(a == b)                                           # Elementwise comparison

# %% [markdown]
# **Broadcasting**

# %%
# Note: See https://numpy.org/doc/stable/user/basics.broadcasting.html
#       for more details.
import numpy as np
a = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float64)
b = np.array([1, 2, 3])
print(a*b)

# %% [markdown]
# **Sum and Mean**

# %%
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print("Sum of array")
print(np.sum(a))                # Sum of all array elements
print(np.sum(a, axis=0))        # Sum of each column
print(np.sum(a, axis=1))        # Sum of each row

print("\nMean of array")
print(np.mean(a))               # Mean of all array elements
print(np.mean(a, axis=0))       # Mean of each column
print(np.mean(a, axis=1))       # Mean of each row

# %% [markdown]
# **Vector and Matrix Operations**

# %%
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 1], [1, 1]])
print("Matrix-matrix product")
print(a.dot(b))                 # Matrix-matrix product
print(a.T.dot(b.T))
print(np.matmul(a,b))
print(np.matmul(a.T, b.T))
print(a @ b)
print(a.T @ b.T)

x = np.array([3, 4])
print("\nMatrix-vector product")
print(a.dot(x))                 # Matrix-vector product
print(np.matmul(a,x))
print(a @ x)

x = np.array([1, 2])
y = np.array([3, 4])
print("\nVector-vector product")
print(x.dot(y))                 # Vector-vector product

# %% [markdown]
# ### Matplotlib
# 
# `Matplotlib` is a plotting library. We will use it to show the result in this assignment.

# %%
%config InlineBackend.figure_format = 'retina' # For high-resolution.
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2., 2., 0.01) * np.pi
plt.plot(x, np.cos(x))
plt.xlabel('x')
plt.ylabel('$\cos(x)$ value') # '$...$' for a LaTeX formula.
plt.title('Cosine function')

plt.show()

# %% [markdown]
# This brief overview introduces many basic functions from `NumPy` and `Matplotlib`, but is far from complete. Check out more operations and their use in documentations for [NumPy](https://docs.scipy.org/doc/numpy/reference/) and [Matplotlib](https://matplotlib.org/).

# %% [markdown]
# ## Problem 1: Image Operations and Vectorization (15 points)
# 
# Vector operations using `NumPy` can offer a significant speedup over doing an operation iteratively on an image. The problem below will demonstrate the time it takes for both approaches to change the color of quadrants of an image.
# 
# The problem reads an image `SunGod.jpg` that you will find in the assignment folder. Two functions are then provided as different approaches for doing an operation on the image.
# 
# The function `iterative()` demonstrates the image divided into 4 parts:<br/>
# 
# (Top Left) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The original image.<br/>
# (Top Right) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Blue channel image.<br/>
# (Bottom Left) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(B,G,R) colored image.<br/>
# (Bottom Right) &nbsp;&nbsp;&nbsp;Grayscale image.<br/>
# 
# For your implementation:
# 
# (1) For the blue channel image, write your implementation to extract a single channel from a colored image. This means that from the $H\times W\times 3$ shaped image, you'll get three matrices of the shape $H\times W$ (Note that it's 2-dimensional).
# 
# (2) For the (B,G,R) colored image, write your implementation to merge those single channel images back into a 3-dimensional colored image in the reversed channels order (B,G,R).
# 
# (3) For the grayscale image, write your implementation to conduct operations with the extracted channels. You must use the following equation for computing the grayscale value from (R,G,B) channels.
# $$gray = 0.21263903 * R + 0.71516871 * G + 0.072192319 * B$$
# 
# Note: In a grayscale image, a pixel has the same grayscale value for its red (R), green (G), and blue (B) channels.
# 
# 
# 
# Your task is to follow through the code and fill the blanks in `vectorized()` function and compare the speed difference between `iterative()` and `vectorized()`.
# Make sure your final generated image in the `vectorized()` is the same as the one generated from `iterative()`.

# %%
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('SunGod.jpg') # Read an image
print("Image shape:", img.shape)           # Print image size and color depth. The shape should be (H,W,C).

plt.imshow(img)                            # Show the original image
plt.show()

# %%
import copy

# Iterative Approach. No change needed here.
def iterative(img):
    """ Iterative operation. """
    image = copy.deepcopy(img)              # Create a copy of the image matrix
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            #Top Right
            if y < image.shape[0]/2 and x >= image.shape[1]/2:
                image[y,x] = image[y,x] * np.array([0,0,1])    # Keep the blue channel
            #Bottom Left
            elif y >= image.shape[0]/2 and x < image.shape[1]/2:
                image[y,x] = [image[y,x][2], image[y,x][1], image[y,x][0]]  #(B,G,R) image
            #Bottom Right
            elif y >= image.shape[0]/2 and x >= image.shape[1]/2:
                r,g,b = image[y,x]
                image[y,x] = 0.21263903 * r + 0.71516871 * g + 0.072192319 * b
    return image

plt.imshow(iterative(img))
plt.show()

# %% [markdown]
# ### 1.1 Extract Channels (5 points)

# %%
import copy

def get_channel(img, channel):
    """ Function to extract 2D image corresponding to a channel index from a color image.
    This function should return a H*W array which is the corresponding channel of the input image. """
    img = copy.deepcopy(img)     # Create a copy so as to not change the original image
    #### Your code start here. ####
    if channel == 2:
        img = img[:,:,2]
    elif channel == 1:
        img = img[:,:,1]
    elif channel == 0:
        img = img[:,:,0]
    return img

b = 2
image_blue = get_channel(img, b)
print(image_blue.shape)
plt.imshow(image_blue)
plt.show()

# %% [markdown]
# ### 1.2 Merge Channels (5 points)

# %%
def merge_channels(img0, img1, img2):
    """ Function to merge three single channel images to form a color image.
    This function should return a H*W*3 array which merges all three single channel images
    (i.e. img0, img1, img2) in the input."""
    #### Your code start here. ####
    # Hint: There are multiple ways to implement it.
    #       1. For example, create a H*W*C array with all values as zero and
    #          fill each channel with given single channel image.
    #          You may refer to the "Modify a subarray" section in the brief NumPy tutorial above.
    #       2. You may find np.stack() / np.concatenate() / np.reshape() useful in this problem.\
    img = np.stack((img0, img1, img2), axis=-1)
    return img

blue = get_channel(img, 2)
green = get_channel(img, 1)
red = get_channel(img, 0)
test = merge_channels(blue, green, red)
print(test.shape)
plt.imshow(test)
plt.show()

# %% [markdown]
# ### 1.3 Vectorized Implement (5 points)

# %%
def vectorized(img):
    """ Vectorized operation. """
    image = copy.deepcopy(img)
    a = int(image.shape[0]/2)
    b = int(image.shape[1]/2)
    # Please also keep the red / green / blue channel respectively in the corresponding part of image
    # with the vectorized operations. You need to make sure your final generated image in this
    # vectorized() function is the same as the one generated from iterative().

    #### Your code start here. ####
    #Top Right: keep the blue channel
    blue = get_channel(image[:a,b:], 2)
    image[:a,b:] = np.stack((np.zeros_like(blue), np.zeros_like(blue), blue), axis=-1)

    #Bottom Left: (B,G,R) image
    blue = get_channel(image[a:,:b], 2)
    green = get_channel(image[a:,:b], 1)
    red = get_channel(image[a:,:b], 0)
    image[a:,:b] = merge_channels(blue, green, red)

    #Bottom Right: Grayscale image
    
    img0 = get_channel(image[a:,b:], 2)
    img1 = get_channel(image[a:,b:], 1)
    img2 = get_channel(image[a:,b:], 0)
    blue = img0 * 0.072192319
    green = img1 * 0.71516871
    red = img2 * 0.21263903
    gray = np.zeros_like(img0)
    gray = blue + green + red
    image[a:,b:] = np.stack([gray, gray, gray], axis=-1)
    
    return image

# %% [markdown]
# Now, run the following cell to compare the difference between iterative and vectorized operation.

# %%
import time

def compare():
    img = plt.imread('SunGod.jpg')
    cur_time = time.time()
    image_iterative = iterative(img)
    print("Iterative operation (sec):", time.time() - cur_time)

    cur_time = time.time()
    image_vectorized = vectorized(img)
    print("Vectorized operation (sec):", time.time() - cur_time)


    # Note: The shown figures of image_iterative and image_vectorized should be identical!
    assert np.array_equal(image_iterative, image_vectorized), "The two images are not identical!"

    return image_iterative, image_vectorized

# Test your implemented get_channel()
assert len(get_channel(img, 0).shape) == 2  # Index 0

# Run the function
image_iterative, image_vectorized = compare()

# Plotting the results in sepearate subplots.
plt.figure(figsize=(12,4))   # Adjust the figure size.
plt.subplot(1, 3, 1)         # Create 1x3 subplots, indexing from 1
plt.imshow(img)              # Original image.

plt.subplot(1, 3, 2)
plt.imshow(image_iterative)  # Iterative operations on the image.

plt.subplot(1, 3, 3)
plt.imshow(image_vectorized) # Vectorized operations on the image.

plt.show()                   # Show the figure.


# %% [markdown]
# ## Problem 2: Coordinates vs Row-col (5 points)
# 
# We have already loaded the `SunGod.jpg` image. Its shape in numpy is $H*W*C$. Now get the color values `[r, g, b]` for -
# 1. the pixel at coordinate $(x,y) = (100, 200)$. The origin is at the top left corner of the image. The positive x axis points to the right (width direction) while the positive y axis points down (height direction).
# 2. the pixel on the $100^{th}\ row$ and $200^{th}\ column$.
# 
# Now call the `show_color` method for these pixels. Are the values/colors the same? Explain the reason for your observation.

# %%
def show_color(pixel):
    """ Function to show the color of a pixel.
    This function takes a 1D array of 3 elements (i.e. a pixel) as input."""
    plt.imshow(pixel.reshape(1,1,3))
    plt.show()

def get_pixel_coord(img, x, y):
    #### Your code start here. ####
    return img[y,x]

def get_pixel_rowcol(img, row, col):
    #### Your code start here. ####
    return img[row,col]

show_color(get_pixel_coord(img, 100, 200))
show_color(get_pixel_rowcol(img, 100, 200))


# %% [markdown]
# #### Write your obersvation here

# %% [markdown]
# The numpy array access is row major, which means it goes down 100 rows (height) then goes across 200 columns (width) with the origin starting from the top left. This is why it the color shown is red because it is the red portion of the Sun God's crown/feathers. In the case of the coordinates, the origin is also in the top left but the x refers to the horizontal axis while the y refers to the vertical axis. This means it points to a different pixel and therefore shows a different color: a light blue on the wing of the Sun God.

# %% [markdown]
# ## Problem 3: More Image Manipulation (30 points)
# 
# In this problem you will use the image `usa_map.png`. Being a color image, this image has three channels, corresponding to the primary colors of red, green and blue.
# 
# (1) Read the image. (**1 point**)
# 
# (2) Write a function to flip the original image from top to bottom. For this function, you **MUST ONLY use Array Indexing** to implement it. To receive credit, you **MUST NOT** use even a single for-loop and you **MUST NOT** use library functions (e.g. `np.flip()`) that directly flips the matrix. (**6 points**)
# 
# (3) Next, write another function to rotate the original image 90 degrees **clockwise**. You **MUST ONLY use Array Indexing** to implement this function. To receive credit, you **MUST NOT** use even a single for-loop and you **MUST NOT** use library functions (e.g. `np.rot90()`) that directly rotates the matrix. Display three images, by applying the rotation function once (i.e. 90-degree rotation), twice (i.e. 180-degree rotation) and thrice (i.e. 270-degree rotation). [**HINT:** Can you reuse the previous flip method you wrote?] (**6 points**)
# 
# (4) Read the `california_map.png` image and the corresponding `california_map_mask.png` binary mask image. (**1 point**)
# 
# (5) Given the start_x and start_y on the USA map image (corresponds to top left corner of California map image), we need to place the California cutout image onto the USA map image (using the given California mask). Also note that the California map image very closely aligns with the California state boundary of the USA map image, but not perfectly. To receive credit, you **MUST NOT** use even a single for-loop. (**6 points**)
# 
# (6) Finally, consider **4 color images** you obtained: 1 original USA map image, 1 from flipping (top to bottom), 1 from rotation (180-degree), and 1 after placing the California cutout on the USA map. (**10 points**)
# 
# Now,
# 
# * change the order of the channels of the original world map image from RGB to GBR (you can reuse problem 1 functions),
# * from the flipped image remove the green channel,
# * from the rotated image, remove the red channel and
# * from the final USA coutout image remove the blue channel.
# 
# Using these 4 images, create one single image by tiling them together. To receive credit, you **MUST NOT** use any loops. The image will have $2\times 2$ tiles making the shape of the final image $2H \times 2W \times 3$. The order in which the images are tiled does not matter. Show the tiled image.
# 
# **Note:** For all the above tasks,  to receive credits, **DO NOT** use any loops.

# %%
import numpy as np
import matplotlib.pyplot as plt
import copy

# %%
# (1) Read the image.
#### Write your code here. ####
img = plt.imread('usa_map.png')

plt.imshow(img) # Show the image after reading.
plt.show()

# %%
# (2) Flip the image from top to bottom.
def flip_img(img):
    """
    Function to mirror image from top to bottom.
    This function should return a H*W*3 array which is the flipped version of original image.
    """
    #### Write your code here. ####
    img = img[::-1, :, :]
    return img

plt.imshow(img)
plt.show()
flipped_img = flip_img(img)
plt.imshow(flipped_img)
plt.show()

# %%
# (3) Rotate image.
def flip_img_left_to_right(img):
    """
    Function to mirror image from left to right.
    This function should return a H*W*3 array which is the flipped version of original image.
    """
    #### Write your code here. ####
    img = img[:, ::-1, :]
    return img

def rotate_90(img):
    """
    Function to rotate image 90 degrees clockwise.
    This function should return a W*H*3 array which is the rotated version of original image. """
    #### Write your code here. ####
    img0 = img[:, :, 0].T
    img1 = img[:, :, 1].T
    img2 = img[:, :, 2].T
    img = np.stack([img0, img1, img2], axis=-1)
    img = flip_img_left_to_right(img)

    return img
    
plt.imshow(img)
plt.show()
rot90_img = rotate_90(img)
plt.imshow(rot90_img)
plt.show()
rot180_img = rotate_90(rotate_90(img))
plt.imshow(rot180_img)
plt.show()
rot270_img = rotate_90(rotate_90(rotate_90(img)))
plt.imshow(rot270_img)
plt.show()

# %%
# (4)Read the usa image and the binary mask image

#### Write your code here. ####

california_img = plt.imread('california_map.png')
bi_mask_img = plt.imread('california_map_mask.png')

print("California Image Size: ")
print(california_img.shape)
print("California Binary Mask Image Size: ")
print(bi_mask_img.shape)

plt.imshow(california_img)
plt.show()
plt.imshow(bi_mask_img)
plt.show()

# %%
# (5) Just place the cutout of CA at the right position on the USA map (very closely aligns with USA map California state boundary, but not perfectly)
start_x = 17
start_y = 970

final_img = copy.deepcopy(img)

#### Write your code here. ####
final_img[start_y:start_y + 1692, start_x:start_x + 1639] = california_img
x,y = np.where(bi_mask_img[:,:,0] == 0)
final_img[x + 970,y + 17] = img[x + 970,y + 17]

plt.imshow(final_img)
plt.show()

# %%
# (6) Write your code here to tile the four images and make a single image.
# You can use the img, flipped_img, rot180_img, final_img to represent the four images.
# After tiling, please display the tiled image.
#The shape of the tiled image should be 2𝐻×2𝑊×3

#### Write your code here. ####
h = img.shape[0]
w = img.shape[1]

tiles = np.zeros([2*h,2*w,3])

red = get_channel(img, 0)
green = get_channel(img, 1)
blue = get_channel(img, 2)
img_copy = merge_channels(green, blue, red)
tiles[:h,:w] += img_copy

flipped_copy = flipped_img
flipped_copy[:,:,1] = 0
tiles[:h,w:] += flipped_copy

rot180_copy = rot180_img
rot180_copy[:,:,0] = 0
tiles[h:,:w] += rot180_copy

final_img_copy = final_img
final_img_copy[:,:,2] = 0
tiles[h:,w:] += final_img_copy

plt.imshow(tiles)
plt.show()


