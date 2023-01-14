import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox  
from PIL import Image, ImageTk
from index import *
from metrics import *
from features import *
from query import *
import os


directory = os.getcwd()


with open("features.json", "r") as infile:
        database = json.load(infile)

query_image_path = None
root = tk.Tk()
root.geometry("950x600")
root.resizable(False, False)
root.title("Image Retrieval System")

# Create left frame
left_frame = tk.Frame(root, width=300, height=600)
left_frame.grid(row=0, column=0, sticky="nw")


# Create separator line between left and right frames
separator = ttk.Separator(root, orient="vertical")
separator.grid(row=0, column=1, sticky="ns", padx=5)

# create the right frame with scroll bar and canvas
right_frame = tk.Frame(root, width=650, height=600)
right_frame.grid(row=0, column=2, sticky="nw")

def load_query_image():
    global query_image_path
    query_image_path = filedialog.askopenfilename(title="Select Query Image")
    # Open the image and display it in the query image preview
    query_image = Image.open(query_image_path)
    query_image = query_image.resize((150, 150), Image.Resampling.LANCZOS)
    query_image = ImageTk.PhotoImage(query_image)
    query_preview.config(image=query_image, padx=10, pady=10)
    query_preview.image = query_image

# Create button for loading query image in left frame
load_button = tk.Button(left_frame, text="Load Query Image", command=load_query_image)
load_button.grid(row=0, column=0, padx=10, pady=10)

# Create label for previewing query image in left frame
query_preview = tk.Label(left_frame, text="Query Image Preview", padx=19, pady=68, relief="groove")
query_preview.grid(row=1, column=0, padx=10, pady=10)

# Create a label for the color space option menu in left frame
color_space_label = tk.Label(left_frame, text="Color Space :")
color_space_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")   

# Create option menu for choosing color space in left frame
color_space_var = tk.StringVar(value="RGB")
color_space_dropdown = tk.OptionMenu(left_frame, color_space_var, "RGB", "HSV", "YCRCB")
color_space_dropdown.grid(row=2, column=0, padx=10, pady=10, sticky="e")

# Create slider for color space weight in left frame
color_space_weight = tk.Scale(left_frame, from_=0, to=100, orient="horizontal")
color_space_weight.set(0.5)
color_space_weight.grid(row=3, column=0, padx=10, pady=10)

# Create a label for the texture descriptor option menu in left frame
texture_desc_label = tk.Label(left_frame, text="Texture :")
texture_desc_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

# Create option menu for choosing texture descriptor in left frame
texture_desc_var = tk.StringVar(value="LBP")
texture_desc_dropdown = tk.OptionMenu(left_frame, texture_desc_var, "LBP", "GLCM", "LPQ")
texture_desc_dropdown.grid(row=4, column=0, padx=10, pady=10, sticky="e")

# Create slider for texture descriptor weight in left frame
texture_desc_weight = tk.Scale(left_frame, from_=0, to=100, orient="horizontal")
texture_desc_weight.set(0.5)
texture_desc_weight.grid(row=5, column=0, padx=10, pady=10)

# Create a label for the shape descriptor option menu in left frame
shape_desc_label = tk.Label(left_frame, text="Shape :")
shape_desc_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")

# Create option menu for choosing shape descriptor in left frame
shape_desc_var = tk.StringVar(value="LOG")
shape_desc_dropdown = tk.OptionMenu(left_frame, shape_desc_var, "LOG", "SOBEL", "HU")
shape_desc_dropdown.grid(row=6, column=0, padx=10, pady=10, sticky="e")

# Create slider for shape descriptor weight in left frame
shape_desc_weight = tk.Scale(left_frame, from_=0, to=100, orient="horizontal")
shape_desc_weight.set(0.5)
shape_desc_weight.grid(row=7, column=0, padx=10, pady=10)

methods_dict = {"RGB": RGB, "HSV": HSV, "YCRCB": YCRCB, "LBP": LBP, "GLCM": GLCM, "LPQ": LPQ, "LOG": LOG, "SOBEL": SOBEL, "HU": HU}

def search():
    # delete all the images in the right frame
    for widget in right_frame.winfo_children():
        widget.destroy()
    if query_image_path is None:
        messagebox.showerror("Error", "Please load a query image.")
        return
    if color_space_weight.get() + texture_desc_weight.get() + shape_desc_weight.get() != 100:
        messagebox.showerror("Error", "Please make sure the weights add up to 100.")
        return
    color_space = color_space_var.get()
    texture_desc = texture_desc_var.get()
    shape_desc = shape_desc_var.get()
    color_space_w = color_space_weight.get()/100
    texture_desc_w = texture_desc_weight.get()/100
    shape_desc_w = shape_desc_weight.get()/100
    similar_images = queryDatabase(os.path.relpath(query_image_path, directory),database, methods_dict[color_space], methods_dict[texture_desc], methods_dict[shape_desc], [color_space_w, texture_desc_w, shape_desc_w], CosineDist)
    similar_images = sorted(similar_images.items(), key=lambda x: x[1])

    for (i, resultID) in enumerate(similar_images):
        if i==20:
            break
        # add a label on top of the image
        distance = "Distance:"+str(round(resultID[1], 2))
        similarity = "Similarity:"+str(round(similarityPercentage(resultID[1], similar_images[-1][1]),2))+"%"
        result_label = ttk.Label(right_frame, text= distance+" | " + similarity)
        result_label.grid(row=i//4, column=i%4, sticky="n", padx=10, pady=1)

        # load the result image and display it
        result = Image.open(resultID[0])
        result = result.resize((110, 110), Image.Resampling.LANCZOS)
        result = ImageTk.PhotoImage(result)
        result_preview = ttk.Label(right_frame, image=result)
        result_preview.image = result
        result_preview.grid(row=i//4, column=i%4, sticky="s", padx=10, pady=20)
    

# Create search button in left frame
search_button = tk.Button(left_frame, text="Search", bg="green", command=search)
search_button.grid(row=8, column=0, padx=10, pady=10, sticky="w")

# Create close button in left frame
close_button = tk.Button(left_frame, text="Close", command=root.destroy)
close_button.grid(row=8, column=0, padx=10, pady=10, sticky="e")

def update_sliders(value, slider):
    try:
        total = color_space_weight.get() + texture_desc_weight.get() + shape_desc_weight.get()
        if total > 100:
            if slider == "color":
                remaining = 100 - color_space_weight.get()
                texture_desc_weight.set(remaining * texture_desc_weight.get() / (shape_desc_weight.get() + texture_desc_weight.get()))
                shape_desc_weight.set(remaining * shape_desc_weight.get() / (shape_desc_weight.get() + texture_desc_weight.get()))
            elif slider == "texture":
                remaining = 100 - texture_desc_weight.get()
                color_space_weight.set(remaining * color_space_weight.get() / (shape_desc_weight.get() + color_space_weight.get()))
                shape_desc_weight.set(remaining * shape_desc_weight.get() / (shape_desc_weight.get() + color_space_weight.get()))
            else:
                remaining = 100 - shape_desc_weight.get()
                color_space_weight.set(remaining * color_space_weight.get() / (texture_desc_weight.get() + color_space_weight.get()))
                texture_desc_weight.set(remaining * texture_desc_weight.get() / (texture_desc_weight.get() + color_space_weight.get()))
    except ZeroDivisionError:
        pass
color_space_weight = tk.Scale(left_frame, from_=0, to=100, orient="horizontal", variable=tk.IntVar(), command=lambda x: update_sliders(x, "color"))
color_space_weight.set(0)
color_space_weight.grid(row=3, column=0, padx=10, pady=10)

texture_desc_weight = tk.Scale(left_frame, from_=0, to=100, orient="horizontal", variable=tk.IntVar(), command=lambda x: update_sliders(x, "texture"))
texture_desc_weight.set(0)
texture_desc_weight.grid(row=5, column=0, padx=10, pady=10)

shape_desc_weight = tk.Scale(left_frame, from_=0, to=100, orient="horizontal", variable=tk.IntVar(), command=lambda x: update_sliders(x, "shape"))
shape_desc_weight.set(0)
shape_desc_weight.grid(row=7, column=0, padx=10, pady=10)


root.mainloop()
