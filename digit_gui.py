import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras import layers, models


model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.load_weights("digit.weights.h5")



window = tk.Tk()
window.title("Digit Recognizer")

canvas_width = 400
canvas_height = 400

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='black')
canvas.pack()

image = Image.new("L", (canvas_width, canvas_height), color=0)
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    r = 7
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

canvas.bind("<B1-Motion>", draw_lines)

def predict_digit():
    img = image.resize((28,28))
    img = np.array(img)

    
    img = img / 255.0

    

    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Prediction: {digit}")


def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,canvas_width,canvas_height], fill=0)
    result_label.config(text="Draw a digit")

btn_predict = tk.Button(window, text="Predict", command=predict_digit)
btn_predict.pack()

btn_clear = tk.Button(window, text="Clear", command=clear_canvas)
btn_clear.pack()

result_label = tk.Label(window, text="Draw a digit", font=("Helvetica", 16))
result_label.pack()

window.mainloop()