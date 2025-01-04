import tkinter as tk

##################### Change this func #########################
def my_func(text, switch_state):
    print(f"Text: {text}, Switch State: {switch_state}")
################################################################

def on_button_click():
    text = text_field.get()
    switch_state = switch_var.get()
    my_func(text, switch_state)

# Create the main window
root = tk.Tk()
root.title("Basic Tkinter Application")

# Create a text field
text_field = tk.Entry(root, width=30)
text_field.pack(pady=10)

# Create a variable to hold the state of the switch
switch_var = tk.BooleanVar()
switch_var.set(False)  # Default state is OFF

# Create a switch (Checkbutton)
switch = tk.Checkbutton(root, text="Switch on Video?", variable=switch_var)
switch.pack(pady=10)

# Create a button
button = tk.Button(root, text="Submit", command=on_button_click)
button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()