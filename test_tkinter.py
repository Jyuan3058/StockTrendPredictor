import tkinter as tk
from tkinter import messagebox

def test_window():
    # Create the main window
    root = tk.Tk()
    root.title("Tkinter Test")
    root.geometry("300x200")
    
    # Create a label
    label = tk.Label(root, text="If you can see this, Tkinter is working!")
    label.pack(pady=20)
    
    # Create a button
    button = tk.Button(root, text="Click Me!", 
                      command=lambda: messagebox.showinfo("Test", "Button works!"))
    button.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    try:
        test_window()
    except Exception as e:
        print(f"Error: {e}") 