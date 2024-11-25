from src.GUI import StockCryptoGUI
import tkinter as tk

def main():
    root = tk.Tk()
    app = StockCryptoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
