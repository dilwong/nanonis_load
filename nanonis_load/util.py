'''
This file will contain miscellaneous functions that might be useful elsewhere in the package.
'''

from tkinter import Tk

def copy_text_to_clipboard(text : str):
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(text)
    r.update()
    r.destroy()