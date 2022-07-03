from functions import *# запускать необходимо этот файл
from tkinter import *
from tkinter import filedialog
import os

root = Tk()
root.geometry("550x300+300+150")
root.resizable(width=True, height=True)


def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename


def open_img():
    # x = openfn()
    # names = findfiles()
    # img = Image.open(x)
    name = message1.get()
    print(name)
    res = sq_method(name)
    print(res)


message2 = StringVar()
message1 = StringVar()
message_entry = Entry(textvariable=message1).pack()

label = Label(root, textvariable=message2, relief=RAISED).pack
btn = Button(root, text='Пуск', command=open_img).pack()

root.mainloop()
