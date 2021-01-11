import os
import tkinter as tk
from tkinter import filedialog




def walkFile():
    root = tk.Tk()
    root.withdraw()

    file = filedialog.askdirectory()
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            root_file=os.path.join(root, f)
            print(root_file)
        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))

walkFile()