import random
import json
import copy
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
from skimage.draw import ellipse
from skimage.filters import gaussian
from skimage.util import random_noise
from scipy.interpolate import PchipInterpolator

def gen_image(img_size, num_obj, seed):
    random.seed(seed)

    img = np.ones((img_size + 10, img_size + 10), dtype=np.float32) * 0.08

    for i in range(num_obj):
        rr, cc = ellipse(random.randint(0, img_size), random.randint(0, img_size),
                         random.randint(5, 10), random.randint(5, 10), shape=img.shape)
        img[rr, cc] = random.random()

    REGIONS = 20
    REGION_SIZE = img_size // REGIONS
    OVERLAP=25
    MAX_SIGMA = 3
    MIN_SIGMA = 0.5

    for i in range(REGIONS):
        for j in range(REGIONS):
            rand_x_start = max(i * REGION_SIZE - OVERLAP, 0)
            rand_x_end = min((i + 1) * REGION_SIZE + OVERLAP, img_size)

            rand_y_start = max(j * REGION_SIZE - OVERLAP, 0)
            rand_y_end = min((j + 1) * REGION_SIZE + OVERLAP, img_size)

            img[rand_x_start:rand_x_end, rand_y_start:rand_y_end] = gaussian(img[rand_x_start:rand_x_end, rand_y_start:rand_y_end], sigma=(random.random() * (MAX_SIGMA - MIN_SIGMA)) + MIN_SIGMA)
    
    img = gaussian(img, sigma = 0.05)

    return img[5:-5, 5:-5]


def interpolate(answer):
    x = [-1, answer, 1]
    y = [-1, 0, 1]
    cs = PchipInterpolator(x, y)

    return cs.__call__


def gen_answer(lthres = 0.1, uthres = 0.8):
    ans = random.random() * 2 - 1

    while abs(ans) < lthres and abs(ans) > uthres:
        ans = random.random() * 2 - 1
    
    return ans

class Tab1:
    def __init__(self, tabControl, images):
        self.tab = ttk.Frame(tabControl)

        self.config = []

        for i in range(len(images)):
            self.config.append({
                "image": images[i],
                "mod": 0
            })
            self.config[i]["noise"] = np.zeros(self.config[i]["image"].shape)
            self.config[i]["noise"] = random_noise(self.config[i]["noise"], 'gaussian', var = 0.1, clip = True)
            self.config[i]["answer"] = gen_answer(0.2)
            self.config[i]["history"] = []

        self.current_image_index = 0

        self.top_frame = tk.Frame(self.tab)

        # Create and pack the front and back buttons
        self.back_button = tk.Button(self.top_frame, text="<", command=lambda: self.update_image(-1))
        self.back_button.pack(side=tk.LEFT)
        self.front_button = tk.Button(self.top_frame, text=">", command=lambda: self.update_image(1))
        self.front_button.pack(side=tk.RIGHT)
        # Create and pack the image number label
        self.image_number_label = tk.Label(self.top_frame, text=f"Image {self.current_image_index + 1}")
        self.image_number_label.pack()
        self.top_frame.pack()

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(self.modulate_noise() * 255).resize((512, 512)))
        self.image_label = tk.Label(self.tab, image = self.tk_image)
        self.image_label.pack()
        
        self.slider_label = tk.Label(self.tab, text = "Adjust Noise:")
        self.slider_label.pack()
        self.slider = ttk.Scale(
            self.tab,
            from_ = -1.0,
            to = 1.0,
            orient = 'horizontal',
            command = lambda value: self.update_image_function(value),
            value = 0
        )
        self.slider.pack(fill = tk.X)

        self.tab.pack()

    def return_config(self):
        conf = copy.deepcopy(self.config)
        
        for entry in conf:
            del entry["image"]
            del entry["noise"]

        return conf

    def update_image(self, step):
        self.current_image_index = (self.current_image_index + step) % 5

        self.slider.configure(value = self.config[self.current_image_index]["mod"])
        processed_image = self.modulate_noise()

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(processed_image * 255).resize((512, 512)))
        self.image_label.configure(image=self.tk_image)

        # Update the image number label
        self.image_number_label.config(text=f"Image {self.current_image_index + 1}")

    def modulate_noise(self):
        image = self.config[self.current_image_index]["image"]
        noise = self.config[self.current_image_index]["noise"]
        mod = self.config[self.current_image_index]["mod"]

        modn = interpolate(self.config[self.current_image_index]["answer"])(mod)
        if modn < 0:
            noisen = noise**(abs(modn) / 2)
        else:
            noisen = noise**(1 - (abs(modn) / 2))

        dilute_img = image + abs(modn) * noisen * (1 - image)
        dilute_img = np.clip(dilute_img, 0, 1)
        
        return dilute_img

    def update_image_function(self, value):
        self.config[self.current_image_index]["mod"] = float(value)
        processed_image = self.modulate_noise()
        self.config[self.current_image_index]["history"].append(self.config[self.current_image_index]["mod"])

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(processed_image * 255).resize((512, 512)))
        self.image_label.configure(image=self.tk_image)


class Tab2:
    def __init__(self, tabControl, images):
        self.tab = ttk.Frame(tabControl)

        self.config = []

        for i in range(len(images)):
            self.config.append({
                "image": images[i],
                "mod1": 0,
                "mod2": 0
            })
            self.config[i]["noise1"] = np.zeros(self.config[i]["image"].shape)
            self.config[i]["noise1"] = random_noise(self.config[i]["noise1"], 'gaussian', var = 0.1, clip = True)
            self.config[i]["noise2"] = np.zeros(self.config[i]["image"].shape)
            self.config[i]["noise2"] = random_noise(self.config[i]["noise2"], 'gaussian', var = 0.5, clip = True)
            self.config[i]["answer1"] = gen_answer(0.3)
            self.config[i]["answer2"] = gen_answer()
            self.config[i]["history1"] = []
            self.config[i]["history2"] = []

        self.current_image_index = 0

        self.top_frame = tk.Frame(self.tab)

        # Create and pack the front and back buttons
        self.back_button = tk.Button(self.top_frame, text="<", command=lambda: self.update_image(-1))
        self.back_button.pack(side=tk.LEFT)
        self.front_button = tk.Button(self.top_frame, text=">", command=lambda: self.update_image(1))
        self.front_button.pack(side=tk.RIGHT)
        # Create and pack the image number label
        self.image_number_label = tk.Label(self.top_frame, text=f"Image {self.current_image_index + 1}")
        self.image_number_label.pack()
        self.top_frame.pack()

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(self.modulate_noise() * 255).resize((512, 512)))
        self.image_label = tk.Label(self.tab, image = self.tk_image)
        self.image_label.pack()
        
        self.slider_label = tk.Label(self.tab, text = "Adjust Noise:")
        self.slider_label.pack()
        self.slider1 = ttk.Scale(
            self.tab,
            from_ = -1,
            to = 1,
            orient = 'horizontal',
            command = lambda value: self.update_image_function(value, self.slider2.get()),
            value = 0
        )
        self.slider1.pack(fill = tk.X)
        self.slider2 = ttk.Scale(
            self.tab,
            from_ = -1,
            to = 1,
            orient = 'horizontal',
            command = lambda value: self.update_image_function(self.slider1.get(), value),
            value = 0
        )
        self.slider2.pack(fill = tk.X)

        self.tab.pack()

    def return_config(self):
        conf = copy.deepcopy(self.config)
        
        for entry in conf:
            del entry["image"]
            del entry["noise1"]
            del entry["noise2"]

        return conf

    def update_image(self, step):
        self.current_image_index = (self.current_image_index + step) % 5

        self.slider1.configure(value = self.config[self.current_image_index]["mod1"])
        self.slider2.configure(value = self.config[self.current_image_index]["mod2"])
        processed_image = self.modulate_noise()

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(processed_image * 255).resize((512, 512)))
        self.image_label.configure(image=self.tk_image)

        # Update the image number label
        self.image_number_label.config(text=f"Image {self.current_image_index + 1}")

    def modulate_noise(self):
        image = self.config[self.current_image_index]["image"]
        noise1 = self.config[self.current_image_index]["noise1"]
        mod1 = self.config[self.current_image_index]["mod1"]
        noise2 = self.config[self.current_image_index]["noise2"]
        mod2 = self.config[self.current_image_index]["mod2"]

        modn1 = interpolate(self.config[self.current_image_index]["answer1"])(mod1)
        modn2 = interpolate(self.config[self.current_image_index]["answer2"])(mod2)

        if modn1 < 0:
            noisen1 = noise1**(abs(modn1) / 2)
        else:
            noisen1 = noise1**(1 - (abs(modn1) / 2))
        if modn2 < 0:
            noisen2 = noise2**(abs(modn2) / 2)
        else:
            noisen2 = noise2**(1 - (abs(modn2) / 2))

        dilute_img = image + (abs(modn1) * noisen1 + abs(modn2) * noisen2) * (1 - image)
        dilute_img = np.clip(dilute_img, 0, 1)
        
        return dilute_img

    def update_image_function(self, value1, value2):
        self.config[self.current_image_index]["mod1"] = float(value1)
        self.config[self.current_image_index]["mod2"] = float(value2)
        processed_image = self.modulate_noise()
        self.config[self.current_image_index]["history1"].append(self.config[self.current_image_index]["mod1"])
        self.config[self.current_image_index]["history2"].append(self.config[self.current_image_index]["mod2"])

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(processed_image * 255).resize((512, 512)))
        self.image_label.configure(image=self.tk_image)


class Tab3:
    def __init__(self, tabControl, images):
        self.tab = ttk.Frame(tabControl)

        self.config = []

        for i in range(len(images)):
            self.config.append({
                "image": np.clip(images[i] + 0.15 * random_noise(1 - images[i], 'speckle', var = 5), 0, 1),
                "mod1": 0,
                "mod2": 0
            })
            self.config[i]["noise1"] = np.zeros(self.config[i]["image"].shape)
            self.config[i]["noise1"] = random_noise(self.config[i]["noise1"], 'gaussian', var = 0.1, clip = True)
            self.config[i]["noise2"] = np.zeros(self.config[i]["image"].shape)
            self.config[i]["noise2"] = random_noise(self.config[i]["noise2"], 'gaussian', var = 0.5, clip = True)
            self.config[i]["answer1"] = gen_answer(0.3)
            self.config[i]["answer2"] = gen_answer()
            self.config[i]["history1"] = []
            self.config[i]["history2"] = []

        self.current_image_index = 0

        self.top_frame = tk.Frame(self.tab)

        # Create and pack the front and back buttons
        self.back_button = tk.Button(self.top_frame, text="<", command=lambda: self.update_image(-1))
        self.back_button.pack(side=tk.LEFT)
        self.front_button = tk.Button(self.top_frame, text=">", command=lambda: self.update_image(1))
        self.front_button.pack(side=tk.RIGHT)
        # Create and pack the image number label
        self.image_number_label = tk.Label(self.top_frame, text=f"Image {self.current_image_index + 1}")
        self.image_number_label.pack()
        self.top_frame.pack()

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(self.modulate_noise() * 255).resize((512, 512)))
        self.image_label = tk.Label(self.tab, image = self.tk_image)
        self.image_label.pack()
        
        self.slider_label = tk.Label(self.tab, text = "Adjust Noise:")
        self.slider_label.pack()
        self.slider1 = ttk.Scale(
            self.tab,
            from_ = -1,
            to = 1,
            orient = 'horizontal',
            command = lambda value: self.update_image_function(value, self.slider2.get()),
            value = 0
        )
        self.slider1.pack(fill = tk.X)
        self.slider2 = ttk.Scale(
            self.tab,
            from_ = -1,
            to = 1,
            orient = 'horizontal',
            command = lambda value: self.update_image_function(self.slider1.get(), value),
            value = 0
        )
        self.slider2.pack(fill = tk.X)

        self.tab.pack()

    def return_config(self):
        conf = copy.deepcopy(self.config)
        
        for entry in conf:
            del entry["image"]
            del entry["noise1"]
            del entry["noise2"]

        return conf

    def update_image(self, step):
        self.current_image_index = (self.current_image_index + step) % 5

        self.slider1.configure(value = self.config[self.current_image_index]["mod1"])
        self.slider2.configure(value = self.config[self.current_image_index]["mod2"])
        processed_image = self.modulate_noise()

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(processed_image * 255).resize((512, 512)))
        self.image_label.configure(image=self.tk_image)

        # Update the image number label
        self.image_number_label.config(text=f"Image {self.current_image_index + 1}")

    def modulate_noise(self):
        image = self.config[self.current_image_index]["image"]
        noise1 = self.config[self.current_image_index]["noise1"]
        mod1 = self.config[self.current_image_index]["mod1"]
        noise2 = self.config[self.current_image_index]["noise2"]
        mod2 = self.config[self.current_image_index]["mod2"]

        modn1 = interpolate(self.config[self.current_image_index]["answer1"])(mod1)
        modn2 = interpolate(self.config[self.current_image_index]["answer2"])(mod2)

        if modn1 < 0:
            noisen1 = noise1**(abs(modn1) / 2)
        else:
            noisen1 = noise1**(1 - (abs(modn1) / 2))
        if modn2 < 0:
            noisen2 = noise2**(abs(modn2) / 2)
        else:
            noisen2 = noise2**(1 - (abs(modn2) / 2))

        dilute_img = image + (abs(modn1) * noisen1 + abs(modn2) * noisen2) * (1 - image)
        dilute_img = np.clip(dilute_img, 0, 1)
        
        return dilute_img

    def update_image_function(self, value1, value2):
        self.config[self.current_image_index]["mod1"] = float(value1)
        self.config[self.current_image_index]["mod2"] = float(value2)
        processed_image = self.modulate_noise()
        self.config[self.current_image_index]["history1"].append(self.config[self.current_image_index]["mod1"])
        self.config[self.current_image_index]["history2"].append(self.config[self.current_image_index]["mod2"])

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(processed_image * 255).resize((512, 512)))
        self.image_label.configure(image=self.tk_image)



class SaveTab:
    def __init__(self, tabControl, seeds, tabs):
        self.tab = ttk.Frame(tabControl)

        self.seeds = seeds
        self.tabs = tabs

        self.top_frame = tk.Frame(self.tab)

        self.save_button = tk.Button(self.top_frame, text="Save", command=lambda: self.save())
        self.save_button.pack()

        self.top_frame.pack()

        self.tab.pack()

    def save(self):
        save_dict = {
            "seeds": self.seeds.tolist()
        }

        # change order based on experiment
        save_dict["single_slider"] = self.tabs[0].return_config()
        save_dict["double_slider"] = self.tabs[1].return_config()
        save_dict["double_slider+noise"] = self.tabs[2].return_config()

        cwd = Path.cwd().absolute()

        with open(cwd / "save.json", "w") as f:
            json.dump(save_dict, f)

class TabbedImageSliderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("HCI Noise Study")

        self.seeds = np.random.randint(0, 10000000, (5, 3))
        
        self.task_1_images = [gen_image(1024, 500, i) for i in self.seeds[:, 0]]
        self.task_2_images = [gen_image(1024, 500, i) for i in self.seeds[:, 1]]
        self.task_3_images = [gen_image(1024, 500, i) for i in self.seeds[:, 2]]

        self.tabControl = ttk.Notebook(self.master)
        self.tab1 = Tab1(self.tabControl, self.task_1_images)
        self.tabControl.add(self.tab1.tab, text="Task 1")

        self.tab2 = Tab2(self.tabControl, self.task_2_images)
        self.tabControl.add(self.tab2.tab, text="Task 2")

        self.tab3 = Tab3(self.tabControl, self.task_3_images)
        self.tabControl.add(self.tab3.tab, text="Task 3")

        self.saveTab = SaveTab(self.tabControl, self.seeds, [self.tab1, self.tab2, self.tab3])
        self.tabControl.add(self.saveTab.tab, text="Save")

        self.tabControl.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = TabbedImageSliderApp(root)
    root.mainloop()
