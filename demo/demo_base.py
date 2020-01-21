import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter.filedialog import askopenfilename
import tkinter; tkinter.Tk().withdraw()

from models.base_model import BaseModel
import numpy as np
import cv2

class Demo():
    def __init__(self, opt):
        self.opt = opt
        self.model = BaseModel(opt)
        self.load_image()
        self.predict()
        self.init_plot()

    def init_plot(self):

        width = 1024  # pixels
        height = 1024
        margin = 50  # pixels
        dpi = 170.  # dots per inch
        figsize = ((width + 10 * margin) / dpi, (height + 2 * margin) / dpi)  # inches
        left = 5 * margin / dpi / figsize[0]  # axes ratio
        bottom = margin / dpi / figsize[1]

        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.fig.subplots_adjust(left=left, bottom=bottom, right=1. - left, top=1. - bottom)

        plt.axis('off')
        plt.rcParams['keymap.save'] = ''
        # input image
        self.ax_in_img = plt.axes()
        self.ax_in_img.axis('off')
        self.im_input = plt.imshow(self.output, animated=True)

        self.ax_next = plt.axes([0.05, 0.1, 0.15, 0.04])
        button_next = Button(self.ax_next, 'Load image', color='lightgray', hovercolor='0.975')
        button_next.on_clicked(self.load_image_pressed)

        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidzoom = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cidkpress = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.toolbar.set_cursor = lambda cursor: None  # disable the annoying busy cursor
        self.is_pressed = False

        tkagg.defaultcursor = "crosshair"
        self.fig.canvas.toolbar.set_cursor(1)
        self.fig.canvas.toolbar.set_cursor = lambda cursor: None  # disable the annoying busy cursor

        self.timer = self.fig.canvas.new_timer(interval=50, callbacks=[(self.on_timer, [], {})])
        self.timer.start()

        plt.show()

    def on_scroll(self,event):
        # get the current x and y limits
        cur_xlim = self.ax_in_img.get_xlim()
        cur_ylim = self.ax_in_img.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        if event.button == 'up':
            scale_factor = 1 / 1.05
        elif event.button == 'down':
            scale_factor = 1.05
        else:
            scale_factor = 1
        self.ax_in_img.set_xlim([512 - cur_xrange * scale_factor,
                     512 + cur_xrange * scale_factor])
        self.ax_in_img.set_ylim([512 - cur_yrange * scale_factor,
                     512 + cur_yrange * scale_factor])
        plt.draw()  # force re-draw

    def on_press(self, event):
        pass

    def on_motion(self, event):
        pass

    def on_release(self, event):
        pass

    def on_key_press(self, event):
        pass

    def on_timer(self):
        self.predict()
        self.update_figure()

    def load_image_pressed(self,event):
        self.load_image()
        self.predict()
        self.update_figure()

    def load_image(self):
        filename = askopenfilename()
        self.data = {}
        self.image = cv2.imread(filename)
        self.pose = np.array([0, 0, 0, 0, 0, 0])
        self.pose_cur = self.pose.astype(np.float)
        self.z = None
        self.predict()

    def update_figure(self):
        self.im_input.set_array(self.output)
        self.fig.canvas.draw_idle()

    def predict(self):
        self.output = self.model.get_high_res(self.image,self.pose_cur,self.z)