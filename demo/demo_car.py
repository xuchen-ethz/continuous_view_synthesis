from options.test_options import TestOptions
from demo.demo_base import Demo
import numpy as np
import copy

class DemoCar(Demo):

    def on_press(self, event):
        self.press = event.xdata, event.ydata
        self.is_pressed = True
        self.pose_cur = \
            self.pose.astype(np.float)

    def on_release(self, event):
        self.press = None
        self.is_pressed = False
        self.pose = copy.copy(self.pose_cur)

    def on_motion(self, event):
        if not self.is_pressed: return

        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.pose_cur[1] = self.pose[1] + dx/64/10
        # self.pose_cur[0] = min(self.pose_cur[0],np.pi*1.2/18)
        # self.pose_cur[0] = max(self.pose_cur[0],-np.pi*1.2/18)

        self.pose_cur[0] = self.pose[0] - dy/192/10
        # self.pose_cur[1] = min(self.pose_cur[1], np.pi * .5 / 18)
        # self.pose_cur[1] = max(self.pose_cur[1], -np.pi *.5 / 18)


if __name__ == "__main__":
    opt = TestOptions().parse()
    demo = DemoCar(opt)
