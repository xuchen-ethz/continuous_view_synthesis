from options.test_options import TestOptions
from demo.demo_base import Demo

class DemoKITTI(Demo):

    def on_press(self, event):
        self.press = event.xdata, event.ydata
        self.is_pressed = True

    def on_motion(self, event):
        if not self.is_pressed: return

        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        self.pose_cur[1] = self.pose_cur[1] + dx/3000
        self.pose_cur[0] = self.pose_cur[0] - dy/6000
        self.press = event.xdata, event.ydata

    def on_key_press(self, event):
        if event.key == 'w':
            self.pose_cur[5] += 0.0005
        elif event.key == 's':
            self.pose_cur[5] -= 0.0005
        elif event.key == 'a':
            self.pose_cur[3] -= 0.0002
        elif event.key == 'd':
            self.pose_cur[3] += 0.0002
        else:
            return

if __name__ == "__main__":
    opt = TestOptions().parse()
    demo = DemoKITTI(opt)
