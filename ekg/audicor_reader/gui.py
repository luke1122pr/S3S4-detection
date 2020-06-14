import re
import datetime
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import platform
import os
if platform.system() == 'Linux':
    os.environ['TKDND_LIBRARY'] = './tkdnd/tkdnd2.9.2_linux'
elif platform.system() == 'Windows':
    os.environ['TKDND_LIBRARY'] = './tkdnd/tkdnd2.9.2_win'
else: # mac
    os.environ['TKDND_LIBRARY'] = './tkdnd/tkdnd2.9.2_mac'
from tkdnd_wrapper import TkDND

import reader

class App:
    def __init__(self):
        self.tk_root = tk.Tk()
        self.tk_root.geometry('+300+100')
        self.tk_root.title('Please drop in *.raw file to draw!')

        # has not been loaded yet
        self.signal, self.sampling_rates = None, None
        self.signal_length = -1 # in seconds

        # add signal figure
        self.time_interval = 10
        self.figure = Figure(figsize=(15, 12), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tk_root)
        self.canvas.get_tk_widget().grid(row=0, columnspan=3, sticky='NSEW')
        self.axes = None
        self.lines = None

        # set resizable
        self.tk_root.columnconfigure(0, weight=1)
        self.tk_root.rowconfigure(0, weight=100)

        # control units
        self.time_frame = tk.Frame(borderwidth=10)
        self.time_frame.grid(row=1, column=0, columnspan=8, sticky='NSEW')

        # set resizable
        self.tk_root.columnconfigure(0, weight=1)
        self.tk_root.rowconfigure(1, weight=1)

        # add rescale button
        self.rescale_button = tk.Button(self.time_frame, width=5, text='Rescale', command=self.rescale_plot)
        self.rescale_button.pack(side=tk.LEFT, padx=5)

        # add slider bar for time
        self.time_slider = tk.Scale(self.time_frame,
                                    from_=0, to=self.signal_length-self.time_interval-1,
                                    resolution=self.time_interval // 2,
                                    length=1200,
                                    showvalue='no',
                                    orient='horizontal',
                                    command=self.time_slider_callback)
        self.time_slider.pack(side=tk.LEFT, padx=5, expand=True, fill='both')

        # add time input
        self.time_box_hour = tk.Entry(self.time_frame, validate='key', width=3, validatecommand=self.get_numerical_check(), justify=tk.RIGHT)
        self.time_box_min = tk.Entry(self.time_frame, validate='key', width=3, validatecommand=self.get_numerical_check(), justify=tk.RIGHT)
        self.time_box_sec = tk.Entry(self.time_frame, validate='key', width=3, validatecommand=self.get_numerical_check(), justify=tk.RIGHT)
        self.time_box_sec.pack(sid=tk.RIGHT)
        tk.Label(self.time_frame, text=':', width=1).pack(sid=tk.RIGHT)
        self.time_box_min.pack(sid=tk.RIGHT)
        tk.Label(self.time_frame, text=':', width=1).pack(sid=tk.RIGHT)
        self.time_box_hour.pack(sid=tk.RIGHT)

        # bind enter to input boxes
        self.time_box_hour.bind('<Return>', self.time_box_callback)
        self.time_box_min.bind('<Return>', self.time_box_callback)
        self.time_box_sec.bind('<Return>', self.time_box_callback)

        # support drag and drop
        self.dnd = TkDND(self.tk_root)
        self.dnd.bindtarget(self.canvas.get_tk_widget(), self.load_data, 'text/uri-list')

        # bind left and right arrow keys to move through time
        self.tk_root.bind('<Left>', self.go_back_time)
        self.tk_root.bind('<Right>', self.go_to_future)
        self.tk_root.bind('<space>', self.rescale_plot)
        self.tk_root.bind('<Up>', self.increase_time_interval)
        self.tk_root.bind('<Down>', self.decrease_time_interval)

    def increase_time_interval(self, _):
        self.time_interval = min(3600, self.time_interval * 2)
        self.time_slider.configure(resolution=self.time_interval // 2)
        print('Time interval:', self.time_interval, 'sec')

        self.initial_plot()
        self.update_plot(self.time_slider.get())

    def decrease_time_interval(self, _):
        self.time_interval = max(10, self.time_interval // 2)
        self.time_slider.configure(resolution=self.time_interval // 2)
        print('Time interval:', self.time_interval, 'sec')

        self.initial_plot()
        self.update_plot(self.time_slider.get())

    def go_to_future(self, _):
        sec = self.time_slider.get()
        sec = min(self.signal_length-self.time_interval-1, sec+self.time_interval)
        self.time_slider.set(sec)
        self.time_slider_callback(sec)

    def go_back_time(self, _):
        sec = self.time_slider.get()
        sec = max(0, sec-self.time_interval)
        self.time_slider.set(sec)
        self.time_slider_callback(sec)

    def time_box_callback(self, _):
        hours = int(self.time_box_hour.get())
        minutes = int(self.time_box_min.get())
        seconds = int(self.time_box_sec.get())

        total_sec = hours*60*60 + minutes*60 + seconds
        total_sec = min(self.signal_length-self.time_interval-1, total_sec)

        # reset box contant
        self.time_slider.set(total_sec)
        self.set_time_of_box(int(total_sec))
        self.update_plot(int(total_sec))

    def load_data(self, event):
        filename = re.sub(r'^\{|}$', '', event.data) # remove {} in the begining or the end if any
        self.tk_root.title(filename)
        self.signal, self.sampling_rates = reader.get_heart_sounds(filename)
        self.signal_length = self.signal[0].shape[0] // self.sampling_rates[0] # in seconds
        self.time_slider.configure(to=self.signal_length-self.time_interval-1)
        self.time_slider.set(0)
        self.initial_plot()

    def rescale_plot(self, _=None):
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw()

    def get_numerical_check(self):
        def valid(action, index, value_if_allowed,
                           prior_value, text, validation_type, trigger_type, widget_name):
            return True if re.match(r'^[0-9]*$', value_if_allowed) else False

        return (self.tk_root.register(valid),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

    def set_time_of_box(self, time_in_sec):
        self.time_box_hour.delete(0, tk.END)
        self.time_box_min.delete(0, tk.END)
        self.time_box_sec.delete(0, tk.END)
        self.time_box_hour.insert(0, str(time_in_sec // 60 // 60))
        self.time_box_min.insert(0, str((time_in_sec // 60) % 60))
        self.time_box_sec.insert(0, str(time_in_sec % 60))

    def time_slider_callback(self, time_in_sec):
        self.set_time_of_box(int(time_in_sec))
        self.update_plot(int(time_in_sec))

    @staticmethod
    def sec_to_timestring(sec):
        return str(datetime.timedelta(seconds=int(sec)))

    def initial_plot(self):
        self.figure.clf()

        start_s = 0
        end_s = start_s + self.time_interval

        self.axes = list()
        self.lines = list()
        for index_channel, (channel_data, sampling_rate) in enumerate(zip(self.signal, self.sampling_rates)):
            ax = self.figure.add_subplot(self.signal.shape[0], 1, index_channel+1)

            channel_data = channel_data[ int(start_s*sampling_rate): int(end_s*sampling_rate)]
            line, = ax.plot(channel_data)
            ax.set_xticks(np.linspace(0., self.time_interval*sampling_rate, num=10))
            ax.set_xticklabels([self.sec_to_timestring(s) for s in np.linspace(start_s, end_s, num=10)])

            ax.margins(x=0, y=0)
            self.lines.append(line)
            self.axes.append(ax)

        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot(self, start_time):
        if self.signal is None:
            return

        start_s = start_time
        end_s = start_s + self.time_interval

        for index_channel, (ax, line, channel_data, sampling_rate) in enumerate(zip(self.axes, self.lines, self.signal, self.sampling_rates)):
            channel_data = channel_data[ int(start_s*sampling_rate): int(end_s*sampling_rate)]
            ax.set_xticks(np.linspace(0., self.time_interval*sampling_rate, num=10))
            ax.set_xticklabels([self.sec_to_timestring(s) for s in np.linspace(start_s, end_s, num=10)])
            line.set_ydata(channel_data)

        self.canvas.draw()

    def loop(self):
        self.tk_root.mainloop()

if __name__ == '__main__':
    App().loop()
