import keyboard
import pickle

class Macrome:

    def __init__(self):
        self.recording = None
        with open('saved_macs', 'rb') as f:
            self.my_macs = pickle.load(f)
        self.rec_start = 'right shift'
        self.rec_mod = 'ctrl'
        self.rec_stop = 'pause'
        self.play_mod = 'alt'
        self.play_start = 'f1'
        self.play_save = 'f2'
        self.play_load = 'f3'
        self.exit_key = 'z'
        print(f'Start Recording: {self.rec_mod} + {self.rec_start}\n\
            Stop Recording: {self.rec_stop}\n\
            Use Recorded Macro: {self.play_mod} + {self.play_start}\n\
            Save Macro: {self.play_mod} + {self.play_save}\n\
            Load Macro: {self.play_mod} + {self.play_load}\n\
            Exit: {self.play_mod} + {self.exit_key}'\
        )

    def start(self):
        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_UP:
                if keyboard.is_pressed(self.rec_mod) and event.name == self.rec_start:
                    print('Recording...')
                    self.recording = keyboard.record(self.rec_stop, suppress=True)
                    print('Recording stopped.')
                elif keyboard.is_pressed(self.play_mod):
                    if event.name == self.play_start:
                        keyboard.play(self.recording, 5)
                    elif event.name == self.play_save:
                        self.save_mac()
                        with open('saved_macs', 'wb') as f:
                            pickle.dump(self.my_macs, f)
                    elif event.name == self.play_load:
                        self.load_mac()
                    elif event.name == self.exit_key:
                        return

    def save_mac(self):
        self.my_macs[input('Enter macro save name.')] = self.recording

    def load_mac(self):
        inp = input('Enter macro load name.')
        if inp not in self.my_macs:
            print(f'{inp} not found in macros.')
        else:
            self.recording = self.my_macs[inp]


def main():
    mac = Macrome()
    mac.start()

if __name__ == '__main__':
    main()