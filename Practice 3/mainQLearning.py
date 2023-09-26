import numpy as np
import keyboard


if __name__ == '__main__':

    ambient = np.zeros((4, 12))
    print(ambient)
    pos_x = 0
    pos_y = 0

    action = False

    while True:
        if keyboard.is_pressed('esc'):
            print("Escape key pressed. Exiting Loop!!")
            break
        elif keyboard.is_pressed('up') and not action:
            print("Moving up!")
            pos_x += 1
            action = True
        elif keyboard.is_pressed('down') and not action:
            print("Moving down!")
            pos_x -= 1
            action = True
        elif keyboard.is_pressed('right') and not action:
            print("Moving right!")
            pos_y += 1
            action = True
        elif keyboard.is_pressed('left') and not action:
            print("Moving left!")
            pos_y -= 1
            action = True
        elif (not keyboard.is_pressed('up') and not keyboard.is_pressed('down')
              and not keyboard.is_pressed('left') and not keyboard.is_pressed('right')):
            action = False
        else:
            pass


