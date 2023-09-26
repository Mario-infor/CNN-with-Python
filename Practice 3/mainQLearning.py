import numpy as np
import keyboard
import os


def clear_previous_move(x, y, ambient_matrix):
    ambient_matrix[x, y] = 0


def move(x, y, ambient_matrix):
    ambient_matrix[x, y] = 1
    print(ambient_matrix)


if __name__ == '__main__':

    ambient = np.zeros((4, 12))
    ambient[0, 0] = 1
    print(ambient)
    pos_x = 0
    pos_y = 0

    action = False

    while True:
        if keyboard.is_pressed('esc'):
            print("Escape key pressed. Exiting Loop!!")
            break
        elif keyboard.is_pressed('up') and not action:
            if pos_x - 1 >= 0:
                print("Moving down!")
                clear_previous_move(pos_x, pos_y, ambient)
                pos_x -= 1
                move(pos_x, pos_y, ambient)
                action = True
        elif keyboard.is_pressed('down') and not action:
            if pos_x + 1 < ambient.shape[0]:
                print("Moving down!")
                clear_previous_move(pos_x, pos_y, ambient)
                pos_x += 1
                move(pos_x, pos_y, ambient)
                action = True
        elif keyboard.is_pressed('right') and not action:
            if pos_y + 1 < ambient.shape[1]:
                print("Moving right!")
                clear_previous_move(pos_x, pos_y, ambient)
                pos_y += 1
                move(pos_x, pos_y, ambient)
                action = True
        elif keyboard.is_pressed('left') and not action:
            if pos_y - 1 >= 0:
                print("Moving left!")
                clear_previous_move(pos_x, pos_y, ambient)
                pos_y -= 1
                move(pos_x, pos_y, ambient)
                action = True
        elif (not keyboard.is_pressed('up') and not keyboard.is_pressed('down')
              and not keyboard.is_pressed('left') and not keyboard.is_pressed('right')):
            action = False
        else:
            pass


