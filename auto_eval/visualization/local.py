"""
show the current image of the robot through local display
"""
import cv2


def display_img(img, language_instruction, wait_time=0):
    cv2.imshow(
        f"image: {language_instruction}",
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),  # cv2 reads in BGR
    )
    if wait_time > 0:
        # need this to force the display to update
        # however, this could slow down operation (e.g. teleop.py should not use this)
        cv2.waitKey(1)
    # # capture "r" key and reset
    # if cv2.waitKey(10) & 0xFF == ord("r"):
    #     break
