import vest
import keyboard

def fwd():
    bhaptics_manager.submit_registered("forward")
    print("pressed forward")

def bwd():
    bhaptics_manager.submit_registered("backward")
    print("pressed backward")

def right():
    bhaptics_manager.submit_registered("right")
    print("pressed right")

def left():
    bhaptics_manager.submit_registered("left")
    print("pressed left")


if __name__ == "__main__":
    bhaptics_manager = vest.BhapticsManager()
    bhaptics_manager.register("forward", "assets/tacts/FF.tact ")
    bhaptics_manager.submit_registered("forward")
    bhaptics_manager.register("backward", "assets/tacts/BB.tact")
    bhaptics_manager.submit_registered("backward")
    bhaptics_manager.register("right", "assets/tacts/RR.tact")
    bhaptics_manager.submit_registered("right")
    bhaptics_manager.register("left", "assets/tacts/LL.tact")
    bhaptics_manager.submit_registered("left")

    keyboard.on_press_key("up", lambda _: fwd())
    keyboard.on_press_key("down", lambda _: bwd())
    keyboard.on_press_key("right", lambda _: right())
    keyboard.on_press_key("left", lambda _: left())

    print("Press ESC to stop.")
    keyboard.wait("esc")