import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')
# 

def draw_object_for_network(img_size, mario_s, mushroom_s, original_map_s,
                            mario_position, mushroom_position, window_extense_size=10):
    """
    Draw state images for network input (simplified version without sub_state_2)

    Parameters:
        img_size: Image dimensions (height, width)
        mario_s: Mario small template image
        mushroom_s: Mushroom small template image
        original_map_s: Original map height data
        mario_position: Current Mario position [x, y]
        mushroom_position: Mushroom position [x, y]
        window_extense_size: Window extension size

    Returns:
        state_show: Combined state image [img_size, img_size, 3]
        dynamic_ROI: Dynamic window coordinates [x_start, y_start, x_end, y_end]
    """
    # Initialize state image
    state_show = np.zeros([img_size[0], img_size[1], 3])

    # Helper function to draw objects
    def draw_object(channel, template, position):
        h, w = template.shape
        x, y = position

        # Calculate bounds
        x_start = max(x - h // 2, 0)
        x_end = min(x + h // 2, img_size[0])
        y_start = max(y - w // 2, 0)
        y_end = min(y + w // 2, img_size[1])

        # Calculate template region
        temp_x_start = max(h // 2 - (x - x_start), 0)
        temp_x_end = h // 2 + (x_end - x)
        temp_y_start = max(w // 2 - (y - y_start), 0)
        temp_y_end = w // 2 + (y_end - y)

        # Draw object
        channel[x_start:x_end, y_start:y_end] = template[temp_x_start:temp_x_end, temp_y_start:temp_y_end]

    # Draw Mario (channel 0)
    draw_object(state_show[:, :, 0], mario_s, mario_position)

    # Draw Mushroom (channel 1)
    draw_object(state_show[:, :, 1], mushroom_s, mushroom_position)

    # Map height (channel 2)
    state_show[:, :, 2] = original_map_s

    # Calculate dynamic window by extending window_extense_size pixels in all directions
    min_x = min(mario_position[0], mushroom_position[0])
    max_x = max(mario_position[0], mushroom_position[0])
    min_y = min(mario_position[1], mushroom_position[1])
    max_y = max(mario_position[1], mushroom_position[1])

    # Extend the window in all directions by window_extense_size pixels
    window_x_start = max(min_x - window_extense_size, 0)
    window_y_start = max(min_y - window_extense_size, 0)
    window_x_end = min(max_x + window_extense_size, img_size[0])
    window_y_end = min(max_y + window_extense_size, img_size[1])


    # Calculate required size with protection against too small values
    # raw_size = max(max_x - min_x, max_y - min_y)
    # required_size = min(
    #     max(raw_size + 2 * window_extense_size, 10),  # Minimum size of 10
    #     min(img_size[0], img_size[1])  # Cannot exceed image dimensions
    # )
    #
    # # Recalculate start positions if window exceeds bounds
    # if window_x_end - window_x_start < required_size:
    #     window_x_start = max(0, window_x_end - required_size)
    # if window_y_end - window_y_start < required_size:
    #     window_y_start = max(0, window_y_end - required_size)

    # Get the actual cropped region (may be smaller than required_size)
    cropped = state_show[window_x_start:window_x_end, window_y_start:window_y_end]
    actual_height, actual_width = cropped.shape[:2]

    # Create centered dynamic state
    dynamic_state = np.zeros_like(state_show)

    # Calculate center position
    center_pos_x = (img_size[0] - actual_height) // 2
    center_pos_y = (img_size[1] - actual_width) // 2

    # Place the cropped region in the center
    dynamic_state[
    center_pos_x:center_pos_x + actual_height,
    center_pos_y:center_pos_y + actual_width
    ] = cropped

    return state_show, dynamic_state


def load_templates(basepath, img_size=(100, 100)):
    """Load all required templates"""
    return (
        np.load(basepath + "Mario_s.npy"),  # mario_s
        np.load(basepath + "Mushroom_s.npy"),  # mushroom_s
        np.load(basepath + 'Map_s5.npy')  # original_map_s
    )


def simulate_mario_movement(mario_start, mushroom_pos, steps=10):
    """Generate a sequence of Mario positions moving toward the mushroom"""
    positions = []
    current_pos = np.array(mario_start)
    target_pos = np.array(mushroom_pos)

    for i in range(steps):
        # Calculate direction vector
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        # If very close, stop moving
        if distance < 2:
            break

        # Normalize direction and move 1/10th of the distance
        step_size = max(1, int(distance / 30))
        move = (direction / distance * step_size).astype(int)
        current_pos += move

        # Ensure we don't overshoot the target
        if np.all(np.sign(direction) != np.sign(target_pos - current_pos)):
            current_pos = target_pos.copy()

        positions.append(current_pos.copy())

    return positions


# Load templates
basepath = 'C:/Users/Guoming/Desktop/AOA_clean_version/map_s/'
mario_s, mushroom_s, original_map_s = load_templates(basepath)

# Set up initial positions
mario_start = [10, 50]
mushroom_pos = [80, 30]

# Generate movement path
mario_positions = simulate_mario_movement(mario_start, mushroom_pos, steps=20)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Mario Moving Toward Mushroom with Dynamic Observation Window")

# Initialize plots
global_img = ax1.imshow(np.zeros((100, 100, 3)))
ax1.set_title("Global View")
ax1.axis('off')

window_img = ax2.imshow(np.zeros((100, 100, 3)))
ax2.set_title("Observation Window")
ax2.axis('off')


def update(frame):
    # Get current Mario position
    current_mario_pos = mario_positions[frame]

    # Generate state and window
    state_show, dynamic_state = draw_object_for_network(
        img_size=(100, 100),
        mario_s=mario_s,
        mushroom_s=mushroom_s,
        original_map_s=original_map_s,
        mario_position=current_mario_pos,
        mushroom_position=mushroom_pos,
        window_extense_size=10
    )

    # Update global view
    global_img.set_array(state_show)

    # Update window view
    # window_view = np.zeros_like(state_show)
    # window_view[dynamic_ROI[0]:dynamic_ROI[2], dynamic_ROI[1]:dynamic_ROI[3]] = \
    #     state_show[dynamic_ROI[0]:dynamic_ROI[2], dynamic_ROI[1]:dynamic_ROI[3]]
    window_img.set_array(dynamic_state)

    # Add position info to titles
    ax1.set_title(f"Global View\nMario: {current_mario_pos}, Mushroom: {mushroom_pos}")
    ax2.set_title(f"Observation Window\nWindow: {dynamic_state}")

    return global_img, window_img


# Create animation
ani = FuncAnimation(fig, update, frames=len(mario_positions),
                    interval=500, blit=False, repeat=False)

# plt.tight_layout()
plt.show()