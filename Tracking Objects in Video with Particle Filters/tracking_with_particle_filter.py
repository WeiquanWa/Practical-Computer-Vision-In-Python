import numpy as np
import cv2

# from IPython.display import clear_output, Image, display, HTML
# import ipywidgets as widgets
# import threading

# Repeatability
np.random.seed(0)

VFILENAME = "walking.mp4"

frame_height = 720
frame_width = 1280
NUM_PARTICLES = 500
VEL_RANGE = 0.5


def get_frames(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, cvframe = video.read()
        if ret:
            yield cvframe
        else:
            break
    video.release()
    yield None


def display(frame, particles, location):
    if len(particles) > 0:
        for i in range(NUM_PARTICLES):
            x = int(particles[i, 0])
            y = int(particles[i, 1])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    if len(location) > 0:
        cv2.circle(frame, location, 15, (0, 0, 255), 5)

    cv2.imshow('frame', frame)
    # stop the video if pressing q button
    if cv2.waitKey(30) & 0xFF == ord('q'):
        return True

    return False


def initialize_particles():
    particles = np.random.rand(NUM_PARTICLES, 4)
    particles = particles * np.array((frame_width, frame_height, VEL_RANGE, VEL_RANGE))
    particles[:, 2:4] -= VEL_RANGE / 2.0
    return particles


location = []
particles = initialize_particles()

for frame in get_frames(VFILENAME):
    if frame is None: break
    terminate = display(frame, particles, location)
    if terminate:
        break

cv2.destroyAllWindows()


def apply_velocity(particles):
    particles[:, 0] += particles[:, 2]
    particles[:, 1] += particles[:, 3]

    return particles


location = []
particles = initialize_particles()

for frame in get_frames(VFILENAME):
    if frame is None: break
    particles = apply_velocity(particles)

    terminate = display(frame, particles, location)
    if terminate:
        break

cv2.destroyAllWindows()


def enforce_edges(particles):
    for i in range(NUM_PARTICLES):
        particles[i, 0] = max(0, min(frame_width - 1, particles[i, 0]))
        particles[i, 1] = max(0, min(frame_height - 1, particles[i, 1]))
    return particles


location = []
particles = initialize_particles()

for frame in get_frames(VFILENAME):
    if frame is None: break
    particles = apply_velocity(particles)
    particles = enforce_edges(particles)
    terminate = display(frame, particles, location)
    if terminate:
        break

cv2.destroyAllWindows()

TARGET_COLOR = np.array((66, 63, 105))


def compute_errors(particles, frame):
    errors = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        x = int(particles[i, 0])
        y = int(particles[i, 1])
        pixel_color = frame[y, x, :]
        errors[i] = np.sum((TARGET_COLOR - pixel_color) ** 2)

    return errors


def compute_weights(errors):
    weights = np.max(errors) - errors

    weights[
        (particles[:, 0] == 0) |
        (particles[:, 0] == frame_width - 1) |
        (particles[:, 1] == 0) |
        (particles[:, 1] == frame_height - 1)] = 0

    weights = weights ** 16

    return weights


def resample(particles, weights):
    probabilities = weights / np.sum(weights)
    index_numbers = np.random.choice(
        NUM_PARTICLES,
        size=NUM_PARTICLES,
        p=probabilities)
    particles = particles[index_numbers, :]

    x = np.mean(particles[:, 0])
    y = np.mean(particles[:, 1])
    return particles, [int(x), int(y)]

#
# particles = initialize_particles()
#
# for frame in get_frames(VFILENAME):
#     if frame is None: break
#     particles = apply_velocity(particles)
#     particles = enforce_edges(particles)
#     errors = compute_errors(particles, frame)
#     weights = compute_weights(errors)
#     particles, location = resample(particles, weights)
#
#     terminate = display(frame, particles, location)
#     if terminate:
#         break
#
# cv2.destroyAllWindows()

POS_SIGMA = 0.75
VEL_SIGMA = 0.1


def apply_noise(particles):
    noise = np.concatenate((
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES, 1)),
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES, 1)),
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES, 1)),
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES, 1))
        ),
        axis=1)

    particles += noise
    return particles


particles = initialize_particles()

for frame in get_frames(VFILENAME):
    if frame is None: break

    particles = apply_velocity(particles)
    particles = enforce_edges(particles)
    errors = compute_errors(particles, frame)
    weights = compute_weights(errors)
    particles, location = resample(particles, weights)
    particles = apply_noise(particles)
    terminate = display(frame, particles, location)
    if terminate:
        break
cv2.destroyAllWindows()

#
# def compute_weights(errors):
#     weights = np.max(errors) - errors
#
#     weights[
#         (particles[:, 0] == 0) |
#         (particles[:, 0] == frame_width - 1) |
#         (particles[:, 1] == 0) |
#         (particles[:, 1] == frame_height - 1)] = 0
#
#     weights = weights ** 2
#
#     return weights

#
# def compute_weights(errors):
#     weights = np.max(errors) - errors
#
#     weights[
#         (particles[:, 0] == 0) |
#         (particles[:, 0] == frame_width - 1) |
#         (particles[:, 1] == 0) |
#         (particles[:, 1] == frame_height - 1)] = 0
#
#     weights = weights ** 8
#
#     return weights

#
# def compute_weights(errors):
#     weights = np.max(errors) - errors
#
#     weights[
#         (particles[:, 0] == 0) |
#         (particles[:, 0] == frame_width - 1) |
#         (particles[:, 1] == 0) |
#         (particles[:, 1] == frame_height - 1)] = 0
#
#     weights = weights ** 16
#
#     return weights
