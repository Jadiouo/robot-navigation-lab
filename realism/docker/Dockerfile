# Realism Phase 4 reproducible stack: ROS 2 Jazzy + Gazebo Harmonic + Nav2 (AMCL), headless, software rendering.
# Build context = realism/docker/ (only this file is needed; phase4 code and src/navlab are bind-mounted at run time).
FROM ros:jazzy
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-ros-gz \
    ros-jazzy-nav2-minimal-tb3-sim ros-jazzy-nav2-minimal-tb4-sim \
    ros-jazzy-nav2-simple-commander ros-jazzy-tf2-ros-py \
    python3-numpy python3-scipy python3-yaml python3-psutil \
    libegl1 libgl1-mesa-dri libglx-mesa0 libegl-mesa0 mesa-vulkan-drivers xvfb xauth \
    && rm -rf /var/lib/apt/lists/*
# nothing is copied in; run_in_docker.sh mounts code read-only
WORKDIR /work
