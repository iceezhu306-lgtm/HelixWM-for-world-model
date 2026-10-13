# Source this file in an interactive shell. This does not start training.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/helixwm/env
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
cd /root/autodl-tmp/helixwm/source
