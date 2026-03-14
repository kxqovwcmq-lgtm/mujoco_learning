import time
import math
from enum import Enum

import mujoco
import mujoco.viewer


# =========================
# 配置区
# =========================
MODEL_PATH = "../../API-MJCF/pointer.xml"
RUN_TIME = 30.0                 # 运行总时长（秒）
CTRL_FREQ = 1.0                 # 正弦控制频率（Hz）
CTRL_AMP = 1.0                  # 正弦控制幅值
CTRL_INDEX = 1                  # 要控制的 actuator 索引
QPOS_INDEX = 0
QVEL_INDEX = 0
QACC_INDEX = 0
SHOW_CONTACT_POINTS = True


class SimMode(Enum):
    STEP = "step"
    STEP1_STEP2 = "step1_step2"
    FORWARD = "forward"
    INVERSE = "inverse"


SIM_MODE = SimMode.STEP


# =========================
# 工具函数
# =========================
def sinusoid(t: float, amp: float = 1.0, freq: float = 1.0) -> float:
    """基于仿真时间生成正弦信号。"""
    return amp * math.sin(2.0 * math.pi * freq * t)


def validate_indices(model: mujoco.MjModel):
    """检查索引是否合法，避免运行时报错。"""
    if model.nu <= CTRL_INDEX and SIM_MODE in {SimMode.STEP, SimMode.STEP1_STEP2, SimMode.FORWARD}:
        raise IndexError(
            f"CTRL_INDEX={CTRL_INDEX} 超出 actuator 数量范围，当前 model.nu={model.nu}"
        )

    if model.nq <= QPOS_INDEX and SIM_MODE in {SimMode.FORWARD, SimMode.INVERSE}:
        raise IndexError(
            f"QPOS_INDEX={QPOS_INDEX} 超出 qpos 范围，当前 model.nq={model.nq}"
        )

    if model.nv <= QVEL_INDEX and SIM_MODE == SimMode.INVERSE:
        raise IndexError(
            f"QVEL_INDEX={QVEL_INDEX} 超出 qvel 范围，当前 model.nv={model.nv}"
        )

    if model.nv <= QACC_INDEX and SIM_MODE == SimMode.INVERSE:
        raise IndexError(
            f"QACC_INDEX={QACC_INDEX} 超出 qacc 范围，当前 model.nv={model.nv}"
        )


def run_step(model: mujoco.MjModel, data: mujoco.MjData, t: float):
    """标准一步仿真：写 ctrl -> mj_step。"""
    data.ctrl[CTRL_INDEX] = sinusoid(t, CTRL_AMP, CTRL_FREQ)
    mujoco.mj_step(model, data)


def run_step1_step2(model: mujoco.MjModel, data: mujoco.MjData, t: float):
    """拆分 step1 / step2，适合中间插入控制逻辑。"""
    mujoco.mj_step1(model, data)
    data.ctrl[CTRL_INDEX] = sinusoid(t, CTRL_AMP, CTRL_FREQ)
    mujoco.mj_step2(model, data)


def run_forward(model: mujoco.MjModel, data: mujoco.MjData, t: float):
    """
    手动修改状态后调用 forward。
    注意：forward 不推进时间，只重新计算动力学相关量。
    """
    data.ctrl[CTRL_INDEX] = sinusoid(t, CTRL_AMP, CTRL_FREQ)
    data.qpos[QPOS_INDEX] = sinusoid(t, CTRL_AMP, CTRL_FREQ)
    mujoco.mj_forward(model, data)

    print(
        f"[FORWARD] time={data.time:.4f}, "
        f"qpos={data.qpos[QPOS_INDEX]:.4f}, "
        f"qvel={data.qvel[QVEL_INDEX]:.4f}, "
        f"qacc={data.qacc[QACC_INDEX]:.4f}"
    )


def run_inverse(model: mujoco.MjModel, data: mujoco.MjData, t: float):
    """
    给定 qpos/qvel/qacc，调用 inverse 求所需广义力。
    注意：inverse 也不推进时间。
    """
    data.qpos[QPOS_INDEX] = 0.0
    data.qvel[QVEL_INDEX] = 0.0
    data.qacc[QACC_INDEX] = sinusoid(t, CTRL_AMP, CTRL_FREQ)
    mujoco.mj_inverse(model, data)

    print(
        f"[INVERSE] time={data.time:.4f}, "
        f"qacc={data.qacc[QACC_INDEX]:.4f}, "
        f"qfrc_inverse={data.qfrc_inverse[QACC_INDEX]:.4f}"
    )


def simulate_once(model: mujoco.MjModel, data: mujoco.MjData, mode: SimMode):
    """根据模式执行一次仿真/计算。"""
    t = data.time

    if mode == SimMode.STEP:
        run_step(model, data, t)
    elif mode == SimMode.STEP1_STEP2:
        run_step1_step2(model, data, t)
    elif mode == SimMode.FORWARD:
        run_forward(model, data, t)
    elif mode == SimMode.INVERSE:
        run_inverse(model, data, t)
    else:
        raise ValueError(f"不支持的模式: {mode}")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    validate_indices(model)

    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Simulation mode: {SIM_MODE.value}")
    print(f"timestep: {model.opt.timestep}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()

        while viewer.is_running() and (time.time() - wall_start < RUN_TIME):
            step_wall_start = time.time()

            simulate_once(model, data, SIM_MODE)

            with viewer.lock():
                if SHOW_CONTACT_POINTS:
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2 < 1)

            viewer.sync()

            # 仅对真正推进仿真时间的模式做 sleep 控制
            if SIM_MODE in {SimMode.STEP, SimMode.STEP1_STEP2}:
                elapsed = time.time() - step_wall_start
                remaining = model.opt.timestep - elapsed
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    main()
