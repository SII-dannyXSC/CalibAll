import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset
import math

class TfdsDataset(IterableDataset):
    DEFAULT_LAYOUT = {
        "qpos": slice(0,7),
    }

    def __init__(
        self,
        root_dir: str,
        name: str,
        split: str = "train",
        *,
        obs_image_key: str = "image",
        obs_wrist_key: str = "image",
        obs_state_key: str = "state",
        action_key: str = "None",
        language_key: str = "None",
        layout: dict | None = None,
        max_episodes: int | None = None,
        max_steps: int | None = None,
        step_stride: int = 1,
        qpos_dim: int | None = None,   # 例如 UR5 传 6；Franka 传 7；None=不截断
        strict: bool = True,
        force_tf_cpu: bool = True,     
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.name = name
        self.split = split

        self.obs_image_key = obs_image_key
        self.obs_wrist_key = obs_wrist_key
        self.obs_state_key = obs_state_key
        self.action_key = action_key
        self.language_key = language_key

        self.layout = layout or dict(self.DEFAULT_LAYOUT)
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.step_stride = int(step_stride)
        self.qpos_dim = qpos_dim
        self.strict = strict

        # 只做“配置 + 建pipeline”，不遍历、不转 numpy、不缓存
        if force_tf_cpu:
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass

        with tf.device("/CPU:0"):
            self._ds = tfds.load(
                name,
                data_dir=root_dir,
                split=split,
                shuffle_files=False,
            )

    def _episode_name(self, episode, default: str) -> str:
        try:
            fp = episode["episode_metadata"]["file_path"].numpy()
            if isinstance(fp, (bytes, bytearray)):
                return fp.decode("utf-8", errors="replace")
            return str(fp)
        except Exception:
            return default

    def _convert_episode(self, episode, ep_i: int) -> dict:
        ep_name = self._episode_name(episode, default=f"{self.name}_ep_{ep_i}")

        frames = []
        wrists = []
        qpos_seq = []
        full_state_seq = []
        action_seq = []
        lang_seq = []

        kept = 0
        for step_i, step in enumerate(episode["steps"]):
            if step_i % self.step_stride != 0:
                continue
            if self.max_steps is not None and kept >= self.max_steps:
                break
            kept += 1

            try:
                obs = step["observation"]
                
                img = obs[self.obs_image_key].numpy()  # (H,W,3) uint8
                frames.append(img)

                if self.obs_wrist_key in obs:
                    wrists.append(obs[self.obs_wrist_key].numpy())
                else:
                    wrists.append(None)

                state = obs[self.obs_state_key].numpy().astype(np.float32)
                state += [0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2, math.pi / 4]
                full_state_seq.append(state)

                qpos = state[self.layout["qpos"]].astype(np.float32)
                if self.qpos_dim is not None:
                    qpos = qpos[: self.qpos_dim]
                qpos_seq.append(qpos)

                if self.action_key in step['action']:
                    action_seq.append(step['action'][self.action_key].numpy().astype(np.float32))
                else:
                    action_seq.append(None)

                if self.language_key in step:
                    lang_seq.append(step[self.language_key].numpy())  # bytes
                else:
                    lang_seq.append(None)

            except Exception as e:
                if self.strict:
                    raise RuntimeError(f"[{self.name}][{self.split}] episode={ep_i} step={step_i} parse failed: {e}") from e
                continue

        video = np.asarray(frames, dtype=np.uint8)
        full_state = np.asarray(full_state_seq, dtype=np.float32)
        states = np.asarray(qpos_seq, dtype=np.float32)

        # wrist
        if all(w is None for w in wrists):
            wrist = None
        else:
            first = next(w for w in wrists if w is not None)
            wrist = np.asarray([(np.zeros_like(first) if w is None else w) for w in wrists], dtype=np.uint8)

        # action
        if any(a is not None for a in action_seq):
            first = next(a for a in action_seq if a is not None)
            action = np.asarray([(np.zeros_like(first) if a is None else a) for a in action_seq], dtype=np.float32)
        else:
            action = None

        # language
        language = np.asarray(lang_seq, dtype=object) if any(l is not None for l in lang_seq) else None

        return dict(
            video=video,
            wrist=wrist,
            states=states,          # 关节角 qpos
            full_state=full_state,  # 原始 state（调试用）
            action=action,
            language=language,
            name=ep_name,
        )

    def __iter__(self):
        for ep_i, episode in enumerate(self._ds):
            if self.max_episodes is not None and ep_i >= self.max_episodes:
                break
            yield self._convert_episode(episode, ep_i)


if __name__ == "__main__":
    ROOT = "/inspire/hdd/global_user/xiesicheng-253108120120/project/dzj/CalibAll/dataset/tfds/"
    NAME = "toto"

    ds = TfdsDataset(
        root_dir=ROOT,
        name=NAME,
        split="train",       # 现在用 train 也不会在 init 爆内存（逐 episode 读）
        max_episodes=3,
        max_steps=200,       # 先小一点验证；跑通再放开/设为 None
    )

    for i, ep in enumerate(ds):
        print("=" * 80)
        print("episode", i, "name:", ep["name"])
        print("video", ep["video"].shape, ep["video"].dtype)
        print("states(qpos)", ep["states"].shape, ep["states"].dtype)

