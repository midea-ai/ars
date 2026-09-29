import cv2
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.controllers.utils.delta_pose import get_numpy, to_numpy
from PIL import Image

class VLADataCollector:
    def __init__(self, env: BaseEnv, camera_name: str, is_image_encode: bool = False, *args, **kwargs,):
        self.env = env.unwrapped
        self.camera_name = camera_name
        self.is_image_encode = is_image_encode

        self.data_dict = self.get_empty_data_dict()


    def get_empty_data_dict(self):
        data_dict = {
            "is_image_encode": self.is_image_encode,
            "image": [],
            "state": [],
            "instruction": None,
            "wrong_instruction": None,
            "mismatch_type": None,
            "action": [],
            "info": [],
        }
        return data_dict

    def clear_data(self):
        """Clear all collected data."""
        self.data_dict = self.get_empty_data_dict()

    def get_data(self):
        return to_numpy(self.data_dict, self.env.unwrapped.device)

    def save_data(self, save_path, is_compressed=False):
        """Save data as .npy file with dictionary structure."""
        saving_data = to_numpy(self.data_dict, self.env.unwrapped.device)
        saving_data["image"] = [Image.fromarray(im).convert("RGB") for im in saving_data["image"]]
        if is_compressed:
            np.savez_compressed(save_path, saving_data)
            print(f"save data at {save_path}.npz.")
        else:
            np.save(save_path, saving_data)
            print(f"save data at {save_path}.npy.")
        self.clear_data()

    def update_instruction(self):
        if self.data_dict["instruction"] == None:
            self.data_dict["instruction"] = self.env.get_language_instruction()
            mismatch = self._build_mismatch_instruction()
            if mismatch is not None:
                self.data_dict["wrong_instruction"] = mismatch["wrong_instruction"]
                self.data_dict["mismatch_type"] = mismatch["mismatch_type"]
        else:
            return

    def _build_mismatch_instruction(self):
        """Return a task-mismatch instruction for supported distractor envs."""

        env_name = self._env_name()
        if "MultiCarrot" not in env_name and "MultiPlate" not in env_name:
            return None
        if not hasattr(self.env, "select_extra_ids"):
            return None

        obj_ids = self._tensor_ids(self.env.select_carrot_ids)
        plate_ids = self._tensor_ids(self.env.select_plate_ids)
        extra_ids = self._tensor_ids(self.env.select_extra_ids)

        wrong_instruction = []
        if "MultiCarrot" in env_name:
            for obj_id, plate_id, extra_id in zip(obj_ids, plate_ids, extra_ids):
                del obj_id  # The correct source object remains encoded in data["instruction"].
                plate_name = self.model_db_plate_name(plate_id)
                extra_obj_name = self.model_db_carrot_name(extra_id)
                wrong_instruction.append(f"put {extra_obj_name} on {plate_name}")
            return {
                "wrong_instruction": wrong_instruction,
                "mismatch_type": "object",
            }

        if "MultiPlate" in env_name:
            for obj_id, plate_id, extra_id in zip(obj_ids, plate_ids, extra_ids):
                del plate_id  # The correct target plate remains encoded in data["instruction"].
                obj_name = self.model_db_carrot_name(obj_id)
                extra_plate_name = self.model_db_plate_name(extra_id)
                wrong_instruction.append(f"put {obj_name} on {extra_plate_name}")
            return {
                "wrong_instruction": wrong_instruction,
                "mismatch_type": "plate",
            }

        return None

    def _env_name(self):
        """Read a stable env identifier without depending on one wrapper shape."""

        spec = getattr(self.env, "spec", None)
        spec_id = getattr(spec, "id", None)
        if spec_id:
            return str(spec_id)
        return type(self.env).__name__

    def _tensor_ids(self, value):
        """Convert tensor/list ids to Python ints for model-db indexing."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return [int(item) for item in np.asarray(value).reshape(-1).tolist()]

    def model_db_carrot_name(self, model_id: int) -> str:
        """Return the display name for a selected source object id."""

        model_key = self.env.carrot_names[int(model_id)]
        return self.env.model_db_carrot[model_key]["name"]

    def model_db_plate_name(self, model_id: int) -> str:
        """Return the display name for a selected target plate id."""

        model_key = self.env.plate_names[int(model_id)]
        return self.env.model_db_plate[model_key]["name"]


    # should run before env.step()
    def update_image(self, camera_name: str=None):
        if camera_name==None:
            rgb = self.env.render().squeeze(0).to(torch.uint8)
        else:
            rgb = self.env.get_obs()['sensor_data'][camera_name]['rgb'].squeeze(0).to(torch.uint8)
        if self.is_image_encode:
            success, encoded_rgb = cv2.imencode('.jpeg', get_numpy(rgb,self.env.unwrapped.device), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("JPEG encode error.")
            img_bytes = np.frombuffer(encoded_rgb.tobytes(), dtype=np.uint8)
            rgb = img_bytes

        self.data_dict["image"].append(rgb)

    # should run before env.step()
    def update_action(self, action):
        self.data_dict['action'].append(action)

    # should run before env.step()
    def update_state(self):
        # Export WidowX joint state as an 8D qpos vector:
        # 6 arm joint positions + 2 gripper joint positions.
        qpos = self.env.agent.robot.get_qpos().squeeze(0).to(torch.float32)
        if qpos.shape[-1] != 8:
            raise ValueError(f"Expected 8D WidowX qpos state, got shape {tuple(qpos.shape)}.")
        self.data_dict["state"].append(qpos)

    def updata_info(self,):
        info = self.env.get_info()
        self.data_dict['info'].append(info)

    # should run before env.step()
    def update_data_dict(self, action):
        self.update_instruction()
        self.update_image(self.camera_name)
        self.update_state()
        self.updata_info()
        self.update_action(action.astype(np.float32))
